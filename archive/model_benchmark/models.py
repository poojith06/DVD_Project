"""All regression models of the benchmark. Each predicts log10|I| of id, ig, is, ib from the 20 standardised features.

Tuning rule (same for every model): try each setting of a small grid, fit on TRAIN, and for every current keep the
setting with the lowest VALIDATION mean absolute error in decades. Boosting models also stop early on validation
(like the MLP keeps its best validation epoch).
"""
import time
import numpy as np, torch
from torch import nn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping
from catboost import CatBoostRegressor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
mae_dec = lambda a, b: float(np.mean(np.abs(a - b)))


class GridModel:
    """Small grid search; per current keep the best setting on validation. Subclasses define grid() and fit_one()."""
    multi = False            # True: one fit predicts all 4 currents

    def __init__(self, case, polarity):
        self.case, self.polarity, self.info = case, polarity, {}

    def grid(self): return [{}]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva): raise NotImplementedError
    def pred_one(self, est, X, j): return est.predict(X)[:, j] if self.multi else est.predict(X)

    def fit(self, data):
        Xtr, Ytr, _ = data["train"]; Xva, Yva, _ = data["validation"]
        best = [(np.inf, None, None)] * 4
        for cfg in self.grid():
            if self.multi:
                est = self.fit_one(cfg, Xtr, Ytr, Xva, Yva); ests = [est] * 4
            else:
                ests = [self.fit_one(cfg, Xtr, Ytr[:, j], Xva, Yva[:, j]) for j in range(4)]
            for j in range(4):
                e = mae_dec(self.pred_one(ests[j], Xva, j), Yva[:, j])
                if e < best[j][0]: best[j] = (e, cfg, ests[j])
        self.best = best
        self.info = {c: {**b[1], "val_mae_dec": round(b[0], 5)} for c, b in zip(["id", "ig", "is", "ib"], best)}
        return self

    def predict(self, X, df=None):
        return np.column_stack([self.pred_one(self.best[j][2], X, j) for j in range(4)])


# ---------------- linear / polynomial ----------------
class RidgePoly(GridModel):
    """Ridge (squared loss + L2) on polynomial features of degree 1-4, exact solve via the eigen-decomposition of the
    smaller Gram matrix, so every alpha is cheap. Degree and alpha picked per current on validation."""
    multi = True

    def grid(self):
        return [{"degree": d, "alpha": a} for d in (1, 2, 3, 4) for a in (1e-8, 1e-6, 1e-4, 1e-2, 1, 100)]

    def fit(self, data):
        Xtr, Ytr, _ = data["train"]; Xva, Yva, _ = data["validation"]
        self.solutions, best = {}, [(np.inf, None)] * 4
        for d in (1, 2, 3, 4):
            pf = PolynomialFeatures(d, include_bias=False).fit(Xtr); sc = StandardScaler().fit(pf.transform(Xtr))
            Z = sc.transform(pf.transform(Xtr)); Zv = sc.transform(pf.transform(Xva)); ym = Ytr.mean(0); Yc = Ytr - ym
            if Z.shape[1] <= Z.shape[0]:
                G = Z.T @ Z; lam, V = np.linalg.eigh(G); ZtY = V.T @ (Z.T @ Yc)
                coef = lambda a: V @ (ZtY / (lam + a)[:, None])
            else:                                 # dual form: w = Z^T (Z Z^T + a I)^-1 Y
                K = Z @ Z.T; lam, U = np.linalg.eigh(K); UtY = U.T @ Yc
                coef = lambda a: Z.T @ (U @ (UtY / (lam + a)[:, None]))
            for a in (1e-8, 1e-6, 1e-4, 1e-2, 1, 100):
                W = coef(a * len(Z)); pv = Zv @ W + ym
                for j in range(4):
                    e = mae_dec(pv[:, j], Yva[:, j])
                    if e < best[j][0]: best[j] = (e, {"degree": d, "alpha": a}); self.solutions[j] = (pf, sc, W[:, j], ym[j])
        self.info = {c: {**b[1], "val_mae_dec": round(b[0], 5)} for c, b in zip(["id", "ig", "is", "ib"], best)}
        return self

    def predict(self, X, df=None):
        out = []
        for j in range(4):
            pf, sc, w, m = self.solutions[j]; out.append(sc.transform(pf.transform(X)) @ w + m)
        return np.column_stack(out)


class PolyLosses(GridModel):
    """Polynomial (degree 3) linear model + L2 penalty, trained on the GPU with different losses:
    squared, Huber (0.05 decades), absolute (L1) and log-cosh. Loss and penalty picked per current on validation."""
    LOSSES = ("squared", "huber", "l1", "logcosh")

    def fit(self, data):
        Xtr, Ytr, _ = data["train"]; Xva, Yva, _ = data["validation"]
        self.pf = PolynomialFeatures(3, include_bias=False).fit(Xtr); self.sc = StandardScaler().fit(self.pf.transform(Xtr))
        T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
        Z, Zv = T(self.sc.transform(self.pf.transform(Xtr))), T(self.sc.transform(self.pf.transform(Xva)))
        ym = Ytr.mean(0); Y, Yv = T(Ytr - ym), Yva - ym
        lossf = {"squared": lambda r: (r ** 2).mean(), "huber": lambda r: nn.functional.huber_loss(r, torch.zeros_like(r), delta=0.05),
                 "l1": lambda r: r.abs().mean(), "logcosh": lambda r: torch.log(torch.cosh(r / 0.05)).mean()}
        best = [(np.inf, None, None)] * 4
        for loss in self.LOSSES:
            for lam in (1e-7, 1e-5, 1e-3):
                torch.manual_seed(0)
                W = torch.zeros(Z.shape[1], 4, device=DEV, requires_grad=True); b = torch.zeros(4, device=DEV, requires_grad=True)
                opt = torch.optim.Adam([W, b], lr=3e-3); steps = 4000
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
                bestW, bestE = [None] * 4, [np.inf] * 4
                for s in range(steps):
                    r = Z @ W + b - Y
                    l = lossf[loss](r) + lam * (W ** 2).sum()
                    opt.zero_grad(); l.backward(); opt.step(); sch.step()
                    if s % 50 == 49 or s == steps - 1:
                        with torch.no_grad(): pv = (Zv @ W + b).cpu().numpy()
                        for j in range(4):
                            e = mae_dec(pv[:, j], Yv[:, j])
                            if e < bestE[j]: bestE[j] = e; bestW[j] = (W[:, j].detach().cpu().numpy().copy(), float(b[j]))
                for j in range(4):
                    if bestE[j] < best[j][0]: best[j] = (bestE[j], {"loss": loss, "l2": lam}, bestW[j])
        self.best, self.ym = best, ym
        self.info = {c: {**bb[1], "val_mae_dec": round(bb[0], 5)} for c, bb in zip(["id", "ig", "is", "ib"], best)}
        return self

    def predict(self, X, df=None):
        Z = self.sc.transform(self.pf.transform(X))
        return np.column_stack([Z @ self.best[j][2][0] + self.best[j][2][1] + self.ym[j] for j in range(4)])


# ---------------- kernel methods ----------------
class KRR(GridModel):
    """Kernel ridge regression (squared loss + RKHS penalty) with an RBF or Laplacian kernel on the 20 features."""
    multi = True
    def __init__(self, case, polarity, kernel="rbf"):
        super().__init__(case, polarity); self.kernel = kernel
    def grid(self):
        gam = (0.005, 0.01, 0.02, 0.05, 0.1) if self.kernel == "rbf" else (0.01, 0.03, 0.1)
        return [{"gamma": g, "alpha": a} for g in gam for a in (1e-6, 1e-4, 1e-2)]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva):
        m = KernelRidge(kernel=self.kernel, **cfg).fit(Xtr, ytr - ytr.mean(0)); m.ymean = ytr.mean(0); return m
    def pred_one(self, est, X, j): return est.predict(X)[:, j] + est.ymean[j]


class _ExactGP:
    """Exact GP with an ARD kernel (one length-scale per feature) + signal variance + white noise, written in torch.
    Hyperparameters: maximise the log marginal likelihood (Adam, 300 steps, float64, GPU) on 2,000 training rows;
    then condition on ALL 7,000 training rows with those hyperparameters (Cholesky, float64). Target standardised."""
    def __init__(self, kernel): self.kernel = kernel
    def _k(self, A, B, ls, sf):
        A, B = A / ls, B / ls
        r2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
        r2 = r2.clamp_min(0)
        if self.kernel == "rbf": return sf * torch.exp(-0.5 * r2)
        r = torch.sqrt(r2 + 1e-12) * np.sqrt(5.0)                         # Matern 5/2
        return sf * (1 + r + r * r / 3) * torch.exp(-r)
    def fit(self, X, y):
        T = lambda a: torch.tensor(a, dtype=torch.float64, device=DEV)
        self.mu, self.sd = y.mean(), y.std(); yn = (y - self.mu) / self.sd
        idx = np.random.default_rng(0).choice(len(X), min(2000, len(X)), replace=False)
        Xs, ys = T(X[idx]), T(yn[idx])
        p = {n: torch.zeros(s, dtype=torch.float64, device=DEV, requires_grad=True)
             for n, s in (("lls", X.shape[1]), ("lsf", 1), ("lsn", 1))}
        with torch.no_grad(): p["lls"] += np.log(np.sqrt(X.shape[1])); p["lsn"] -= 6
        opt = torch.optim.Adam(p.values(), lr=0.05)
        for _ in range(300):
            K = self._k(Xs, Xs, p["lls"].exp(), p["lsf"].exp()) + (p["lsn"].exp() + 1e-8) * torch.eye(len(Xs), dtype=torch.float64, device=DEV)
            Lc = torch.linalg.cholesky(K); a = torch.cholesky_solve(ys[:, None], Lc)
            nll = 0.5 * (ys[:, None] * a).sum() + torch.log(torch.diagonal(Lc)).sum()
            opt.zero_grad(); nll.backward(); opt.step()
        self.ls, self.sf, self.sn = (p[n].detach().exp() for n in ("lls", "lsf", "lsn"))
        self.Xt = T(X)
        K = self._k(self.Xt, self.Xt, self.ls, self.sf) + (self.sn + 1e-8) * torch.eye(len(X), dtype=torch.float64, device=DEV)
        self.alpha = torch.cholesky_solve(T(yn)[:, None], torch.linalg.cholesky(K))
        return self
    def predict(self, X):
        out = []
        for i in range(0, len(X), 2000):
            Ks = self._k(torch.tensor(X[i:i + 2000], dtype=torch.float64, device=DEV), self.Xt, self.ls, self.sf)
            out.append((Ks @ self.alpha)[:, 0].cpu().numpy())
        return np.concatenate(out) * self.sd + self.mu


class GPR(GridModel):
    """Gaussian process regression (exact GP, ARD kernel + noise; see _ExactGP). RBF vs Matern-5/2 picked on validation."""
    def grid(self): return [{"kernel": "rbf"}, {"kernel": "matern52"}]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva): return _ExactGP(cfg["kernel"]).fit(Xtr, ytr)


class SVRModel(GridModel):
    """Support vector regression, RBF kernel, epsilon-insensitive loss (epsilon in decades)."""
    def grid(self): return [{"C": c, "gamma": g, "epsilon": 0.005} for c, g in ((10, 0.03), (100, 0.03), (100, 0.01))]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva):
        m = SVR(kernel="rbf", cache_size=2000, **cfg).fit(Xtr, ytr - ytr.mean()); m.ymean = ytr.mean(); return m
    def pred_one(self, est, X, j): return est.predict(X) + est.ymean


class KNN(GridModel):
    multi = True
    def grid(self): return [{"n_neighbors": k} for k in (3, 5, 10, 20)]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva): return KNeighborsRegressor(weights="distance", **cfg).fit(Xtr, ytr)


# ---------------- trees ----------------
class RF(GridModel):
    multi = True
    def grid(self): return [{"max_features": f, "min_samples_leaf": l} for f in (0.33, 0.66, 1.0) for l in (1, 3)]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva):
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0, **cfg).fit(Xtr, ytr)


class ET(RF):
    def fit_one(self, cfg, Xtr, ytr, Xva, yva):
        return ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=0, **cfg).fit(Xtr, ytr)


class Ada(GridModel):
    """AdaBoost.R2 with regression trees."""
    def grid(self): return [{"depth": d, "learning_rate": lr, "loss": l} for d in (6, 10) for lr in (0.1, 1.0) for l in ("linear", "square")]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva):
        return AdaBoostRegressor(DecisionTreeRegressor(max_depth=cfg["depth"]), n_estimators=300,
                                 learning_rate=cfg["learning_rate"], loss=cfg["loss"], random_state=0).fit(Xtr, ytr)


class HGB(GridModel):
    """sklearn histogram gradient boosting (squared loss)."""
    def grid(self): return [{"learning_rate": lr, "max_leaf_nodes": n} for lr in (0.05, 0.1) for n in (31, 63)]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva):
        return HistGradientBoostingRegressor(max_iter=3000, early_stopping=False, random_state=0, **cfg).fit(Xtr, ytr)


class XGB(GridModel):
    def grid(self): return [{"max_depth": d} for d in (4, 6, 8)]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva):
        return XGBRegressor(n_estimators=20000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            device="cpu", early_stopping_rounds=300, random_state=0, **cfg).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)


class LGBM(GridModel):
    def grid(self): return [{"num_leaves": n} for n in (15, 31, 63)]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva):
        return LGBMRegressor(n_estimators=20000, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             min_child_samples=5, random_state=0, verbose=-1, **cfg).fit(
            Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb_early_stopping(300, verbose=False)])


class Cat(GridModel):
    def grid(self): return [{"depth": d} for d in (6, 8)]
    def fit_one(self, cfg, Xtr, ytr, Xva, yva):
        return CatBoostRegressor(iterations=20000, learning_rate=0.05, od_type="Iter", od_wait=300, random_seed=0,
                                 verbose=0, thread_count=-1, **cfg).fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)


# ---------------- reference: the chosen MLP ----------------
class FinalMLP:
    """The current final model (Final_models), not retrained."""
    def __init__(self, case, polarity):
        import train_final as tf
        self.tf, self.bundle, self.info = tf, tf.load(case), {"source": "Final_models/models"}
    def fit(self, data): return self
    def predict(self, X, df): return self.tf.predict_log(self.bundle, df)


def _lazy(module, cls):
    def make(case, polarity):
        import importlib
        return getattr(importlib.import_module(module), cls)(case, polarity)
    return make


REGISTRY = {
    "ridge_poly": RidgePoly,
    "poly_losses": PolyLosses,
    "krr_rbf": KRR,
    "krr_laplacian": lambda c, p: KRR(c, p, "laplacian"),
    "gpr": GPR,
    "svr_rbf": SVRModel,
    "knn": KNN,
    "random_forest": RF,
    "extra_trees": ET,
    "adaboost": Ada,
    "hist_gboost": HGB,
    "xgboost": XGB,
    "lightgbm": LGBM,
    "catboost": Cat,
    "mlp_final": FinalMLP,
    "analytical": _lazy("model_analytic", "AnalyticModel"),
    "pinn": _lazy("model_pinn", "PINNModel"),
    "components_mlp": _lazy("model_components", "ComponentModel"),
    "components_mlp_vf": _lazy("model_components", "ComponentModelVF"),
    # from the literature review (models_research.py)
    "resnet_mlp": _lazy("models_research", "ResNetMLP"),
    "mlp_multitask": _lazy("models_research", "MultiTaskMLP"),
    "tiny_ann": _lazy("models_research", "TinyANN"),
    "moe": _lazy("models_research", "MoEModel"),
    "kan": _lazy("models_research", "KANModel"),
    "ft_transformer": _lazy("models_research", "FTTModel"),
    "hybrid_analytic_mlp": _lazy("models_research", "HybridAnalyticMLP"),
    "hybrid_analytic_lgbm": _lazy("models_research", "HybridAnalyticLGBM"),
    "lgbm_linear_tree": _lazy("models_research", "LGBMLinearTree"),
    "ensemble_top3": _lazy("models_research", "EnsembleTop3"),
}
