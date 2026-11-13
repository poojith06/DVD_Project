"""Models suggested by the literature review (same harness, same data, same tuning rule - validation only).

  resnet_mlp          ResNet-style MLP (skip connections)            - Agarwal/Jain/Amuru/Abbas, IIITH (PTM 22nm HP leakage)
  mlp_multitask       one MLP for all 4 currents (multi-task ANN)   - Chai et al., IEEE Access 2025
  tiny_ann            15-10 sigmoid ANN; ON id/is use I = 10^y * |Vds| output conversion - Samsung, IEEE TED 2021
  moe                 mixture of 3 expert MLPs with a learned softmax gate - PRIME, arXiv 2505.11523
  kan                 Kolmogorov-Arnold network (B-spline edge functions) - KAN vs MLP for FinFET, arXiv 2503.15209
  ft_transformer      FT-Transformer: one token per feature + self-attention (Gorishniy et al., NeurIPS 2021)
  hybrid_analytic_mlp analytical compact model + MLP that learns the residual - Kao/Hu 2022, PRISM-UDE
  hybrid_analytic_lgbm analytical compact model + LightGBM on the residual
  lgbm_linear_tree    LightGBM with linear models in the leaves (piecewise-linear instead of piecewise-constant)
  ensemble_top3       per current, the average of the 3 models with the lowest validation error (MetaCirc-style
                      meta-ensemble; run it LAST - it reads the saved predictions; its validation numbers are optimistic
                      because validation chose the members, test is clean)
Neural nets: same recipe as Final_models (one net per group (id,is), ig, ib unless stated; Huber beta 0.05 decades on
the standardised log10 target, AdamW lr 1e-3 wd 1e-4, cosine LR, batch 128, 500 epochs, best validation epoch).
"""
import copy, json
from pathlib import Path
import numpy as np, torch
from torch import nn
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GROUPS = {"id+is": [0, 2], "ig": [1], "ib": [3]}
HERE = Path(__file__).resolve().parent


# ---------------- network architectures ----------------
def mlp(n_in, n_out, hidden=(256, 128, 64), act=nn.SiLU):
    layers, a = [], n_in
    for h in hidden: layers += [nn.Linear(a, h), act()]; a = h
    return nn.Sequential(*layers, nn.Linear(a, n_out))

class ResMLP(nn.Module):
    def __init__(self, n_in, n_out, width=256, blocks=3):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)) for _ in range(blocks)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, n_out))
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks: h = h + b(h)
        return self.out(h)

class MoE(nn.Module):
    def __init__(self, n_in, n_out, experts=3, hidden=(128, 64)):
        super().__init__()
        self.experts = nn.ModuleList([mlp(n_in, n_out, hidden) for _ in range(experts)])
        self.gate = nn.Sequential(nn.Linear(n_in, 32), nn.SiLU(), nn.Linear(32, experts))
    def forward(self, x):
        g = torch.softmax(self.gate(x), dim=1)                          # [rows, experts]
        return torch.stack([e(x) for e in self.experts], dim=2).mul(g[:, None, :]).sum(2)

class KANLayer(nn.Module):
    """phi(x) = w_base * silu(x) + sum_k c_k B_k(x), cubic B-splines on a fixed grid over [-3, 3] (inputs layer-normed)."""
    def __init__(self, n_in, n_out, grid=8, k=3, lo=-3.0, hi=3.0):
        super().__init__()
        h = (hi - lo) / grid
        self.register_buffer("t", torch.arange(-k, grid + k + 1, dtype=torch.float32) * h + lo)
        self.k, self.norm = k, nn.LayerNorm(n_in)
        self.base = nn.Linear(n_in, n_out)
        self.coef = nn.Parameter(torch.randn(n_out, n_in, grid + k) * 0.1 / np.sqrt(n_in))
    def bases(self, x):                                                  # x [rows, in] -> [rows, in, grid+k]
        x = x.unsqueeze(-1); t = self.t
        B = ((x >= t[:-1]) & (x < t[1:])).float()
        for p in range(1, self.k + 1):
            B = (x - t[:-(p + 1)]) / (t[p:-1] - t[:-(p + 1)]) * B[..., :-1] + (t[p + 1:] - x) / (t[p + 1:] - t[1:-p]) * B[..., 1:]
        return B
    def forward(self, x):
        x = self.norm(x)
        return self.base(nn.functional.silu(x)) + torch.einsum("rik,oik->ro", self.bases(x), self.coef)

class KAN(nn.Module):
    def __init__(self, n_in, n_out, hidden=(32, 16)):
        super().__init__()
        sizes = [n_in, *hidden, n_out]
        self.layers = nn.Sequential(*[KANLayer(a, b) for a, b in zip(sizes[:-1], sizes[1:])])
    def forward(self, x): return self.layers(x)


class FTTransformer(nn.Module):
    """FT-Transformer (Gorishniy et al., NeurIPS 2021): every feature -> a d-dim token (x_i * w_i + b_i), a [CLS]
    token is prepended, pre-norm transformer encoder layers, regression head on the [CLS] output."""
    def __init__(self, n_in, n_out, d=64, layers=3, heads=8, ff=128, dropout=0.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, d) * 0.1); self.b = nn.Parameter(torch.zeros(n_in, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d, n_out))
    def forward(self, x):
        tok = x.unsqueeze(-1) * self.w + self.b                          # [rows, features, d]
        h = self.enc(torch.cat([self.cls.expand(len(x), -1, -1), tok], dim=1))
        return self.head(h[:, 0])


# ---------------- shared trainer ----------------
def train_net(make, Xtr, ytr, Xva, yva, seed=0, epochs=500, lr=1e-3, wd=1e-4, bs=128, beta=0.05):
    """Fit make(n_in, n_out) to ytr (decades). Returns a predict(X) -> decades function."""
    torch.manual_seed(seed)
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-12
    T = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float32, device=DEV)
    Xt, yt, Xv, yv, MU, SD = T(Xtr), T((ytr - mu) / sd), T(Xva), T(yva), T(mu), T(sd)
    net = make(Xt.shape[1], yt.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, state = float("inf"), None
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            loss = nn.functional.smooth_l1_loss(MU + SD * net(Xt[idx]), MU + SD * yt[idx], beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad(): vl = nn.functional.smooth_l1_loss(MU + SD * net(Xv), yv, beta=beta).item()
        if vl < best: best, state = vl, copy.deepcopy(net.state_dict())
    net.load_state_dict(state); net.eval()
    def predict(X):
        with torch.no_grad(): return (MU + SD * net(T(X))).cpu().numpy()
    return predict


class GroupNN:
    """One network per output group. Subclasses set make / groups and may add extra inputs or a log10 prior (base)."""
    groups = GROUPS
    make = staticmethod(mlp)
    def __init__(self, case, polarity): self.case, self.polarity, self.info = case, polarity, {}
    def inputs(self, X, df): return X
    def base(self, X, df): return np.zeros((len(X), 4))
    def fit(self, data):
        (Xtr, Ytr, dtr), (Xva, Yva, dva) = data["train"], data["validation"]
        Itr, Iva = self.inputs(Xtr, dtr), self.inputs(Xva, dva)
        Rtr, Rva = Ytr - self.base(Xtr, dtr), Yva - self.base(Xva, dva)
        self.pred = {g: train_net(self.make, Itr, Rtr[:, c], Iva, Rva[:, c], seed=i) for i, (g, c) in enumerate(self.groups.items())}
        pv = self.predict(Xva, dva)
        self.info = {**self.info, "val_mae_dec": [round(float(np.mean(np.abs(pv[:, j] - Yva[:, j]))), 5) for j in range(4)]}
        return self
    def predict(self, X, df):
        out, I = self.base(X, df).copy(), self.inputs(X, df)
        for g, c in self.groups.items(): out[:, c] += self.pred[g](I)
        return out


class ResNetMLP(GroupNN):
    make = staticmethod(ResMLP)

class MultiTaskMLP(GroupNN):
    groups = {"all": [0, 1, 2, 3]}

class TinyANN(GroupNN):
    make = staticmethod(lambda i, o: mlp(i, o, (15, 10), nn.Sigmoid))
    def base(self, X, df):          # Samsung output conversion I = 10^y * |Vds| for the ON channel current
        b = np.zeros((len(X), 4))
        if self.case.endswith("_on"):
            b[:, 0] = b[:, 2] = np.log10(np.maximum(np.abs(df.vd.to_numpy() - df.vs.to_numpy()), 1e-6))
        return b

class MoEModel(GroupNN):
    make = staticmethod(MoE)

class KANModel(GroupNN):
    make = staticmethod(KAN)

class FTTModel(GroupNN):
    """FT-Transformer, one per output group. AdamW lr 1e-3 is the common FT-Transformer default range (1e-4..1e-3)."""
    make = staticmethod(FTTransformer)


class HybridAnalyticMLP(GroupNN):
    """log10|I| = analytical compact model + MLP correction; the MLP also sees the analytical prediction."""
    def fit(self, data):
        from model_analytic import AnalyticModel
        self.an = AnalyticModel(self.case, self.polarity).fit(data)
        self._cache = {}
        return super().fit(data)
    def _an(self, X, df):
        key = (len(X), float(X[0, 0]), float(X[-1, -1]))
        if key not in self._cache: self._cache[key] = np.asarray(self.an.predict(X, df), float)
        return self._cache[key]
    def base(self, X, df): return self._an(X, df)
    def inputs(self, X, df):
        a = self._an(X, df)
        if not hasattr(self, "a_mu"): self.a_mu, self.a_sd = a.mean(0), a.std(0) + 1e-9
        return np.hstack([X, (a - self.a_mu) / self.a_sd])


class HybridAnalyticLGBM(HybridAnalyticMLP):
    """analytical compact model + LightGBM on the residual (inputs: features + analytical prediction)."""
    def fit(self, data):
        from model_analytic import AnalyticModel
        (Xtr, Ytr, dtr), (Xva, Yva, dva) = data["train"], data["validation"]
        self.an = AnalyticModel(self.case, self.polarity).fit(data); self._cache = {}
        Itr, Iva = self.inputs(Xtr, dtr), self.inputs(Xva, dva)
        Rtr, Rva = Ytr - self.base(Xtr, dtr), Yva - self.base(Xva, dva)
        self.m, val = [], []
        for j in range(4):
            best = None
            for leaves in (15, 31, 63):
                m = LGBMRegressor(n_estimators=20000, learning_rate=0.03, num_leaves=leaves, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, min_child_samples=5, random_state=0, verbose=-1).fit(
                    Itr, Rtr[:, j], eval_set=[(Iva, Rva[:, j])], callbacks=[lgb_early_stopping(300, verbose=False)])
                e = float(np.mean(np.abs(m.predict(Iva) - Rva[:, j])))
                if best is None or e < best[0]: best = (e, m, leaves)
            self.m.append(best[1]); val.append((best[2], round(best[0], 5)))
        self.info = {"leaves, val_mae_dec": val}
        return self
    def predict(self, X, df):
        I = self.inputs(X, df)
        return self.base(X, df) + np.column_stack([m.predict(I) for m in self.m])


class LGBMLinearTree:
    """LightGBM with a linear regression in every leaf."""
    def __init__(self, case, polarity): self.info = {}
    def fit(self, data):
        (Xtr, Ytr, _), (Xva, Yva, _) = data["train"], data["validation"]
        self.m, val = [], []
        for j in range(4):
            best = None
            for leaves in (7, 15, 31):
                m = LGBMRegressor(n_estimators=20000, learning_rate=0.03, num_leaves=leaves, linear_tree=True, linear_lambda=1e-4,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20, random_state=0, verbose=-1).fit(
                    Xtr, Ytr[:, j], eval_set=[(Xva, Yva[:, j])], callbacks=[lgb_early_stopping(300, verbose=False)])
                e = float(np.mean(np.abs(m.predict(Xva) - Yva[:, j])))
                if best is None or e < best[0]: best = (e, m, leaves)
            self.m.append(best[1]); val.append((best[2], round(best[0], 5)))
        self.info = {"leaves, val_mae_dec": val}
        return self
    def predict(self, X, df=None): return np.column_stack([m.predict(X) for m in self.m])


class EnsembleTop3:
    """Per current: average log10|I| of the 3 saved models with the lowest validation mean |error| in decades."""
    def __init__(self, case, polarity): self.case, self.info = case, {}
    def fit(self, data):
        Yva = data["validation"][1]; self.preds = {}
        for f in sorted((HERE / "results").glob(f"*/{self.case}_logpred.npz")):
            name = f.parent.name
            if name in ("ensemble_top3", "components_mlp", "components_mlp_vf"): continue   # those train on extra SPICE data
            self.preds[name] = dict(np.load(f))
        err = {n: np.mean(np.abs(p["validation"] - Yva), axis=0) for n, p in self.preds.items()}
        self.members = [sorted(err, key=lambda n: err[n][j])[:3] for j in range(4)]
        self.info = {c: m for c, m in zip(["id", "ig", "is", "ib"], self.members)}
        self._X = {"train": data["train"][0], "validation": data["validation"][0]}
        return self
    def predict(self, X, df):
        # identify the split by its rows (validation and test both have 1,500 rows, so size alone is ambiguous)
        split = next((k for k, A in self._X.items() if A.shape == X.shape and np.array_equal(A, X)), "test")
        return np.column_stack([np.mean([self.preds[m][split][:, j] for m in self.members[j]], axis=0) for j in range(4)])
