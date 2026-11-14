"""The four model types, all in one file. Each predicts log10|I| of (id, ig, is, ib) for one transistor case.

| Key                  | Model                                                                                |
|----------------------|--------------------------------------------------------------------------------------|
| gpr                  | exact Gaussian process, ARD RBF or Matern-5/2 kernel per current (picked on validation) |
| hybrid_analytic_mlp  | analytical compact model (closed-form device equations) + MLP for the remaining error |
| resnet_mlp           | MLP with residual (skip) connections, one network per output group                    |
| pinn                 | physics-informed MLP: Kirchhoff / monotonicity penalties, EKV channel-current prior   |

Common interface:  m = MODELS[key](case, seed, params)
                   m.fit(X_train, y_train, X_val, y_val, df_train, df_val, w_train=None)
                   m.predict(X, df) -> (rows, 4) log10|I|
X = scaled features (pipeline.build_features), df = raw rows, y = log10|I|, w_train = per-row training-loss
weights (pipeline.rail_weights; used by the three neural-network models, GPR has no per-row loss and ignores it).
Only the four dataset currents are used as targets.
"""
import copy
import os
import time

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import nn
from torch.func import jvp

from config import CASES, GROUPS, MODEL_SEED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================== shared network trainer

NET_PARAMS = {"epochs": 500, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128, "huber_beta": 0.05}


def mlp(n_in, n_out, hidden=(256, 128, 64), act=nn.SiLU):
    layers, size = [], n_in
    for width in hidden:
        layers += [nn.Linear(size, width), act()]
        size = width
    return nn.Sequential(*layers, nn.Linear(size, n_out))


class TrainedNet:
    """Trained network (on the CPU) + target scaling; predict(X) -> decades."""

    def __init__(self, net, y_mean, y_std):
        self.net, self.y_mean, self.y_std = net.cpu().eval(), y_mean, y_std

    def predict(self, X):
        with torch.no_grad():
            z = self.net(torch.as_tensor(np.asarray(X, float), dtype=torch.float32)).numpy()
        return self.y_mean + self.y_std * z


def train_net(make, X_train, y_train, X_val, y_val, seed=0, params=None, w_train=None):
    """Huber loss (0.05 decades) on log10|I|, AdamW, cosine LR, batch 128, 500 epochs, best validation epoch.
    w_train: optional per-row weights on the training loss (validation loss is unweighted)."""
    p = {**NET_PARAMS, **(params or {})}
    torch.manual_seed(seed)
    mu, sd = y_train.mean(0), y_train.std(0) + 1e-12
    T = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float32, device=DEVICE)
    X, y, Xv, yv, MU, SD = T(X_train), T((y_train - mu) / sd), T(X_val), T(y_val), T(mu), T(sd)
    W = T(np.ones(len(X_train)) if w_train is None else w_train)
    net = make(X.shape[1], y.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, p["epochs"])
    best, best_state = float("inf"), None
    for _ in range(p["epochs"]):
        net.train()
        perm = torch.randperm(len(X), device=DEVICE)
        for i in range(0, len(X), p["batch_size"]):
            idx = perm[i:i + p["batch_size"]]
            per_row = nn.functional.smooth_l1_loss(MU + SD * net(X[idx]), MU + SD * y[idx],
                                                   beta=p["huber_beta"], reduction="none").mean(1)
            loss = (per_row * W[idx]).sum() / W[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            val = nn.functional.smooth_l1_loss(MU + SD * net(Xv), yv, beta=p["huber_beta"]).item()
        if val < best:
            best, best_state = val, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    return TrainedNet(net, mu, sd)

# =============================================================================== 1. ResNet MLP

class ResMLP(nn.Module):
    """Linear(20 -> 256), 3 residual blocks h = h + Linear(SiLU(Linear(SiLU(h)))), SiLU, Linear(256 -> outputs)."""

    def __init__(self, n_in, n_out, width=256, blocks=3):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                                   nn.Linear(width, width)) for _ in range(blocks)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, n_out))

    def forward(self, x):
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


class ResNetMLPModel:
    name = "resnet_mlp"

    def __init__(self, case, seed=MODEL_SEED, params=None):
        self.case, self.seed, self.params = case, seed, params

    def fit(self, X_train, y_train, X_val, y_val, df_train=None, df_val=None, w_train=None):
        p = dict(self.params or {})
        arch = {k: int(p.pop(k)) for k in ("width", "blocks") if k in p}     # architecture; the rest -> trainer
        make = lambda n_in, n_out: ResMLP(n_in, n_out, **arch)
        self.nets = {g: train_net(make, X_train, y_train[:, c], X_val, y_val[:, c], self.seed + i, p or None, w_train)
                     for i, (g, c) in enumerate(GROUPS.items())}
        return self

    def predict(self, X, df=None):
        out = np.zeros((len(X), 4))
        for g, c in GROUPS.items():
            out[:, c] = self.nets[g].predict(X)
        return out

# =============================================================================== 2. hybrid: analytical model + MLP

K_B, Q = 1.380649e-23, 1.602176634e-19
EPS_OX = 3.9 * 8.854e-12
L_CH = 22e-9
TNOM = 300.15
NOM = {"nmos": dict(vth0=0.50308, toxe=1.05e-9, ndep=5.5e18, u0=0.04),
       "pmos": dict(vth0=0.4606, toxe=1.1e-9, ndep=4.4e18, u0=0.0095)}


def _softplus(x):
    return np.logaddexp(0.0, x)


def _eg(T):
    return 1.16 - 7.02e-4 * T ** 2 / (T + 1108.0)


def _ex(x):
    return np.exp(np.clip(x, -700, 700))


class AnalyticModel:
    """Closed-form device equations with ~20-30 fitted constants per case (PMOS mirrored onto NMOS form):
    EKV channel current (DIBL, body effect, temperature, process), BSIM4-style gate-S/D overlap tunnelling,
    gate-channel and gate-body tunnelling (ON), reverse junction leakage, GIDL/GISL (OFF), body floor.
    Fitted by least squares on log10|I| (staged: channel -> gate -> body -> joint); per current, a separate
    refit is kept if it is better on validation."""

    def __init__(self, case, polarity):
        self.case, self.polarity, self.on, self.nom = case, polarity, case.endswith("_on"), NOM[polarity]
        p = dict(dw=10.0, sig2=0.0, gam=0.0, kt1=-0.11, k_tox=0.05, k_nd=0.05, sigma=0.05, n=1.4, ute=-1.5,
                 c_is=np.log(0.3), o0=-8.0, o1=2.0, o2=-5.0, o3=0.0, o4=0.0,
                 j0=np.log(1e-10), ea=1.0, j1=0.0, f0=np.log(1e-14))
        if self.on:
            p.update(lam=0.1, th_mob=1.0, th_vsat=2.0, c0=-6.0, c1=2.0, c2=-5.0, p_gcd=1.0,
                     b0=-9.0, b1=2.0, b2=-5.0, eb=0.5, nb=3.0)
        else:
            p.update(g0=11.1, g1=6.6, eg=0.8)
        self.names = list(p)
        self.p0 = np.array([p[k] for k in self.names], float)

    def _vars(self, df):
        vb = df["vb"].to_numpy(float)
        sgn = 1.0 if self.polarity == "nmos" else -1.0
        vd, vs, vg = (sgn * (df[k].to_numpy(float) - vb) for k in ("vd", "vs", "vg"))
        T = df["temp"].to_numpy(float) + 273.15
        V = dict(vd=vd, vs=vs, vg=vg, vhi=np.maximum(vd, vs), vlo=np.minimum(vd, vs), T=T, Vt=K_B * T / Q,
                 W0=df["w"].to_numpy(float), vth=np.abs(df["vth0"].to_numpy(float)), toxe=df["toxe"].to_numpy(float),
                 u0=df["u0"].to_numpy(float), d_tox=df["toxe"].to_numpy(float) / self.nom["toxe"] - 1,
                 d_nd=df["ndep"].to_numpy(float) / self.nom["ndep"] - 1)
        V["vds"] = V["vhi"] - V["vlo"]
        V["d_hi"] = vd >= vs
        return V

    def currents(self, theta, V):
        P = dict(zip(self.names, theta))
        Vt, T, dtox = V["Vt"], V["T"], V["d_tox"]
        W = np.maximum(V["W0"] - abs(P["dw"]) * 1e-9, 1e-9)
        vg, vhi, vlo, vds = V["vg"], V["vhi"], V["vlo"], V["vds"]
        tr = T / TNOM - 1
        n = 1 + (abs(P["n"] - 1) + 1e-3) * (1 + dtox) * np.sqrt(1 + V["d_nd"])
        Vth = (V["vth"] + P["kt1"] * tr + P["k_tox"] * dtox + P["k_nd"] * V["d_nd"] - P["sigma"] * vds
               - P["sig2"] * vds ** 2 + P["gam"] * (np.sqrt(0.9 + vlo) - np.sqrt(0.9)))
        Vp = (vg - Vth) / n
        q_lo, q_hi = _softplus((Vp - vlo) / (2 * Vt)), _softplus((Vp - vhi) / (2 * Vt))
        Is = 2 * n * V["u0"] * EPS_OX / V["toxe"] * (W / L_CH) * Vt ** 2 * (T / TNOM) ** P["ute"] * _ex(P["c_is"])
        Ich = Is * (q_lo ** 2 - q_hi ** 2)
        if self.on:
            Ich = Ich * (1 + P["lam"] * vds) / (1 + abs(P["th_mob"]) * Vt * (q_lo + q_hi)
                                               + abs(P["th_vsat"]) * 2 * Vt * (q_lo - q_hi))

        def overlap(Vx):
            a = np.abs(Vx)
            return np.sign(Vx) * W * Vx ** 2 / (1 + dtox) ** 2 * _ex(P["o0"] + P["o1"] * a + dtox * (P["o2"] + P["o3"] * a)
                                                                  + P["o4"] * tr)
        Ig_hi, Ig_lo = overlap(vg - vhi), overlap(vg - vlo)
        tfac = (T / TNOM) ** 3 * _ex(P["ea"] * (_eg(TNOM) / (K_B * TNOM / Q) - _eg(T) / Vt))

        def junction(v):
            return W * _ex(P["j0"]) * tfac * (1 - np.exp(-np.maximum(v, 0) / Vt)) * np.maximum(1 + P["j1"] * v, 1e-3)
        J_hi, J_lo = junction(vhi), junction(vlo)
        I0 = _ex(P["f0"])
        i_hi = Ich - Ig_hi + J_hi + I0 / 2
        i_lo = -Ich - Ig_lo + J_lo + I0 / 2
        ig = Ig_hi + Ig_lo
        ib = -(J_hi + J_lo + I0)
        if self.on:
            Vgs = np.maximum(vg - vlo, 0)
            Vaux = n * Vt * _softplus((Vgs - Vth) / (n * Vt))
            Igc = W * Vgs * Vaux * _ex(P["c0"] + P["c1"] * Vgs + P["c2"] * dtox)
            pv = abs(P["p_gcd"]) * vds
            den = pv ** 2 + 2e-4
            Igcs = Igc * (pv + np.exp(-pv) - 1 + 1e-4) / den
            Igcd = Igc * (1 - (pv + 1) * np.exp(-pv) + 1e-4) / den
            nb = abs(P["nb"]) + 0.1
            Igb = W * Vgs * nb * Vt * _softplus((Vgs - P["eb"]) / (nb * Vt)) * _ex(P["b0"] + P["b1"] * Vgs + P["b2"] * dtox)
            i_hi, i_lo = i_hi - Igcd, i_lo - Igcs
            ig = ig + Igcs + Igcd + Igb
            ib = ib - Igb
        else:
            def gidl(vx):
                Ve = 0.01 * _softplus((vx - vg - P["eg"]) / 0.01)
                return W * _ex(P["g0"]) * Ve * _ex(-abs(P["g1"]) * (1 + dtox) / np.maximum(Ve, 1e-6))
            G_hi, G_lo = gidl(vhi), gidl(vlo)
            i_hi, i_lo, ib = i_hi + G_hi, i_lo + G_lo, ib - G_hi - G_lo
        dh = V["d_hi"]
        return np.column_stack([np.where(dh, i_hi, i_lo), ig, np.where(dh, i_lo, i_hi), ib])

    def logI(self, theta, V):
        return np.log10(np.maximum(np.abs(self.currents(theta, V)), 1e-18))

    def _fit(self, theta, V, Y, free, outs, nfev=400):
        idx = [self.names.index(k) for k in free]

        def res(x):
            th = theta.copy()
            th[idx] = x
            r = self.logI(th, V)[:, outs] - Y[:, outs]
            return np.nan_to_num(r, nan=10.0, posinf=10.0, neginf=-10.0).ravel()
        sol = least_squares(res, theta[idx], loss="soft_l1", f_scale=0.1, max_nfev=nfev, x_scale="jac")
        th = theta.copy()
        th[idx] = sol.x
        return th

    def fit(self, dtr, Ytr, dva, Yva):
        Vtr, Vva = self._vars(dtr), self._vars(dva)
        th = self.p0.copy()
        gate = ["dw", "o0", "o1", "o2", "o3", "o4"] + (["c0", "c1", "c2", "p_gcd", "b0", "b1", "b2", "eb", "nb"]
                                                     if self.on else [])
        body = ["j0", "ea", "j1", "f0"] + ([] if self.on else ["g0", "g1", "eg"])
        chan = ["dw", "sig2", "gam", "kt1", "k_tox", "k_nd", "sigma", "n", "ute", "c_is"] + (
            ["lam", "th_mob", "th_vsat"] if self.on else [])

        def shift(th, keys, j):
            off = np.log(10) * np.median(Ytr[:, j] - self.logI(th, Vtr)[:, j])
            th = th.copy()
            for k in keys:
                th[self.names.index(k)] += off
            return th
        if self.on:
            th = self._fit(shift(th, ["c_is"], 0), Vtr, Ytr, chan, [0, 2], 200)
        th = self._fit(shift(th, ["o0"] + (["c0", "b0"] if self.on else []), 1), Vtr, Ytr, gate, [1], 400)
        th = self._fit(shift(th, ["j0"] if not self.on else ["j0", "f0"], 3), Vtr, Ytr, body, [3], 400)
        th = self._fit(th, Vtr, Ytr, chan, [0, 2], 200)
        th = self._fit(th, Vtr, Ytr, self.names, [0, 1, 2, 3], 150)
        mae = lambda t, j: float(np.mean(np.abs(self.logI(t, Vva)[:, j] - Yva[:, j])))
        self.theta, val = [th] * 4, [mae(th, j) for j in range(4)]
        for j in range(4):
            free = {1: gate, 3: body + (["b0", "b1", "b2", "eb", "nb"] if self.on else [])}.get(j, self.names)
            tj = self._fit(th, Vtr, Ytr, free, [j], 150)
            e = mae(tj, j)
            if e < val[j]:
                self.theta[j], val[j] = tj, e
        return self

    def predict(self, df):
        V = self._vars(df)
        return np.column_stack([self.logI(self.theta[j], V)[:, j] for j in range(4)])


class HybridAnalyticMLPModel:
    """log10|I| = analytical(df) + MLP(features, scaled analytical prediction), one MLP per output group."""
    name = "hybrid_analytic_mlp"

    def __init__(self, case, seed=MODEL_SEED, params=None):
        self.case, self.seed, self.params = case, seed, params

    def _inputs(self, X, a):
        return np.hstack([X, (a - self.a_mean) / self.a_std])

    def fit(self, X_train, y_train, X_val, y_val, df_train, df_val, w_train=None):
        self.analytic = AnalyticModel(self.case, CASES[self.case][0]).fit(df_train, y_train, df_val, y_val)
        a_tr, a_va = self.analytic.predict(df_train), self.analytic.predict(df_val)
        self.a_mean, self.a_std = a_tr.mean(0), a_tr.std(0) + 1e-9
        I_tr, I_va = self._inputs(X_train, a_tr), self._inputs(X_val, a_va)
        r_tr, r_va = y_train - a_tr, y_val - a_va
        self.nets = {g: train_net(mlp, I_tr, r_tr[:, c], I_va, r_va[:, c], self.seed + i, self.params, w_train)
                     for i, (g, c) in enumerate(GROUPS.items())}
        return self

    def predict(self, X, df):
        a = self.analytic.predict(df)
        out, inputs = a.copy(), self._inputs(X, a)
        for g, c in GROUPS.items():
            out[:, c] += self.nets[g].predict(inputs)
        return out

# =============================================================================== 3. Gaussian process

GP_KERNELS = ("rbf", "matern52")
# GPR can run on the CPU (env GPR_DEVICE=cpu) so it trains in parallel with the networks on the GPU;
# float64 on a laptop GPU is no faster than on a multi-core CPU.
GP_DEVICE = os.environ.get("GPR_DEVICE", DEVICE)
if GP_DEVICE == "cpu" and os.environ.get("GPR_THREADS"):
    torch.set_num_threads(int(os.environ["GPR_THREADS"]))


class ExactGP:
    """Exact GP, ARD kernel + signal variance + noise. Hyper-parameters: log marginal likelihood (Adam, 300 steps,
    float64) on 2,000 random training rows; then conditioned on all training rows."""

    def __init__(self, kernel, seed=MODEL_SEED):
        self.kernel, self.seed = kernel, seed

    def _k(self, A, B, ls, sf):
        A, B = A / ls, B / ls
        r2 = ((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T).clamp_min(0)
        if self.kernel == "rbf":
            return sf * torch.exp(-0.5 * r2)
        r = torch.sqrt(r2 + 1e-12) * np.sqrt(5.0)
        return sf * (1 + r + r * r / 3) * torch.exp(-r)

    def fit(self, X, y):
        T = lambda a: torch.tensor(a, dtype=torch.float64, device=GP_DEVICE)
        self.mu, self.sd = y.mean(), y.std()
        yn = (y - self.mu) / self.sd
        idx = np.random.default_rng(self.seed).choice(len(X), min(2000, len(X)), replace=False)
        Xs, ys = T(X[idx]), T(yn[idx])
        p = {n: torch.zeros(s, dtype=torch.float64, device=GP_DEVICE, requires_grad=True)
             for n, s in (("lls", X.shape[1]), ("lsf", 1), ("lsn", 1))}
        with torch.no_grad():
            p["lls"] += np.log(np.sqrt(X.shape[1]))
            p["lsn"] -= 6
        opt = torch.optim.Adam(p.values(), lr=0.05)
        eye = torch.eye(len(Xs), dtype=torch.float64, device=GP_DEVICE)
        for _ in range(300):
            K = self._k(Xs, Xs, p["lls"].exp(), p["lsf"].exp()) + (p["lsn"].exp() + 1e-8) * eye
            Lc = torch.linalg.cholesky(K)
            a = torch.cholesky_solve(ys[:, None], Lc)
            nll = 0.5 * (ys[:, None] * a).sum() + torch.log(torch.diagonal(Lc)).sum()
            opt.zero_grad()
            nll.backward()
            opt.step()
        ls, sf, sn = (p[n].detach().exp() for n in ("lls", "lsf", "lsn"))
        Xt = T(X)
        K = torch.empty(len(X), len(X), dtype=torch.float64, device=GP_DEVICE)   # built in blocks to save memory
        for i in range(0, len(X), 1000):
            K[i:i + 1000] = self._k(Xt[i:i + 1000], Xt, ls, sf)
        K.diagonal().add_(float(sn) + 1e-8)
        Lc = torch.linalg.cholesky(K)
        del K
        alpha = torch.cholesky_solve(T(yn)[:, None], Lc)
        self.ls, self.sf, self.Xt, self.alpha = ls.cpu(), sf.cpu(), Xt.cpu(), alpha.cpu()
        return self

    def predict(self, X):
        ls, sf, Xt, alpha = (t.to(GP_DEVICE) for t in (self.ls, self.sf, self.Xt, self.alpha))
        out = []
        for i in range(0, len(X), 2000):
            Ks = self._k(torch.tensor(X[i:i + 2000], dtype=torch.float64, device=GP_DEVICE), Xt, ls, sf)
            out.append((Ks @ alpha)[:, 0].cpu().numpy())
        return np.concatenate(out) * self.sd + self.mu


class GPRModel:
    """One exact GP per current; RBF and Matern-5/2 both fitted, the lower validation MAE (decades) is kept."""
    name = "gpr"

    def __init__(self, case, seed=MODEL_SEED, params=None):
        self.case, self.seed = case, seed

    def fit(self, X_train, y_train, X_val, y_val, df_train=None, df_val=None, w_train=None):
        self.gps, self.kernels = [], []
        for j in range(y_train.shape[1]):
            best = (np.inf, None, None)
            for kernel in GP_KERNELS:
                gp = ExactGP(kernel, self.seed).fit(X_train, y_train[:, j])
                mae = float(np.mean(np.abs(gp.predict(X_val) - y_val[:, j])))
                if mae < best[0]:
                    best = (mae, kernel, gp)
            self.gps.append(best[2])
            self.kernels.append(best[1])
        return self

    def predict(self, X, df=None):
        return np.column_stack([gp.predict(X) for gp in self.gps])

# =============================================================================== 4. PINN

PINN_RAW = ["vdd", "temp", "w", "vd", "vs", "vth0", "toxe", "ndep", "u0", "xj"]
PINN_UNIT = np.array([1, 1, 1e9, 1, 1, 1, 1e9, 1e-18, 1, 1e9])     # w, toxe, xj in nm, ndep in 1e18 cm^-3
PINN_CUR = ["id", "ig", "is", "ib"]
PINN_GROUP_COLS = [(0, 2), (1, 1), (3, 3)]
KB_EV = 8.617333e-5
LN10 = float(np.log(10))
PINN_BS, PINN_BETA, PINN_HIDDEN, PINN_NCOLL = 128, 0.05, (256, 128, 64), 64

PINN_DIRS = {       # derivative directions: per decade of W / toxe, +50 mV of |vth0|, +100 K
    "log10_w": lambda R: _onehot(R, 2, R[..., 2] * LN10),
    "log10_toxe": lambda R: _onehot(R, 6, R[..., 6] * LN10),
    "abs_vth0": lambda R: _onehot(R, 5, torch.sign(R[..., 5]) * 0.05),
    "temp": lambda R: _onehot(R, 1, torch.full_like(R[..., 1], 100.0)),
}
# monotonicity constraints verified on the data: (direction, current, sign, rows)
PINN_CONSTRAINTS = {
    "nmos_on": [("log10_w", c, +1, "all") for c in PINN_CUR] + [("log10_toxe", c, -1, "all") for c in ("ig", "ib")]
               + [("abs_vth0", c, -1, "all") for c in ("id", "is", "ib")],
    "nmos_off": [("log10_w", c, +1, "all") for c in ("id", "is", "ig")] + [("log10_toxe", "ig", -1, "all")]
                + [("log10_toxe", c, -1, "drain") for c in ("id", "is")],
    "pmos_on": [("log10_w", c, +1, "all") for c in PINN_CUR] + [("log10_toxe", c, -1, "all") for c in ("ig", "ib")]
               + [("abs_vth0", c, -1, "all") for c in PINN_CUR] + [("temp", "ig", +1, "all")],
    "pmos_off": [("log10_w", c, +1, "all") for c in ("is", "ig")] + [("log10_toxe", "ig", -1, "all")],
}
# physics variants trained together; per output group the one with the lowest validation median error is kept
PINN_VARIANTS = [
    {"name": "kcl0.01", "kcl": 0.01},
    {"name": "kcl0.1", "kcl": 0.1},
    {"name": "mono1", "mono": 1.0},
    {"name": "struct", "struct": True},
    {"name": "all", "kcl": 0.01, "mono": 0.1, "struct": True},
]


def _onehot(R, j, val):
    t = torch.zeros_like(R)
    t[..., j] = val
    return t


class PINNPhysics:
    """Torch version of the 20 features (so derivatives w.r.t. physical inputs exist) + EKV channel prior."""

    def __init__(self, case, polarity):
        _, g, b = CASES[case]
        self.p = 1.0 if polarity == "nmos" else -1.0
        self.g_vdd, self.b_vdd = g == "vdd", b == "vdd"

    def rails(self, R):
        vdd = R[..., 0]
        z = torch.zeros_like(vdd)
        return (vdd if self.g_vdd else z), (vdd if self.b_vdd else z)

    def features(self, R):
        vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj = R.unbind(-1)
        vg, vb = self.rails(R)
        T = temp + 273.15
        Vt = KB_EV * T
        vds = vd - vs
        avds = vds.abs()
        es = torch.minimum(vd, vs) if self.p > 0 else torch.maximum(vd, vs)
        vgs = vg - es
        lvt = torch.log10(torch.clamp(-torch.expm1(-avds / Vt), min=1e-12))
        return torch.stack([vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj, vds, avds, vgs, vb - es, 1 / T,
                            (vgs.abs() - vth0.abs()) / Vt, (vg - vd).abs() / toxe, (vg - vs).abs() / toxe,
                            torch.log10(w), lvt], -1)

    def ekv(self, R, c0, n, eta):
        vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj = R.unbind(-1)
        vg, vb = self.rails(R)
        T = temp + 273.15
        Vt = KB_EV * T
        p = self.p
        vgb, vsb, vdb = p * (vg - vb), p * (vs - vb), p * (vd - vb)
        vp = (vgb - vth0.abs() + eta[:, None] * (vd - vs).abs()) / n[:, None]
        xa, xb = (vp - torch.minimum(vsb, vdb)) / Vt, (vp - torch.maximum(vsb, vdb)) / Vt
        lF = lambda x: 2 * torch.log(nn.functional.softplus(x / 2))
        la, lb = lF(xa), lF(xb)
        lnI = la + torch.log(torch.clamp(-torch.expm1(lb - la), min=1e-12))
        return c0[:, None] + (lnI + torch.log(n[:, None]) + 2 * torch.log(T)) / LN10 + torch.log10(w * u0 / toxe)


class BatchedPINN(nn.Module):
    """E = 3 x variants small MLPs trained at once (batched matmuls)."""

    def __init__(self, phys, E, fmu, fsd, gen):
        super().__init__()
        self.phys, self.E = phys, E
        self.register_buffer("fmu", fmu)
        self.register_buffer("fsd", fsd)
        sizes = [20, *PINN_HIDDEN, 2]
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for a, c in zip(sizes[:-1], sizes[1:]):
            bound = 1 / np.sqrt(a)
            self.W.append(nn.Parameter((torch.rand(E, a, c, generator=gen) * 2 - 1) * bound))
            self.b.append(nn.Parameter((torch.rand(E, 1, c, generator=gen) * 2 - 1) * bound))
        self.c0 = nn.Parameter(torch.zeros(E))
        self.a = nn.Parameter(torch.full((E,), -1.2))
        self.eta = nn.Parameter(torch.full((E,), 0.05))
        self.register_buffer("MU", torch.zeros(E, 2))
        self.register_buffer("SD", torch.ones(E, 2))
        self.register_buffer("smask", torch.zeros(E))
        self.use_struct = False

    def n(self):
        return 1 + nn.functional.softplus(self.a)

    def prior(self, R, sel=slice(None)):
        return self.phys.ekv(R, self.c0[sel], self.n()[sel], self.eta[sel])

    def z(self, R):
        return (self.phys.features(R) - self.fmu) / self.fsd

    def forward(self, R, x=None, sel=None):
        s = slice(None) if sel is None else sel
        E = self.E if sel is None else len(sel)
        if R.dim() == 2:
            R = R.expand(E, -1, -1)
        if x is None:
            x = self.z(R)
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b[s], x, w[s])
            if i < len(self.W) - 1:
                x = nn.functional.silu(x)
        out = self.MU[s][:, None, :] + self.SD[s][:, None, :] * x
        if self.use_struct:
            out = out + (self.smask[s][:, None] * self.prior(R, s))[..., None]
        return out


def _kcl_penalty(L4, S4):
    m = 10.0 ** (L4 - L4.max(-1, keepdim=True).values.detach())
    return ((S4 * m).sum(-1).abs() / m.sum(-1)).mean(-1)


class PINNModel:
    """Physics-informed MLP. One 256-128-64 SiLU MLP per output group on the 10 raw inputs (features computed in
    torch). Losses: Huber on log10|I| (+ row weights) + Kirchhoff penalty (sum of signed currents) + squared-hinge
    monotonicity penalty on collocation points (d log|I| / d W, toxe, |vth0|, T with verified signs) + EKV prior
    for id/is. Five physics variants are trained together; per group the lowest validation median error wins."""
    name = "pinn"

    def __init__(self, case, seed=MODEL_SEED, params=None):
        self.case, self.seed = case, seed
        self.polarity = CASES[case][0]
        self.epochs = (params or {}).get("epochs", 500)
        self.variants = PINN_VARIANTS
        self.phys = PINNPhysics(case, self.polarity)

    def _raw(self, df):
        return torch.tensor(df[PINN_RAW].to_numpy(float) * PINN_UNIT, dtype=torch.float32, device=DEVICE)

    def _assemble(self, out, V):
        o = out.view(V, 3, out.shape[1], 2)
        return torch.stack([o[:, 0, :, 0], o[:, 1, :, 0], o[:, 0, :, 1], o[:, 2, :, 0]], -1)

    def fit(self, X_train, y_train, X_val, y_val, df_train, df_val, w_train=None):
        V = len(self.variants)
        E = 3 * V
        Rtr, Rva = self._raw(df_train), self._raw(df_val)
        T = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float32, device=DEVICE)
        Ytr_t, Yva_t = T(y_train), T(y_val)
        Wtr = T(np.ones(len(df_train)) if w_train is None else w_train)
        Str = T(np.where(df_train[PINN_CUR].to_numpy() >= 0, 1.0, -1.0))
        lo, hi = Rtr.min(0).values, Rtr.max(0).values
        with torch.no_grad():
            F = self.phys.features(Rtr.double())
            fmu, fsd = F.mean(0).float(), F.std(0).float()
        gen = torch.Generator().manual_seed(1234 + self.seed)
        net = BatchedPINN(self.phys, E, fmu, fsd, gen).to(DEVICE)
        cols = torch.tensor([PINN_GROUP_COLS[e % 3] for e in range(E)], device=DEVICE)
        cmask = torch.tensor([[1.0, 1.0 if e % 3 == 0 else 0.0] for e in range(E)], device=DEVICE)
        lam_k = T([v.get("kcl", 0.0) for v in self.variants])
        lam_m = T([v.get("mono", 0.0) for v in self.variants])
        cons = PINN_CONSTRAINTS.get(self.case, [])
        use_mono, use_kcl = bool(cons) and float(lam_m.sum()) > 0, float(lam_k.sum()) > 0
        with torch.no_grad():
            net.smask.copy_(T([1.0 if (self.variants[e // 3].get("struct") and e % 3 == 0) else 0.0 for e in range(E)]))
            Yg = torch.gather(Ytr_t.expand(E, -1, -1), 2, cols[:, None, :].expand(E, len(Rtr), 2))
            net.use_struct = bool(net.smask.any())
            if net.use_struct:
                pr = net.prior(Rtr.expand(E, -1, -1))
                net.c0.copy_(torch.where(net.smask > 0, (Yg[..., 0] - pr).mean(1), torch.zeros_like(net.c0)))
                pr = net.prior(Rtr.expand(E, -1, -1)) * net.smask[:, None]
            else:
                pr = torch.zeros(E, len(Rtr), device=DEVICE)
            resid = Yg - pr[..., None]
            net.MU.copy_(resid.mean(1))
            net.SD.copy_(resid.std(1).clamp(min=1e-3))
            Ztr = net.z(Rtr)
        use_graph = DEVICE == "cuda" and os.environ.get("PINN_GRAPH", "1") == "1"
        opt = torch.optim.AdamW(net.parameters(), lr=torch.tensor(1e-3, device=DEVICE) if use_graph else 1e-3,
                                weight_decay=1e-4, capturable=use_graph)

        def huber(p, y, w=None):        # per-variant loss; w: [E, B] row weights (training only)
            h = nn.functional.smooth_l1_loss(p, y, beta=PINN_BETA, reduction="none") * cmask[:, None, :]
            if w is not None:
                h = h * w[..., None]
            return h.sum((1, 2)) / (cmask.sum(1) * p.shape[1])

        Yg_va = torch.gather(Yva_t.expand(E, -1, -1), 2, cols[:, None, :].expand(E, len(Rva), 2))
        best = torch.full((E,), float("inf"), device=DEVICE)
        best_state = [p.detach().clone() for p in net.parameters()]
        torch.manual_seed(self.seed)
        n = len(Rtr)
        nb = n // PINN_BS
        dirs = sorted({c[0] for c in cons})
        p_drain = (lambda R: R[..., 3] > R[..., 4]) if self.polarity == "nmos" else (lambda R: R[..., 3] < R[..., 4])
        mvar = [v for v, c in enumerate(self.variants) if c.get("mono")]
        msel = torch.tensor([3 * v + k for v in mvar for k in range(3)], device=DEVICE)
        lam_mv = lam_m[mvar] if mvar else None
        static_idx = torch.zeros(E, PINN_BS, dtype=torch.long, device=DEVICE)

        def train_step():
            idx = static_idx
            out = net(Rtr[idx], Ztr[idx])
            loss = huber(out, torch.gather(Yg, 1, idx[..., None].expand(-1, -1, 2)), Wtr[idx]).sum()
            if use_kcl:
                loss = loss + (lam_k * _kcl_penalty(self._assemble(out, V), Str[idx[0::3]])).sum()
            if use_mono:
                Rc = lo + (hi - lo) * torch.rand(len(dirs) * PINN_NCOLL, 10, device=DEVICE)
                Rc[:, 3:5] = torch.rand(len(Rc), 2, device=DEVICE) * Rc[:, :1]
                Tc = torch.cat([PINN_DIRS[d](Rc[k * PINN_NCOLL:(k + 1) * PINN_NCOLL]) for k, d in enumerate(dirs)])
                _, dO = jvp(lambda r: net(r, sel=msel), (Rc,), (Tc,))
                D4 = self._assemble(dO, len(mvar))
                pen = 0
                for d, cur, sgn, rows in cons:
                    k = dirs.index(d)
                    sl = slice(k * PINN_NCOLL, (k + 1) * PINN_NCOLL)
                    j = PINN_CUR.index(cur)
                    msk = torch.ones(PINN_NCOLL, device=DEVICE)
                    if rows == "drain":
                        dr = p_drain(Rc[sl])
                        msk = (dr if cur == "id" else ~dr).float()
                    pen = pen + ((torch.relu(-sgn * D4[:, sl, j]) ** 2) * msk).sum(1) / msk.sum().clamp(min=1)
                loss = loss + (lam_mv * pen).sum()
            loss.backward()
            opt.step()

        new_perm = lambda: torch.argsort(torch.rand(V, n, device=DEVICE), 1).repeat_interleave(3, 0)
        if use_graph:                    # capture one training step as a CUDA graph (launch overhead dominates)
            static_idx.copy_(new_perm()[:, :PINN_BS])
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    train_step()
            torch.cuda.current_stream().wait_stream(st)
            graph = torch.cuda.CUDAGraph()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                train_step()
        for ep in range(self.epochs):
            lr = 0.5e-3 * (1 + np.cos(np.pi * ep / self.epochs))
            if use_graph:
                opt.param_groups[0]["lr"].fill_(lr)
            else:
                opt.param_groups[0]["lr"] = lr
            perm = new_perm()
            for i in range(nb):
                static_idx.copy_(perm[:, i * PINN_BS:(i + 1) * PINN_BS])
                if use_graph:
                    graph.replay()
                else:
                    opt.zero_grad()
                    train_step()
            with torch.no_grad():
                vl = huber(net(Rva), Yg_va)
                better = vl < best
                best = torch.where(better, vl, best)
                for p, s in zip(net.parameters(), best_state):
                    s[better] = p.detach()[better]
        with torch.no_grad():
            for p, s in zip(net.parameters(), best_state):
                p.copy_(s)
        self.net, self.V = net, V
        # choose the variant per output group on validation median relative error
        Lva = self._predict_all(df_val)
        err = 100 * np.abs(10 ** (Lva - y_val[None]) - 1)
        med = np.median(err, 1)
        gmed = np.stack([med[:, [0, 2]].mean(1), med[:, 1], med[:, 3]], 1)
        self.choice = [int(np.argmin(gmed[:, g])) for g in range(3)]
        self.chosen_variants = {g: self.variants[v]["name"] for g, v in zip(("id+is", "ig", "ib"), self.choice)}
        self.net = net.cpu()            # keep on the CPU so the model can be saved and loaded anywhere
        return self

    def _predict_all(self, df):
        dev = next(self.net.parameters()).device
        R = torch.tensor(df[PINN_RAW].to_numpy(float) * PINN_UNIT, dtype=torch.float32, device=dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(R), 4096):
                out.append(self._assemble(self.net(R[i:i + 4096]), self.V).cpu().numpy())
        return np.concatenate(out, 1).astype(float)

    def predict(self, X, df):
        L = self._predict_all(df)
        v0, v1, v2 = self.choice
        return np.column_stack([L[v0, :, 0], L[v1, :, 1], L[v0, :, 2], L[v2, :, 3]])


MODELS = {
    "gpr": GPRModel,
    "hybrid_analytic_mlp": HybridAnalyticMLPModel,
    "resnet_mlp": ResNetMLPModel,
    "pinn": PINNModel,
}
