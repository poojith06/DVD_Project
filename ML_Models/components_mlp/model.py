"""Component-decomposition model: predict each BSIM4 current component, then rebuild the signed terminal currents.

Why: in the OFF cases id and is change sign (channel current and junction/gate leakage cancel), so a model of
log10|I| per terminal has unbounded relative error near the zero crossing. Each BSIM4 component below has ONE sign
over a whole case, and is smooth in log space; the terminal currents are signed sums of them, so their signs and
Kirchhoff's law (id + ig + is + ib = 0) come out by construction - no sign model is needed.

Training labels: component_data/<case>_components.csv - the BSIM4 instance currents (@m1[id], ibd, ibs, isub, igidl,
igisl, igs, igd, igb, igcs, igcd) of every dataset row, read in ngspice by make_components.py.
Terminal formulas (T = +1 NMOS / -1 PMOS, BSIM4 mode m = +1 if T*(vd - vs) >= 0 else -1):
    id = T * ( m*cid - ibd - igd + igidl + [m>0](isub - igcd) - [m<0] igcs )
    is = T * (-m*cid - ibs - igs + igisl + [m<0](isub - igcd) - [m>0] igcs )
    ig = T * ( igs + igd + igb + igcs + igcd )
    ib = T * ( ibd + ibs - isub - igidl - igisl - igb )
Model : one MLP (256-128-64 SiLU) per kept component, all trained side by side as one batched model (each network has
        its own weights and best epoch), same recipe as common/nets.py.
        Target log10|c| - log10 f, with the known voltage factor f = 1 - exp(-|V|/Vt) for the gate-overlap and
        junction components (V = vgd, vgs, vbd, vbs): these go to zero linearly as V -> 0, which the features
        cannot express in log space ("_vf" variant in Model_comparison/).
        Components that never reach 1e-6 of any terminal current (or are exactly zero) are dropped.
NOTE: needs the component labels, which are internal simulator quantities - confirm with the TA that this is allowed.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common.config import CASES, DATA_DIR, INPUT_COLUMNS, MODEL_SEED, TARGET_COLUMNS
from common.nets import DEVICE, NET_PARAMS

COMPONENT_DIR = Path(__file__).resolve().parent / "component_data"
COMP = ["id", "ibd", "ibs", "isub", "igidl", "igisl", "igs", "igd", "igb", "igcs", "igcd"]
VFACTOR = {"igd": ("vg", "vd"), "igs": ("vg", "vs"), "ibd": ("vb", "vd"), "ibs": ("vb", "vs")}
HIDDEN, FLOOR = (256, 128, 64), 1e-24


def coefficients(df, polarity):
    """(rows, 4 terminals, 11 components) matrix A with terminal currents = A @ components."""
    T = 1.0 if polarity == "nmos" else -1.0
    m = np.where(T * (df["vd"].to_numpy() - df["vs"].to_numpy()) >= 0, 1.0, -1.0)
    p, n = (m > 0).astype(float), (m < 0).astype(float)
    A = np.zeros((len(df), 4, len(COMP)))
    i = {c: k for k, c in enumerate(COMP)}
    A[:, 0, i["id"]] = m; A[:, 0, i["ibd"]] = -1; A[:, 0, i["igd"]] = -1; A[:, 0, i["igidl"]] = 1      # id
    A[:, 0, i["isub"]] = p; A[:, 0, i["igcd"]] = -p; A[:, 0, i["igcs"]] = -n
    for c in ("igs", "igd", "igb", "igcs", "igcd"):                                                    # ig
        A[:, 1, i[c]] = 1
    A[:, 2, i["id"]] = -m; A[:, 2, i["ibs"]] = -1; A[:, 2, i["igs"]] = -1; A[:, 2, i["igisl"]] = 1     # is
    A[:, 2, i["isub"]] = n; A[:, 2, i["igcd"]] = -n; A[:, 2, i["igcs"]] = -p
    A[:, 3, i["ibd"]] = 1; A[:, 3, i["ibs"]] = 1                                                       # ib
    for c in ("isub", "igidl", "igisl", "igb"):
        A[:, 3, i[c]] = -1
    return T * A


def components_for(case, df, data_dir=DATA_DIR):
    """BSIM4 components for the rows of df, matched to the dataset CSV by the 12 input values (exact match)."""
    raw = pd.read_csv(Path(data_dir) / f"{case}.csv")
    comp = pd.read_csv(COMPONENT_DIR / f"{case}_components.csv")
    if len(raw) != len(comp) or not (comp["row"].to_numpy() == np.arange(len(raw))).all():
        raise ValueError(f"{case}: component_data does not match {data_dir}/{case}.csv - rerun make_components.py")
    lut = pd.Series(np.arange(len(raw)), index=pd.MultiIndex.from_frame(raw[INPUT_COLUMNS]))
    idx = lut.reindex(pd.MultiIndex.from_frame(df[INPUT_COLUMNS])).to_numpy()
    if np.isnan(idx.astype(float)).any():
        raise ValueError(f"{case}: some rows are not in the component data - rerun make_components.py")
    idx = idx.astype(int)
    return comp.loc[idx, [f"c_{c}" for c in COMP]].to_numpy()


class BatchedMLP(nn.Module):
    """G independent 256-128-64 SiLU networks (one output each), evaluated together with batched matmuls."""

    def __init__(self, n_in, G, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        sizes = (n_in,) + HIDDEN + (1,)
        nets = [[nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])] for _ in range(G)]
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for k in range(len(sizes) - 1):
            self.W.append(nn.Parameter(torch.stack([n[k].weight.data.T.clone() for n in nets])))           # (G, in, out)
            self.b.append(nn.Parameter(torch.stack([n[k].bias.data.clone() for n in nets])[:, None, :]))   # (G, 1, out)

    def forward(self, x):  # x (B, in) -> (G, B)
        h = x.unsqueeze(0).expand(self.W[0].shape[0], -1, -1)
        for k in range(len(self.W)):
            h = torch.baddbmm(self.b[k], h, self.W[k])
            if k < len(self.W) - 1:
                h = nn.functional.silu(h)
        return h[..., 0]


def train_batched(X_train, Y_train, X_val, Y_val, seed=0, params=None):
    """Y (rows, G) in decades. Same recipe as common/nets.train_net, separately for every network."""
    p = {**NET_PARAMS, **(params or {})}
    G = Y_train.shape[1]
    mu, sd = Y_train.mean(0), Y_train.std(0).clip(1e-6)
    Xt, yt, Xv, yv = (torch.tensor(a, dtype=torch.float32, device=DEVICE)
                      for a in (X_train, ((Y_train - mu) / sd).T, X_val, Y_val.T))
    MU, SD = (torch.tensor(a, dtype=torch.float32, device=DEVICE)[:, None] for a in (mu, sd))
    net = BatchedMLP(Xt.shape[1], G, seed).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, p["epochs"])

    def huber(a, b):  # mean loss per network
        return nn.functional.smooth_l1_loss(a, b, beta=p["huber_beta"], reduction="none").mean(1)

    best = np.full(G, np.inf)
    best_state = [q.detach().clone() for q in net.parameters()]
    t0 = time.time()
    for ep in range(p["epochs"]):
        net.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), p["batch_size"]):
            idx = perm[i:i + p["batch_size"]]
            loss = huber(MU + SD * net(Xt[idx]), MU + SD * yt[:, idx]).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            val = huber(MU + SD * net(Xv), yv).cpu().numpy()
        better = val < best
        if better.any():
            best[better] = val[better]
            g = torch.tensor(better, device=DEVICE)
            for bs, q in zip(best_state, net.parameters()):
                bs[g] = q.detach()[g]
        if ep % 100 == 99:
            print(f"    epoch {ep + 1}: val Huber (decades) per component {np.array2string(val, precision=4)}"
                  f"  {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        for bs, q in zip(best_state, net.parameters()):
            q.copy_(bs)
    return net.cpu().eval(), mu, sd


class ComponentsMLPModel:
    name = "components_mlp"

    def __init__(self, case, seed=MODEL_SEED, params=None):
        self.case, self.seed, self.params = case, seed, params
        self.polarity = CASES[case][0]

    def _factor(self, df):
        """log10 of the known voltage factor of each kept component (0 for components without one)."""
        F = np.zeros((len(df), len(self.keep)))
        vt = 8.617333262e-5 * (df["temp"].to_numpy() + 273.15)
        for j, k in enumerate(self.keep):
            if COMP[k] in VFACTOR:
                a, b = VFACTOR[COMP[k]]
                v = np.abs(df[a].to_numpy() - df[b].to_numpy())
                F[:, j] = np.log10(np.maximum(-np.expm1(-v / vt), 1e-12))
        return F

    def fit(self, X_train, y_train, X_val, y_val, df_train, df_val):
        C_train, C_val = components_for(self.case, df_train), components_for(self.case, df_val)
        A = coefficients(df_train, self.polarity)
        rebuilt = np.einsum("rtc,rc->rt", A, C_train)
        true = df_train[TARGET_COLUMNS].to_numpy()
        ok = np.abs(rebuilt - true) <= 5e-6 * np.abs(true) + 6e-15   # 6e-15: CSV branch-current rounding
        if ok.mean() < 0.999:
            raise ValueError(f"{self.case}: the component formulas do not rebuild the CSV currents ({ok.mean():.4f})")
        # keep the components that matter, each with its one sign
        contrib = np.abs(A * C_train[:, None, :])
        share = (contrib / contrib.sum(2, keepdims=True).clip(1e-300)).max((0, 1))
        self.keep = [k for k in range(len(COMP)) if share[k] >= 1e-6 and (C_train[:, k] != 0).any()]
        self.sign = np.zeros(len(COMP))
        for k in self.keep:
            nz = np.sign(C_train[:, k])[C_train[:, k] != 0]
            if not (nz == nz[0]).all():
                raise ValueError(f"{self.case}: component {COMP[k]} changes sign")
            self.sign[k] = nz[0]
        Y_tr = np.log10(np.maximum(np.abs(C_train[:, self.keep]), FLOOR)) - self._factor(df_train)
        Y_va = np.log10(np.maximum(np.abs(C_val[:, self.keep]), FLOOR)) - self._factor(df_val)
        self.lo, self.hi = Y_tr.min(0) - 1, Y_tr.max(0) + 1     # predictions clipped to the training range +- 1 decade
        print(f"  [{self.case}] components {[COMP[k] for k in self.keep]}", flush=True)
        self.net, self.mu, self.sd = train_batched(X_train, Y_tr, X_val, Y_va, self.seed, self.params)
        return self

    def predict_components(self, X, df):
        """(rows, 11) signed component currents (dropped components = 0)."""
        with torch.no_grad():
            z = self.net(torch.tensor(np.asarray(X, float), dtype=torch.float32)).numpy().T
        L = np.clip(self.mu + self.sd * z, self.lo, self.hi) + self._factor(df)
        C = np.zeros((len(X), len(COMP)))
        C[:, self.keep] = self.sign[self.keep] * 10.0 ** L
        return C

    def predict_signed(self, X, df):
        return np.einsum("rtc,rc->rt", coefficients(df, self.polarity), self.predict_components(X, df))

    def predict(self, X, df):
        return np.log10(np.maximum(np.abs(self.predict_signed(X, df)), 1e-18))
