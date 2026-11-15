"""Component-decomposition model: predict each BSIM4 current component separately, rebuild signed terminal currents.

Why: in the OFF cases id and is change sign (channel current and junction/gate leakage cancel), so a model of
log10|I| per terminal has unbounded relative error near the zero crossing. Each BSIM4 component below has ONE sign
over a whole case (checked on all 10,000 rows of every case), and is smooth in log space; the terminal currents are
then signed sums, so their signs and Kirchhoff's law (id + ig + is + ib = 0) come out by construction.

Component data: component_data/<case>_components.csv (make_replay.py replays the 10,000 CSV rows in ngspice and
reads the BSIM4 instance currents). Terminal formulas (verified, fit_formulas.py / check in this file), with
T = +1 NMOS / -1 PMOS and the BSIM4 mode m = +1 if T*(vd - vs) >= 0 else -1 (component values as ngspice reports them):
    id = T * ( m*cid - ibd - igd + igidl + [m>0](isub - igcd) - [m<0] igcs )
    is = T * (-m*cid - ibs - igs + igisl + [m<0](isub - igcd) - [m>0] igcs )
    ig = T * ( igs + igd + igb + igcs + igcd )
    ib = T * ( ibd + ibs - isub - igidl - igisl - igb )
(the channel term T*m*cid = sign(vd - vs) * |cid|, i.e. the channel sign is sign(vd - vs)).

Device: CUDA if available (networks are trained on it, prediction on the CPU), else CPU.
Model : one MLP per component (all of a case's networks trained side by side as one batched model - each network has
        its own weights, optimiser state and best epoch; only the mini-batch order is shared), same recipe as
        Final_models: 256-128-64 SiLU, Huber beta 0.05 decades on a standardised log target, AdamW 1e-3 wd 1e-4,
        cosine LR, batch 128, 500 epochs, keeps each network's best validation epoch.
Inputs: the harness X (20 standardised features). Target log10(max(|c|, 1e-24)) (1e-24 A is far below every terminal).
        A component whose largest share of any terminal (|contribution| / sum of |contributions|) on the training rows
        is < 1e-6 is dropped; components that are exactly zero everywhere (GIDL/GISL in the ON cases) too.
Signs : one fixed sign per component per case (from the training rows; asserted single-signed).
"""
import copy, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
CDIR = HERE / "component_data"
COMP = ["id", "ibd", "ibs", "isub", "igidl", "igisl", "igs", "igd", "igb", "igcs", "igcd"]
KEY = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
FLOOR, EPOCHS, BS, BETA, HIDDEN = 1e-24, 500, 128, 0.05, (256, 128, 64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def coefficients(df, polarity):
    """(rows, 4 terminals, 11 components) matrix A with terminal currents = A @ components (ngspice values)."""
    T = 1.0 if polarity == "nmos" else -1.0
    m = np.where(T * (df["vd"].to_numpy() - df["vs"].to_numpy()) >= 0, 1.0, -1.0)
    p, n = (m > 0).astype(float), (m < 0).astype(float)
    A = np.zeros((len(df), 4, len(COMP))); i = {c: k for k, c in enumerate(COMP)}
    # id
    A[:, 0, i["id"]] = m; A[:, 0, i["ibd"]] = -1; A[:, 0, i["igd"]] = -1; A[:, 0, i["igidl"]] = 1
    A[:, 0, i["isub"]] = p; A[:, 0, i["igcd"]] = -p; A[:, 0, i["igcs"]] = -n
    # ig
    for c in ("igs", "igd", "igb", "igcs", "igcd"): A[:, 1, i[c]] = 1
    # is
    A[:, 2, i["id"]] = -m; A[:, 2, i["ibs"]] = -1; A[:, 2, i["igs"]] = -1; A[:, 2, i["igisl"]] = 1
    A[:, 2, i["isub"]] = n; A[:, 2, i["igcd"]] = -n; A[:, 2, i["igcs"]] = -p
    # ib
    A[:, 3, i["ibd"]] = 1; A[:, 3, i["ibs"]] = 1
    for c in ("isub", "igidl", "igisl", "igb"): A[:, 3, i[c]] = -1
    return T * A


def components_for(case, df):
    """True BSIM4 components for the rows of df (matched to the dataset CSV by the 12 input values; exact)."""
    raw = pd.read_csv(REPO / "Datasets" / "data" / f"{case}.csv")
    comp = pd.read_csv(CDIR / f"{case}_components.csv")
    assert len(raw) == len(comp) and (comp["row"].to_numpy() == np.arange(len(raw))).all()
    lut = pd.Series(np.arange(len(raw)), index=pd.MultiIndex.from_frame(raw[KEY]))
    assert lut.index.is_unique, f"{case}: duplicate input rows"
    idx = lut.reindex(pd.MultiIndex.from_frame(df[KEY])).to_numpy()
    assert not np.isnan(idx.astype(float)).any(), f"{case}: rows not found in the component data"
    idx = idx.astype(int)
    assert np.array_equal(raw.loc[idx, ["id", "ig", "is", "ib"]].to_numpy(), df[["id", "ig", "is", "ib"]].to_numpy())
    return comp.loc[idx, [f"c_{c}" for c in COMP]].to_numpy(), idx


class BatchedMLP(nn.Module):
    """G independent 256-128-64 SiLU networks (one output each), evaluated together with batched matmuls.
    Initialised exactly like G separate nn.Linear stacks."""
    def __init__(self, n_in, G, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        nets = [nn.Sequential(*[l for a, b in zip((n_in,) + HIDDEN, HIDDEN + (1,)) for l in (nn.Linear(a, b),)]) for _ in range(G)]
        self.W = nn.ParameterList(); self.b = nn.ParameterList()
        for k in range(len(HIDDEN) + 1):
            self.W.append(nn.Parameter(torch.stack([n[k].weight.data.T.clone() for n in nets])))   # (G, in, out)
            self.b.append(nn.Parameter(torch.stack([n[k].bias.data.clone() for n in nets])[:, None, :]))  # (G, 1, out)

    def forward(self, x):                      # x (B, in) -> (G, B)
        h = x.unsqueeze(0).expand(self.W[0].shape[0], -1, -1)
        for k in range(len(self.W)):
            h = torch.baddbmm(self.b[k], h, self.W[k])
            if k < len(self.W) - 1: h = nn.functional.silu(h)
        return h[..., 0]


def train_batched(Xtr, Ytr, Xva, Yva, seed=0, log=None):
    """Ytr (rows, G) in decades. Same recipe as Final_models.train_net, per network."""
    G = Ytr.shape[1]
    mu, sd = Ytr.mean(0), Ytr.std(0).clip(1e-6)
    Xt, yt, Xv, yv = (torch.tensor(a, dtype=torch.float32, device=DEV) for a in (Xtr, ((Ytr - mu) / sd).T, Xva, Yva.T))
    MU, SD = (torch.tensor(a, dtype=torch.float32, device=DEV)[:, None] for a in (mu, sd))
    net = BatchedMLP(Xt.shape[1], G, seed).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    huber = lambda a, b: nn.functional.smooth_l1_loss(a, b, beta=BETA, reduction="none").mean(1)   # per network
    best = np.full(G, np.inf); best_ep = np.zeros(G, int)
    best_state = [p.detach().clone() for p in net.parameters()]
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]
            loss = huber(MU + SD * net(Xt[idx]), MU + SD * yt[:, idx]).sum()   # sum of the per-network mean losses
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            vl = huber(MU + SD * net(Xv), yv).cpu().numpy()
        better = vl < best
        if better.any():
            best[better] = vl[better]; best_ep[better] = ep
            g = torch.tensor(better, device=DEV)
            for bs, p in zip(best_state, net.parameters()): bs[g] = p.detach()[g]
        if log and (ep % 100 == 99): log(f"    epoch {ep+1}: val Huber (dec) {np.array2string(vl, precision=4)}  {time.time()-t0:.0f}s")
    with torch.no_grad():
        for bs, p in zip(best_state, net.parameters()): p.copy_(bs)
    net = net.cpu().eval()
    return net, mu, sd, best, best_ep


class ComponentModel:
    VFACTOR, SUFFIX = {}, ""        # see ComponentModelVF

    def factor(self, df):
        """Known log10 voltage factor per kept component (zeros in the base model)."""
        F = np.zeros((len(df), len(self.keep)))
        for j, k in enumerate(self.keep):
            if COMP[k] in self.VFACTOR:
                a, b = self.VFACTOR[COMP[k]]
                vt = 8.617333262e-5 * (df["temp"].to_numpy() + 273.15)
                F[:, j] = np.log10(np.maximum(-np.expm1(-np.abs(df[a].to_numpy() - df[b].to_numpy()) / vt), 1e-12))
        return F

    def __init__(self, case, polarity):
        self.case, self.polarity, self.info = case, polarity, {}

    def fit(self, data):
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
        Xtr, _, dtr = data["train"]; Xva, _, dva = data["validation"]
        Ctr, _ = components_for(self.case, dtr); Cva, _ = components_for(self.case, dva)
        # sanity: the formulas rebuild the CSV currents of the training rows
        rec = np.einsum("rtc,rc->rt", coefficients(dtr, self.polarity), Ctr); true = dtr[["id", "ig", "is", "ib"]].to_numpy()
        ok = np.abs(rec - true) <= 5e-6 * np.abs(true) + 6e-15            # 6e-15: CSV branch-current quantisation (nmos_on ig, pmos_on ib)
        assert ok.mean() > 0.999, f"formula check failed ({ok.mean():.4f})"
        # which components matter, and their signs
        contrib = np.abs(coefficients(dtr, self.polarity) * Ctr[:, None, :])                      # (rows, 4, 11)
        share = (contrib / contrib.sum(2, keepdims=True).clip(1e-300)).max((0, 1))
        self.keep = [k for k in range(len(COMP)) if share[k] >= 1e-6 and (Ctr[:, k] != 0).any()]
        sg = np.sign(Ctr[:, self.keep]); self.sign = np.zeros(len(COMP))
        for j, k in enumerate(self.keep):
            nz = sg[:, j][sg[:, j] != 0]; assert (nz == nz[0]).all(), f"{COMP[k]} changes sign"; self.sign[k] = nz[0]
        Ytr = np.log10(np.maximum(np.abs(Ctr[:, self.keep]), FLOOR)) - self.factor(dtr)
        Yva = np.log10(np.maximum(np.abs(Cva[:, self.keep]), FLOOR)) - self.factor(dva)
        self.lo, self.hi = Ytr.min(0) - 1, Ytr.max(0) + 1
        print(f"  [{self.case}] components {[COMP[k] for k in self.keep]} (dropped {[COMP[k] for k in range(len(COMP)) if k not in self.keep]})", flush=True)
        self.net, self.mu, self.sd, best, best_ep = train_batched(Xtr, Ytr, Xva, Yva, seed=0, log=lambda s: print(s, flush=True))
        self.info = {"components": [COMP[k] for k in self.keep], "signs": {COMP[k]: int(self.sign[k]) for k in self.keep},
                     "val_huber_dec": {COMP[k]: round(float(b), 5) for k, b in zip(self.keep, best)},
                     "best_epoch": {COMP[k]: int(e) for k, e in zip(self.keep, best_ep)},
                     "max_share": {COMP[k]: float(f"{share[k]:.2e}") for k in range(len(COMP))}}
        (CDIR / "models").mkdir(exist_ok=True); self.save(CDIR / "models" / f"{self.case}{self.SUFFIX}.pt")
        return self

    def predict_components(self, X, df):
        """(rows, 11) signed component currents in ngspice's convention (dropped ones = 0)."""
        with torch.no_grad():
            z = self.net(torch.tensor(X, dtype=torch.float32)).numpy().T
        L = np.clip(self.mu + self.sd * z, self.lo, self.hi) + self.factor(df)
        C = np.zeros((len(X), len(COMP))); C[:, self.keep] = self.sign[self.keep] * 10.0 ** L
        return C

    def predict_signed(self, X, df):
        return np.einsum("rtc,rc->rt", coefficients(df, self.polarity), self.predict_components(X, df))

    def predict(self, X, df):
        return np.log10(np.maximum(np.abs(self.predict_signed(X, df)), 1e-18))

    def save(self, path):
        torch.save({"case": self.case, "polarity": self.polarity, "keep": self.keep, "sign": self.sign, "mu": self.mu,
                    "sd": self.sd, "lo": self.lo, "hi": self.hi, "state": self.net.state_dict(), "info": self.info}, path)

    @classmethod
    def load(cls, path, n_in=20):
        s = torch.load(path, weights_only=False); m = cls(s["case"], s["polarity"])
        for k in ("keep", "sign", "mu", "sd", "lo", "hi", "info"): setattr(m, k, s[k])
        m.net = BatchedMLP(n_in, len(m.keep)); m.net.load_state_dict(s["state"]); m.net.eval()
        return m


class ComponentModelVF(ComponentModel):
    """Variant (not the registry model): same X and recipe, but the gate-overlap and junction components get a known
    voltage factor f = 1 - exp(-|V|/Vt) (V = vgd, vgs, vbd, vbs from df; Vt = kT/q): the network predicts
    log10|c| - log10 f and log10 f is added back. These currents go to zero linearly as their voltage -> 0
    (fitted slope ~1 on rows with |V| < 10 mV), a log-singularity the 20 features cannot express."""
    VFACTOR = {"igd": ("vg", "vd"), "igs": ("vg", "vs"), "ibd": ("vb", "vd"), "ibs": ("vb", "vs")}
    SUFFIX = "_vf"
