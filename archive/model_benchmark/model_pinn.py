"""Physics-informed neural network (PINN) for the four terminal currents of one transistor case.

Network : same body as Final_models - one 256-128-64 SiLU MLP per output group (id+is), ig, ib, target log10|I| in
          decades (standardised for training), AdamW lr 1e-3 wd 1e-4, cosine LR, batch 128, 500 epochs, each network keeps
          its best validation epoch. Difference: the network takes the 10 RAW physical inputs and computes the repo's 20
          features inside torch (same formulas as common/features.py), so derivatives w.r.t. physical inputs exist.
Losses  : data   - Huber (beta 0.05 decades) on log10|I|, as the reference
          kcl    - Kirchhoff: with the true training signs s_k, |sum_k s_k 10^pred_k| / sum_k 10^pred_k (row-max scaled),
                   on the training batch; couples the three networks of one model
          mono   - monotonicity on collocation points drawn uniformly in the input box (vd, vs uniform in 0..vdd):
                   forward-mode autograd (torch.func.jvp) gives d log10|I| / d(physical input); a squared hinge
                   penalises the wrong sign. Only constraints whose sign was verified on the data are used (CONSTRAINTS).
          struct - physics-structured output for id/is: log10|I| = EKV channel-current prior (all regions, symmetric in
                   drain/source; learnable I0, slope factor n, DIBL eta) + MLP correction.
Ablation: all variants (data-only x3 seeds, +kcl 0.01/0.1, +mono, +struct, all) are trained at once as one batched GPU
          model (one CUDA-graph-captured step). Per group the physics variant (not data-only) with the lowest validation
          median relative error is used; info holds the whole validation ablation and the constraint check.
          Signs and metrics are the harness's.
"""
import os, time
import numpy as np, torch
from torch import nn
from torch.func import jvp
from common.config import CASES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RAW = ["vdd", "temp", "w", "vd", "vs", "vth0", "toxe", "ndep", "u0", "xj"]
UNIT = np.array([1, 1, 1e9, 1, 1, 1, 1e9, 1e-18, 1, 1e9])     # w, toxe, xj in nm, ndep in 1e18 cm^-3
CUR = ["id", "ig", "is", "ib"]
GROUP_COLS = [(0, 2), (1, 1), (3, 3)]                            # outputs of the (id+is), ig, ib networks
KB = 8.617333e-5
LN10 = float(np.log(10))
EPOCHS = int(os.environ.get("PINN_EPOCHS", 500)); BS, BETA, HIDDEN = 128, 0.05, (256, 128, 64)
N_COLL = 64                                                      # collocation points per direction per step

# directions for derivatives: name -> (raw column, tangent per row as function of raw R in UNIT units)
DIRS = {
    "log10_w":    lambda R: _onehot(R, 2, R[..., 2] * LN10),      # per decade of W
    "log10_toxe": lambda R: _onehot(R, 6, R[..., 6] * LN10),      # per decade of toxe
    "abs_vth0":   lambda R: _onehot(R, 5, torch.sign(R[..., 5]) * 0.05),   # per +50 mV of |vth0|
    "temp":       lambda R: _onehot(R, 1, torch.full_like(R[..., 1], 100.0)),  # per +100 K
}
# verified monotonicity constraints: (direction, current, sign of derivative, rows) - rows "drain" = only where that
# terminal is the effective drain (OFF: the effective source current can cross zero, so no constraint there)
CONSTRAINTS = {     # kept if >= 99 % of validation rows agree under a data-only network (see report / info)
    "nmos_on":  [("log10_w", c, +1, "all") for c in CUR] + [("log10_toxe", c, -1, "all") for c in ("ig", "ib")]
                + [("abs_vth0", c, -1, "all") for c in ("id", "is", "ib")],
    "nmos_off": [("log10_w", c, +1, "all") for c in ("id", "is", "ig")] + [("log10_toxe", "ig", -1, "all")]
                + [("log10_toxe", c, -1, "drain") for c in ("id", "is")],
    "pmos_on":  [("log10_w", c, +1, "all") for c in CUR] + [("log10_toxe", c, -1, "all") for c in ("ig", "ib")]
                + [("abs_vth0", c, -1, "all") for c in CUR] + [("temp", "ig", +1, "all")],
    "pmos_off": [("log10_w", c, +1, "all") for c in ("is", "ig")] + [("log10_toxe", "ig", -1, "all")],
}
VARIANTS = [
    {"name": "data",      "seed_tag": 0},
    {"name": "data_s1",   "seed_tag": 1},
    {"name": "data_s2",   "seed_tag": 2},
    {"name": "kcl0.01",   "kcl": 0.01},
    {"name": "kcl0.1",    "kcl": 0.1},
    {"name": "mono1",     "mono": 1.0},
    {"name": "struct",    "struct": True},
    {"name": "all",       "kcl": 0.01, "mono": 0.1, "struct": True},
]


def _onehot(R, j, val):
    T = torch.zeros_like(R); T[..., j] = val; return T


class Physics:
    """Torch version of common.features.build_features (+ EKV prior) on raw inputs R [..., 10] in UNIT units."""

    def __init__(self, case, polarity):
        _, g, b = CASES[case]
        self.p = 1.0 if polarity == "nmos" else -1.0
        self.g_vdd, self.b_vdd = g == "vdd", b == "vdd"

    def rails(self, R):
        vdd = R[..., 0]; z = torch.zeros_like(vdd)
        return (vdd if self.g_vdd else z), (vdd if self.b_vdd else z)

    def features(self, R):
        vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj = R.unbind(-1)
        vg, vb = self.rails(R)
        T = temp + 273.15; Vt = KB * T
        vds = vd - vs; avds = vds.abs()
        es = torch.minimum(vd, vs) if self.p > 0 else torch.maximum(vd, vs)
        vgs = vg - es
        lvt = torch.log10(torch.clamp(-torch.expm1(-avds / Vt), min=1e-12))
        return torch.stack([vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj, vds, avds, vgs, vb - es, 1 / T,
                            (vgs.abs() - vth0.abs()) / Vt, (vg - vd).abs() / toxe, (vg - vs).abs() / toxe,
                            torch.log10(w), lvt], -1)

    def ekv(self, R, c0, n, eta):
        """log10 of an EKV channel current: I0 W u0/toxe T^2 [F(xs) - F(xd)], F(x) = ln^2(1 + e^(x/2)),
        x = (vp - V_terminal,b)/Vt, vp = (Vgb - |vth0| + eta |vds|)/n. c0, n, eta: [E] (broadcast over rows)."""
        vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj = R.unbind(-1)
        vg, vb = self.rails(R)
        T = temp + 273.15; Vt = KB * T; p = self.p
        vgb, vsb, vdb = p * (vg - vb), p * (vs - vb), p * (vd - vb)
        vp = (vgb - vth0.abs() + eta[:, None] * (vd - vs).abs()) / n[:, None]
        xa, xb = (vp - torch.minimum(vsb, vdb)) / Vt, (vp - torch.maximum(vsb, vdb)) / Vt   # xa >= xb
        lF = lambda x: 2 * torch.log(nn.functional.softplus(x / 2))
        la, lb = lF(xa), lF(xb)
        lnI = la + torch.log(torch.clamp(-torch.expm1(lb - la), min=1e-12))
        return c0[:, None] + (lnI + torch.log(n[:, None]) + 2 * torch.log(T)) / LN10 + torch.log10(w * u0 / toxe)


class BatchedPINN(nn.Module):
    def __init__(self, phys, E, fmu, fsd, gen):
        super().__init__()
        self.phys, self.E = phys, E
        self.register_buffer("fmu", fmu); self.register_buffer("fsd", fsd)
        sizes = [20, *HIDDEN, 2]
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for a, c in zip(sizes[:-1], sizes[1:]):
            bound = 1 / np.sqrt(a)
            self.W.append(nn.Parameter((torch.rand(E, a, c, generator=gen) * 2 - 1) * bound))
            self.b.append(nn.Parameter((torch.rand(E, 1, c, generator=gen) * 2 - 1) * bound))
        self.c0 = nn.Parameter(torch.zeros(E)); self.a = nn.Parameter(torch.full((E,), -1.2)); self.eta = nn.Parameter(torch.full((E,), 0.05))
        for nm in ("MU", "SD", "smask"): self.register_buffer(nm, torch.zeros(E, 2) if nm != "smask" else torch.zeros(E))
        self.SD.fill_(1.0); self.use_struct = False          # python flag (no GPU sync in forward)

    def n(self): return 1 + nn.functional.softplus(self.a)     # init ~1.26

    def prior(self, R, sel=slice(None)):    # R [E', N, 10] -> [E', N]
        return self.phys.ekv(R, self.c0[sel], self.n()[sel], self.eta[sel])

    def z(self, R): return (self.phys.features(R) - self.fmu) / self.fsd

    def forward(self, R, x=None, sel=None):
        """R [N,10] or [E,N,10] -> log10|I| [E, N, 2]; x: precomputed standardised features;
        sel: index tensor of a subset of the networks (used for the derivative penalty, cost scales with #networks)."""
        s = slice(None) if sel is None else sel
        E = self.E if sel is None else len(sel)
        if R.dim() == 2: R = R.expand(E, -1, -1)
        if x is None: x = self.z(R)
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b[s], x, w[s])
            if i < len(self.W) - 1: x = nn.functional.silu(x)
        out = self.MU[s][:, None, :] + self.SD[s][:, None, :] * x
        if self.use_struct: out = out + (self.smask[s][:, None] * self.prior(R, s))[..., None]
        return out


def kcl_penalty(L4, S4):
    """L4 log10|I| [V, B, 4], S4 true signs [B, 4] -> per-variant mean |sum s I| / sum |I| (row-max scaled)."""
    m = 10.0 ** (L4 - L4.max(-1, keepdim=True).values.detach())
    return ((S4 * m).sum(-1).abs() / m.sum(-1)).mean(-1)


class PINNModel:
    def __init__(self, case, polarity, variants=None, epochs=None, verbose=True):
        self.case, self.polarity = case, polarity
        self.phys = Physics(case, polarity)
        self.variants = variants or VARIANTS; self.epochs = epochs or EPOCHS; self.verbose = verbose
        self.info = {}

    def raw(self, df):
        return torch.tensor(df[RAW].to_numpy(float) * UNIT, dtype=torch.float32, device=DEV)

    def _assemble(self, out, V):
        """[E=3V, N, 2] -> [V, N, 4] in the order id, ig, is, ib."""
        o = out.view(V, 3, out.shape[1], 2)
        return torch.stack([o[:, 0, :, 0], o[:, 1, :, 0], o[:, 0, :, 1], o[:, 2, :, 0]], -1)

    def fit(self, data):
        t0 = time.time()
        _, Ytr, dtr = data["train"]; _, Yva, dva = data["validation"]
        V = len(self.variants); E = 3 * V
        Rtr, Rva = self.raw(dtr), self.raw(dva)
        T = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float32, device=DEV)
        Ytr_t, Yva_t = T(Ytr), T(Yva)
        Str = T(np.where(dtr[CUR].to_numpy() >= 0, 1.0, -1.0))
        lo, hi = Rtr.min(0).values, Rtr.max(0).values
        with torch.no_grad():
            F = self.phys.features(Rtr.double()); fmu, fsd = F.mean(0).float(), F.std(0).float()
        gen = torch.Generator().manual_seed(1234)
        net = BatchedPINN(self.phys, E, fmu, fsd, gen).to(DEV)
        cols = torch.tensor([GROUP_COLS[e % 3] for e in range(E)], device=DEV)          # [E, 2]
        cmask = torch.tensor([[1.0, 1.0 if e % 3 == 0 else 0.0] for e in range(E)], device=DEV)
        lam_k = T([v.get("kcl", 0.0) for v in self.variants]); lam_m = T([v.get("mono", 0.0) for v in self.variants])
        cons = CONSTRAINTS.get(self.case, [])
        use_mono = bool(cons) and float(lam_m.sum()) > 0; use_kcl = float(lam_k.sum()) > 0
        with torch.no_grad():
            net.smask.copy_(T([1.0 if (self.variants[e // 3].get("struct") and e % 3 == 0) else 0.0 for e in range(E)]))
            Yg = torch.gather(Ytr_t.expand(E, -1, -1), 2, cols[:, None, :].expand(E, len(Rtr), 2))   # [E, n, 2]
            net.use_struct = bool(net.smask.any())
            if net.use_struct:                          # init I0 so the prior matches the mean channel current
                pr = net.prior(Rtr.expand(E, -1, -1))
                net.c0.copy_(torch.where(net.smask > 0, (Yg[..., 0] - pr).mean(1), torch.zeros_like(net.c0)))
                pr = net.prior(Rtr.expand(E, -1, -1)) * net.smask[:, None]
            else:
                pr = torch.zeros(E, len(Rtr), device=DEV)
            resid = Yg - pr[..., None]
            net.MU.copy_(resid.mean(1)); net.SD.copy_(resid.std(1).clamp(min=1e-3))
        with torch.no_grad(): Ztr = net.z(Rtr)
        use_graph = DEV == "cuda" and os.environ.get("PINN_GRAPH", "1") == "1"
        opt = torch.optim.AdamW(net.parameters(), lr=torch.tensor(1e-3, device=DEV) if use_graph else 1e-3,
                                weight_decay=1e-4, capturable=use_graph)
        huber = lambda p, y: (nn.functional.smooth_l1_loss(p, y, beta=BETA, reduction="none") * cmask[:, None, :]).sum((1, 2)) / (cmask.sum(1) * p.shape[1])
        Yg_va = torch.gather(Yva_t.expand(E, -1, -1), 2, cols[:, None, :].expand(E, len(Rva), 2))
        best = torch.full((E,), float("inf"), device=DEV); best_ep = torch.zeros(E, device=DEV)
        best_state = [p.detach().clone() for p in net.parameters()]
        torch.manual_seed(0); n = len(Rtr); nb = n // BS        # full batches only (static shapes for the CUDA graph)
        dirs = sorted({c[0] for c in cons})
        p_drain = (lambda R: R[..., 3] > R[..., 4]) if self.polarity == "nmos" else (lambda R: R[..., 3] < R[..., 4])
        mvar = [v for v, c in enumerate(self.variants) if c.get("mono")]          # variants with the derivative penalty
        msel = torch.tensor([3 * v + k for v in mvar for k in range(3)], device=DEV); lam_mv = lam_m[mvar] if mvar else None
        static_idx = torch.zeros(E, BS, dtype=torch.long, device=DEV)

        def train_step():
            idx = static_idx
            out = net(Rtr[idx], Ztr[idx])
            loss = huber(out, torch.gather(Yg, 1, idx[..., None].expand(-1, -1, 2))).sum()
            if use_kcl:
                loss = loss + (lam_k * kcl_penalty(self._assemble(out, V), Str[idx[0::3]])).sum()
            if use_mono:
                Rc = lo + (hi - lo) * torch.rand(len(dirs) * N_COLL, 10, device=DEV)
                Rc[:, 3:5] = torch.rand(len(Rc), 2, device=DEV) * Rc[:, :1]          # vd, vs uniform in 0..vdd
                Tc = torch.cat([DIRS[d](Rc[k * N_COLL:(k + 1) * N_COLL]) for k, d in enumerate(dirs)])
                _, dO = jvp(lambda r: net(r, sel=msel), (Rc,), (Tc,))
                D4 = self._assemble(dO, len(mvar))                                  # [V_mono, Nc_tot, 4]
                pen = 0
                for d, cur, sgn, rows in cons:
                    k = dirs.index(d); sl = slice(k * N_COLL, (k + 1) * N_COLL); j = CUR.index(cur)
                    msk = torch.ones(N_COLL, device=DEV)
                    if rows == "drain":
                        dr = p_drain(Rc[sl]); msk = (dr if cur == "id" else ~dr).float()
                    pen = pen + ((torch.relu(-sgn * D4[:, sl, j]) ** 2) * msk).sum(1) / msk.sum().clamp(min=1)
                loss = loss + (lam_mv * pen).sum()
            loss.backward(); opt.step()

        new_perm = lambda: torch.argsort(torch.rand(V, n, device=DEV), 1).repeat_interleave(3, 0)  # same rows within a variant
        if use_graph:                                    # capture one whole training step (launch overhead dominates)
            static_idx.copy_(new_perm()[:, :BS])
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3): opt.zero_grad(set_to_none=True); train_step()
            torch.cuda.current_stream().wait_stream(st)
            graph = torch.cuda.CUDAGraph(); opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph): train_step()
        for ep in range(self.epochs):
            lr = 0.5e-3 * (1 + np.cos(np.pi * ep / self.epochs))                 # cosine annealing, as the reference
            if use_graph: opt.param_groups[0]["lr"].fill_(lr)
            else: opt.param_groups[0]["lr"] = lr
            perm = new_perm()
            for i in range(nb):
                static_idx.copy_(perm[:, i * BS:(i + 1) * BS])
                if use_graph: graph.replay()
                else: opt.zero_grad(); train_step()
            with torch.no_grad():
                vl = huber(net(Rva), Yg_va)
                better = vl < best; best = torch.where(better, vl, best); best_ep = torch.where(better, float(ep), best_ep)
                for p, s in zip(net.parameters(), best_state): s[better] = p.detach()[better]
            if self.verbose and (ep % 50 == 49 or ep == 0):
                print(f"   ep {ep+1} {time.time()-t0:.0f}s best val (decades, per variant, group id+is): "
                      + " ".join(f"{x:.4f}" for x in best.view(V, 3)[:, 0].tolist()), flush=True)
        with torch.no_grad():
            for p, s in zip(net.parameters(), best_state): p.copy_(s)
        self.net, self.V = net, V
        # ---- ablation on validation (magnitude error, harness sign rule not applied) + choice per group ----
        Lva = self.predict_all(dva)                      # [V, N, 4]
        err = 100 * np.abs(10 ** (Lva - Yva[None]) - 1)
        abl = {}
        for v, cfg in enumerate(self.variants):
            abl[cfg["name"]] = {c: (round(float(np.median(err[v, :, j])), 3), round(float(np.percentile(err[v, :, j], 95)), 2))
                                for j, c in enumerate(CUR)}
            abl[cfg["name"]]["val_loss"] = [round(x, 5) for x in best.view(V, 3)[v].tolist()]
        physics = [v for v, cfg in enumerate(self.variants) if not cfg["name"].startswith("data")]
        physics = physics or list(range(V))
        # choice per group: physics variant with the lowest validation MEDIAN relative error (the headline metric;
        # the Huber val loss at ~0.002-decade errors is ~MSE, dominated by a few outlier rows, and disagreed with it)
        med = np.median(err, 1)                                                     # [V, 4]
        gmed = np.stack([med[:, [0, 2]].mean(1), med[:, 1], med[:, 3]], 1)        # [V, 3 groups]
        self.choice = [physics[int(np.argmin(gmed[physics, gi]))] for gi in range(3)]
        self.ablation = abl
        self.info = {"chosen": {gname: self.variants[v]["name"] for gname, v in zip(["id+is", "ig", "ib"], self.choice)},
                     "constraints": [list(c) for c in cons], "epochs": self.epochs,
                     "fit_s": round(time.time() - t0, 1)}
        self.info["ablation_val_median_p95_pct"] = abl
        # constraint check on validation rows: % of rows with the expected derivative sign, data-only vs chosen (id+is net)
        dva_d = ((dva.vd > dva.vs) if self.polarity == "nmos" else (dva.vd < dva.vs)).to_numpy()
        chk = {}
        for name, v in (("data", 0), ("chosen", None)):
            Dall = {d: self.derivatives(dva, v if v is not None else self.choice[0])[d] for d in {c[0] for c in cons}} if cons else {}
            for d, cur, sgn, rows in cons:
                a = Dall[d][:, CUR.index(cur)]
                if rows == "drain": a = a[dva_d if cur == "id" else ~dva_d]
                chk.setdefault(f"{d}:{cur}:{rows}", {})[name] = round(100 * float((sgn * a > 0).mean()), 2)
        self.info["constraint_ok_pct_val"] = chk
        if any(self.variants[v].get("struct") for v in self.choice[:1]):
            v = self.choice[0] * 3
            self.info["ekv"] = {"n": round(float(net.n()[v]), 3), "eta": round(float(net.eta[v]), 4)}
        return self

    def predict_all(self, df):
        out = []
        with torch.no_grad():
            R = self.raw(df)
            for i in range(0, len(R), 4096): out.append(self._assemble(self.net(R[i:i + 4096]), self.V).cpu().numpy())
        return np.concatenate(out, 1).astype(float)

    def predict(self, X, df):
        L = self.predict_all(df); v0, v1, v2 = self.choice
        return np.column_stack([L[v0, :, 0], L[v1, :, 1], L[v0, :, 2], L[v2, :, 3]])

    def derivatives(self, df, v=0):
        """d log10|I| / d direction for variant v on the rows of df -> {dir: [N, 4]}."""
        R = self.raw(df); res = {}
        for d, f in DIRS.items():
            _, dO = jvp(lambda r: self.net(r), (R,), (f(R),))
            res[d] = self._assemble(dO.detach(), self.V)[v].cpu().numpy()
        return res
