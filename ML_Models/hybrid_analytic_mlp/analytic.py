"""Analytical (compact-model) part of hybrid_analytic_mlp: closed-form physics equations with a few fitted constants per case.

PMOS cases are mirrored onto NMOS form (every voltage measured from the body rail: v' = vb - v), so one set of
equations serves all four cases; current magnitudes are unchanged by the mirror. All voltages below are mirrored,
vhi / vlo = higher / lower of vd', vs' (effective drain / source), vds = vhi - vlo, W = Weff = w - dW (fitted, nm), Vt = kT/q, T in kelvin,
d_tox = toxe/toxe_nom - 1, d_nd = ndep/ndep_nom - 1.

Components (all >= 0, A):
  Ich   channel (EKV unified charge model, flows vhi -> vlo)
        Vth = |vth0| + kt1 (T/Tnom - 1) + k_tox d_tox + k_nd d_nd - sigma vds - sig2 vds^2 + gam (sqrt(0.9 + vlo) - sqrt(0.9))
        n = 1 + (n0 - 1)(1 + d_tox) sqrt(1 + d_nd)      (temperature, process, DIBL, body effect; n - 1 ~ Cdep/Cox)
        Vp = (vg' - Vth)/n,  F(x) = ln^2(1 + e^(x/2)),  qx = sqrt(F((Vp - vx)/Vt))
        Is = 2 n u0 (eps_ox/toxe) (W/L) Vt^2 (T/Tnom)^ute e^c_is
        Ich = Is [F_lo - F_hi] (1 + lam vds) / (1 + th_mob Vt (q_lo + q_hi) + th_vsat 2 Vt (q_lo - q_hi))
  Igx   gate-S/D overlap tunnelling at terminal x (BSIM4 igs/igd form), V = vg' - vx':
        Igx = W V^2 (tox_nom/toxe)^2 exp(o0 + o1|V| + d_tox (o2 + o3|V|) + o4 (T/Tnom - 1)),  direction gate -> x if V > 0
  Igc   gate-channel tunnelling (ON only), Vgs = vg' - vlo, Vaux = n Vt ln(1 + e^((Vgs - Vth)/(n Vt))):
        Igc = W Vgs Vaux exp(c0 + c1 Vgs + c2 d_tox), split to source / drain with the BSIM4 pigcd partition (p_gcd vds)
  Igb   gate-body tunnelling in inversion (ON only), Vauxb = nb Vt ln(1 + e^((Vgs - eb)/(nb Vt))):
        Igb = W Vgs Vauxb exp(b0 + b1 Vgs + b2 d_tox)
  Jx    reverse junction leakage at terminal x (bias vx'):
        Jx = W exp(j0) (T/Tnom)^3 exp(ea [Eg(Tnom)/Vt(Tnom) - Eg(T)/Vt]) (1 - e^(-vx'/Vt)) (1 + j1 vx')
  Gx    GIDL / GISL (OFF only), Ve = softplus(vx' - vg' - eg):  Gx = W exp(g0) Ve exp(-g1 (toxe/tox_nom) / Ve)
  I0    constant body floor exp(f0) (SPICE shows ~1.0e-14 A on the body at low T), split evenly over D and S

Terminal currents (mirrored-NMOS sign, positive into the terminal; KCL holds by construction):
  i_hi = Ich - Ig_hi + J_hi + G_hi + I0/2 - Igcd      (Ig_x signed: + when gate -> x)
  i_lo = -Ich - Ig_lo + J_lo + G_lo + I0/2 - Igcs
  ig   = Ig_hi + Ig_lo + Igcs + Igcd + Igb
  ib   = -(J_hi + J_lo + G_hi + G_lo + I0) - Igb
Fitting: scipy least_squares (soft_l1, f_scale 0.1 decade) on log10|I| residuals of all 4 currents jointly, after a
staged initialisation (gate -> body -> channel). Then each current is also refitted on its own (same equations, own
copy of the constants); per current, the variant with the lower VALIDATION mean |error| in decades is kept.
"""
import numpy as np
from scipy.optimize import least_squares

K_B, Q = 1.380649e-23, 1.602176634e-19
EPS_OX = 3.9 * 8.854e-12
L = 22e-9
TNOM = 300.15
NOM = {"nmos": dict(vth0=0.50308, toxe=1.05e-9, ndep=5.5e18, u0=0.04),
       "pmos": dict(vth0=0.4606, toxe=1.1e-9, ndep=4.4e18, u0=0.0095)}
FLOOR = 1e-18


def softplus(x):
    return np.logaddexp(0.0, x)


def eg(T):
    return 1.16 - 7.02e-4 * T ** 2 / (T + 1108.0)


def ex(x):
    return np.exp(np.clip(x, -700, 700))


class AnalyticModel:
    def __init__(self, case, polarity):
        self.case, self.polarity, self.info = case, polarity, {}
        self.on = case.endswith("_on")
        self.nom = NOM[polarity]
        # name: initial value
        p = dict(dw=10.0, sig2=0.0, gam=0.0, kt1=-0.11, k_tox=0.05, k_nd=0.05, sigma=0.05, n=1.4, ute=-1.5, c_is=np.log(0.3),
                 o0=-8.0, o1=2.0, o2=-5.0, o3=0.0, o4=0.0,
                 j0=np.log(1e-10), ea=1.0, j1=0.0, f0=np.log(1e-14))
        if self.on:
            p.update(lam=0.1, th_mob=1.0, th_vsat=2.0,
                     c0=-6.0, c1=2.0, c2=-5.0, p_gcd=1.0,
                     b0=-9.0, b1=2.0, b2=-5.0, eb=0.5, nb=3.0)
        else:
            p.update(g0=11.1, g1=6.6, eg=0.8)
        self.names = list(p)
        self.p0 = np.array([p[k] for k in self.names], float)

    # ------------------------------------------------------------------ physics
    def _vars(self, df):
        vdd = df["vdd"].to_numpy(float); vb = df["vb"].to_numpy(float)
        sgn = 1.0 if self.polarity == "nmos" else -1.0
        vd, vs, vg = (sgn * (df[k].to_numpy(float) - vb) for k in ("vd", "vs", "vg"))
        T = df["temp"].to_numpy(float) + 273.15
        V = dict(vdd=vdd, vd=vd, vs=vs, vg=vg, vhi=np.maximum(vd, vs), vlo=np.minimum(vd, vs), T=T,
                 Vt=K_B * T / Q, W0=df["w"].to_numpy(float), vth=np.abs(df["vth0"].to_numpy(float)),
                 toxe=df["toxe"].to_numpy(float), u0=df["u0"].to_numpy(float),
                 d_tox=df["toxe"].to_numpy(float) / self.nom["toxe"] - 1, d_nd=df["ndep"].to_numpy(float) / self.nom["ndep"] - 1)
        V["vds"] = V["vhi"] - V["vlo"]; V["d_hi"] = vd >= vs
        return V

    def currents(self, theta, V):
        P = dict(zip(self.names, theta))
        Vt, T, dtox = V["Vt"], V["T"], V["d_tox"]
        W = np.maximum(V["W0"] - abs(P["dw"]) * 1e-9, 1e-9)                                  # Weff = W - dW
        vg, vhi, vlo, vds = V["vg"], V["vhi"], V["vlo"], V["vds"]
        tr = T / TNOM - 1
        # channel (EKV)
        n = 1 + (abs(P["n"] - 1) + 1e-3) * (1 + dtox) * np.sqrt(1 + V["d_nd"])          # n - 1 ~ Cdep/Cox
        Vth = (V["vth"] + P["kt1"] * tr + P["k_tox"] * dtox + P["k_nd"] * V["d_nd"] - P["sigma"] * vds - P["sig2"] * vds ** 2
               + P["gam"] * (np.sqrt(0.9 + vlo) - np.sqrt(0.9)))
        Vp = (vg - Vth) / n
        q_lo, q_hi = softplus((Vp - vlo) / (2 * Vt)), softplus((Vp - vhi) / (2 * Vt))
        Is = 2 * n * V["u0"] * EPS_OX / V["toxe"] * (W / L) * Vt ** 2 * (T / TNOM) ** P["ute"] * ex(P["c_is"])
        num = Is * (q_lo ** 2 - q_hi ** 2)
        if self.on:
            num = num * (1 + P["lam"] * vds) / (1 + abs(P["th_mob"]) * Vt * (q_lo + q_hi) + abs(P["th_vsat"]) * 2 * Vt * (q_lo - q_hi))
        Ich = num

        # gate-S/D overlap tunnelling (signed: + gate -> terminal)
        def ov(Vx):
            a = np.abs(Vx)
            return np.sign(Vx) * W * Vx ** 2 / (1 + dtox) ** 2 * ex(P["o0"] + P["o1"] * a + dtox * (P["o2"] + P["o3"] * a) + P["o4"] * tr)
        Ig_hi, Ig_lo = ov(vg - vhi), ov(vg - vlo)

        # junctions + floor
        tfac = (T / TNOM) ** 3 * ex(P["ea"] * (eg(TNOM) / (K_B * TNOM / Q) - eg(T) / Vt))
        def jn(v):
            return W * ex(P["j0"]) * tfac * (1 - np.exp(-np.maximum(v, 0) / Vt)) * np.maximum(1 + P["j1"] * v, 1e-3)
        J_hi, J_lo = jn(vhi), jn(vlo)
        I0 = ex(P["f0"])
        i_hi = Ich - Ig_hi + J_hi + I0 / 2
        i_lo = -Ich - Ig_lo + J_lo + I0 / 2
        ig = Ig_hi + Ig_lo
        ib = -(J_hi + J_lo + I0)
        if self.on:
            Vgs = np.maximum(vg - vlo, 0)
            Vaux = n * Vt * softplus((Vgs - Vth) / (n * Vt))
            Igc = W * Vgs * Vaux * ex(P["c0"] + P["c1"] * Vgs + P["c2"] * dtox)
            pv = abs(P["p_gcd"]) * vds; den = pv ** 2 + 2e-4
            Igcs = Igc * (pv + np.exp(-pv) - 1 + 1e-4) / den
            Igcd = Igc * (1 - (pv + 1) * np.exp(-pv) + 1e-4) / den
            nb = abs(P["nb"]) + 0.1
            Vauxb = nb * Vt * softplus((Vgs - P["eb"]) / (nb * Vt))
            Igb = W * Vgs * Vauxb * ex(P["b0"] + P["b1"] * Vgs + P["b2"] * dtox)
            i_hi = i_hi - Igcd; i_lo = i_lo - Igcs
            ig = ig + Igcs + Igcd + Igb
            ib = ib - Igb
        else:
            def gidl(vx):
                Ve = 0.01 * softplus((vx - vg - P["eg"]) / 0.01)
                return W * ex(P["g0"]) * Ve * ex(-abs(P["g1"]) * (1 + dtox) / np.maximum(Ve, 1e-6))
            G_hi, G_lo = gidl(vhi), gidl(vlo)
            i_hi = i_hi + G_hi; i_lo = i_lo + G_lo; ib = ib - G_hi - G_lo
        dh = V["d_hi"]
        idd = np.where(dh, i_hi, i_lo); iss = np.where(dh, i_lo, i_hi)
        return np.column_stack([idd, ig, iss, ib])

    def logI(self, theta, V):
        return np.log10(np.maximum(np.abs(self.currents(theta, V)), FLOOR))

    # ------------------------------------------------------------------ fitting
    def _fit(self, theta, V, Y, free, outs, nfev=400):
        idx = [self.names.index(k) for k in free]
        def res(x):
            th = theta.copy(); th[idx] = x
            r = self.logI(th, V)[:, outs] - Y[:, outs]
            return np.nan_to_num(r, nan=10.0, posinf=10.0, neginf=-10.0).ravel()
        sol = least_squares(res, theta[idx], loss="soft_l1", f_scale=0.1, max_nfev=nfev, x_scale="jac")
        th = theta.copy(); th[idx] = sol.x
        return th

    def fit(self, dtr, Ytr, dva, Yva):
        """dtr/dva: raw training/validation rows, Ytr/Yva: their log10|I| (rows, 4)."""
        Vtr, Vva = self._vars(dtr), self._vars(dva)
        th = self.p0.copy()
        gate = ["dw", "o0", "o1", "o2", "o3", "o4"] + (["c0", "c1", "c2", "p_gcd", "b0", "b1", "b2", "eb", "nb"] if self.on else [])
        body = ["j0", "ea", "j1", "f0"] + ([] if self.on else ["g0", "g1", "eg"])
        chan = ["dw", "sig2", "gam", "kt1", "k_tox", "k_nd", "sigma", "n", "ute", "c_is"] + (["lam", "th_mob", "th_vsat"] if self.on else [])
        def shift(th, keys, j):                       # data-driven start: move prefactors to match the median decade
            off = np.log(10) * np.median(Ytr[:, j] - self.logI(th, Vtr)[:, j])
            th = th.copy()
            for k in keys: th[self.names.index(k)] += off
            return th
        if self.on:
            th = shift(th, ["c_is"], 0)
            th = self._fit(th, Vtr, Ytr, chan, [0, 2], 200)
        th = shift(th, ["o0"] + (["c0", "b0"] if self.on else []), 1)
        th = self._fit(th, Vtr, Ytr, gate, [1], 400)
        th = shift(th, ["j0"] if not self.on else ["j0", "f0"], 3)
        th = self._fit(th, Vtr, Ytr, body, [3], 400)
        th = self._fit(th, Vtr, Ytr, chan, [0, 2], 200)
        th = self._fit(th, Vtr, Ytr, self.names, [0, 1, 2, 3], 150)       # joint, KCL-consistent
        self.theta_joint = th
        mae = lambda t, j: float(np.mean(np.abs(self.logI(t, Vva)[:, j] - Yva[:, j])))
        self.theta = [th] * 4; variant = ["joint"] * 4; val = [mae(th, j) for j in range(4)]
        for j in range(4):                                                  # per-current refit, keep if better on val
            free = {1: gate, 3: body + (["b0", "b1", "b2", "eb", "nb"] if self.on else [])}.get(j, self.names)
            tj = self._fit(th, Vtr, Ytr, free, [j], 150)
            e = mae(tj, j)
            if e < val[j]: self.theta[j], variant[j], val[j] = tj, "separate", e
        cur = ["id", "ig", "is", "ib"]
        self.info = {"formulas": "EKV channel + BSIM4-style overlap/channel/body gate tunnelling + T^3 exp(-Eg/kT) "
                                 "junction + GIDL (OFF) + body floor; see hybrid_analytic_mlp/analytic.py docstring",
                     "n_params": len(self.names),
                     "variant": dict(zip(cur, variant)), "val_mae_dec": dict(zip(cur, [round(v, 4) for v in val])),
                     "params_joint": {k: round(float(v), 5) for k, v in zip(self.names, th)}}
        for j in range(4):
            if variant[j] == "separate":
                self.info[f"params_{cur[j]}"] = {k: round(float(v), 5) for k, v in zip(self.names, self.theta[j])}
        return self

    def predict(self, df):
        """log10|I| (rows, 4) for the raw rows df."""
        V = self._vars(df)
        return np.column_stack([self.logI(self.theta[j], V)[:, j] for j in range(4)])
