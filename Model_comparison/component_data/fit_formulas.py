"""Find and verify the terminal-current formulas in terms of the BSIM4 instance currents.

For each case, mode (sign of vd-vs) and terminal: least squares of the replayed terminal current (-i(v*)) on the 11
components, relative weighting, restricted to rows where the terminal is well resolved (|I| > 1e-13 A, far above the
~2.2e-16 A branch-current quantisation); coefficients rounded to integers and the rounded formula re-checked.
Also compares the replayed terminal currents with the dataset CSV.
Usage: python fit_formulas.py
"""
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
COMP = ["c_id", "c_ibd", "c_ibs", "c_isub", "c_igidl", "c_igisl", "c_igs", "c_igd", "c_igb", "c_igcs", "c_igcd"]
TERMS = ["id", "ig", "is", "ib"]

for case in ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]:
    C = pd.read_csv(HERE / f"{case}_components.csv"); D = pd.read_csv(REPO / "Datasets" / "data" / f"{case}.csv")
    print(f"===== {case}")
    for t in TERMS:   # replay vs CSV
        a, b = C[f"t_{t}"].to_numpy(), D[t].to_numpy()
        big = np.abs(b) > 1e-14
        r = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
        print(f"  replay vs CSV {t}: |I|>1e-14 rows {big.sum()}: median rel {np.median(r[big]):.1e} max {r[big].max():.1e};"
              f" all rows max abs diff {np.abs(a-b).max():.1e}")
    mode = np.sign(D.vd - D.vs).to_numpy()
    for m in (1, -1):
        sel = mode == m
        for t in TERMS:
            y = C.loc[sel, f"t_{t}"].to_numpy(); A = C.loc[sel, COMP].to_numpy()
            ok = np.abs(y) > 1e-13
            if ok.sum() < 20: print(f"  mode {m:+d} {t}: too few resolved rows"); continue
            w = 1 / np.abs(y[ok]); coef = np.linalg.lstsq(A[ok] * w[:, None], y[ok] * w, rcond=None)[0]
            k = np.round(coef)
            rel = np.abs(A[ok] @ k - y[ok]) / np.abs(y[ok])
            share = (np.abs(A[ok]) / np.abs(y[ok])[:, None]).max(0)   # largest share of the terminal per component
            terms = " ".join(f"{int(v):+d}*{c[2:]}" for v, c in zip(k, COMP) if v != 0)
            print(f"  mode {m:+d} {t}: {terms} | raw coef {np.array2string(coef, precision=2, suppress_small=True)}"
                  f" | rel resid median {np.median(rel):.1e} max {rel.max():.1e} (n={ok.sum()})")
            print(f"        max share per comp: " + " ".join(f"{c[2:]}={s:.0e}" for c, s in zip(COMP, share)))

# ---- unified formula (model_components.coefficients) vs the CSV and vs the replayed -i(v), all 10,000 rows ----
sys_path = str(HERE.parent)
import sys; sys.path.insert(0, sys_path)
from model_components import coefficients, COMP as MCOMP
print("\n===== unified formula check (rel = |formula - CSV| / |CSV|; CSV has 6 significant digits)")
for case in ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]:
    C = pd.read_csv(HERE / f"{case}_components.csv"); D = pd.read_csv(REPO / "Datasets" / "data" / f"{case}.csv")
    R = np.einsum("rtc,rc->rt", coefficients(D, "nmos" if case[0] == "n" else "pmos"), C[[f"c_{c}" for c in MCOMP]].to_numpy())
    for j, t in enumerate(TERMS):
        y = D[t].to_numpy(); rel = np.abs(R[:, j] - y) / np.maximum(np.abs(y), 1e-300); big = np.abs(y) > 1e-15
        rr = C[f"t_{t}"].to_numpy()
        print(f"  {case:8s} {t}: vs CSV (|I|>1e-15, n={big.sum()}) median {np.median(rel[big]):.1e} p99 {np.percentile(rel[big], 99):.1e}"
              f" max {rel[big].max():.1e}; max |diff| all rows {np.abs(R[:, j]-y).max():.1e} A;"
              f" vs replay -i(v) max |diff| {np.abs(R[:, j]-rr).max():.1e} A; KCL max |sum| {np.abs(R.sum(1)).max():.1e} A")
