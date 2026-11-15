"""Final PMOS OFF model: one specialist per current, chosen on VALIDATION, then scored once on TEST.
  id, is : average of the two-output (id, is) MLPs, optionally mixed with the four-output MLP average
  ig     : average of the ig-only MLPs, optionally mixed with the four-output MLP average
  ib     : Extra Trees (extended features)
  signs  : repo SignModel (gradient-boosted classifier) on the extended features
  Kirchhoff: none / equal-weight / precision-weighted, picked on validation."""
import glob, importlib
import numpy as np, pandas as pd
from common_setup import load, HERE, TARGET_COLUMNS, CURRENT_FLOOR
from common.surrogate import SignModel
from common.metrics import relative_error_percent
from sklearn.preprocessing import StandardScaler
ext_features = importlib.import_module("03_mlp_ensemble_feats").ext_features
X, Y, I, (tr, va, te), _ = load("remeasured")
sc = StandardScaler().fit(ext_features(tr)); Xe = {k: sc.transform(ext_features(d)) for k, d in (("tr", tr), ("va", va), ("te", te))}
avg = lambda pat, split: np.mean([np.load(f.replace("val_preds", split)) for f in sorted(glob.glob(str(HERE / "val_preds" / pat)))], 0)
n = lambda pat: len(glob.glob(str(HERE / "val_preds" / pat)))
F4, F02, F1 = "ens_ext_256-128-64_e500_s?.npy", "ens_ext_256-128-64_e500_s?_out02.npy", "ens_ext_256-128-64_e500_s?_out1.npy"
parts = {s: dict(p4=avg(F4, s), p02=avg(F02, s), p1=avg(F1, s)[:, None] if avg(F1, s).ndim == 1 else avg(F1, s), et=np.load(HERE / s / "tree_et_ext.npy"))
         for s in ("val_preds", "test_preds")}
magerr = lambda lp, It: 100 * np.abs(10 ** lp - np.abs(It)) / np.maximum(np.abs(It), CURRENT_FLOOR)

def assemble(p, w02, w1):
    L = p["p4"].copy()
    L[:, [0, 2]] = w02 * p["p02"] + (1 - w02) * p["p4"][:, [0, 2]]
    L[:, 1] = w1 * p["p1"][:, 0] + (1 - w1) * p["p4"][:, 1]
    L[:, 3] = p["et"][:, 3]
    return L

# mixing weights chosen on validation (median magnitude error, summed over the currents they affect)
grid = np.linspace(0, 1, 11)
w02 = min(grid, key=lambda w: sum(np.median(magerr(assemble(parts["val_preds"], w, 1)[:, j], I["va"][:, j])) for j in (0, 2)))
w1 = min(grid, key=lambda w: np.median(magerr(assemble(parts["val_preds"], w02, w)[:, 1], I["va"][:, 1])))
signs = SignModel().fit(Xe["tr"], np.where(I["tr"] >= 0, 1.0, -1.0))
Lv, Lt = assemble(parts["val_preds"], w02, w1), assemble(parts["test_preds"], w02, w1)
r = np.array([np.median(magerr(Lv[:, j], I["va"][:, j])) for j in range(4)]) / 100

def kcl(C, rr):
    s2 = (rr * np.abs(C)) ** 2; return C - s2 / s2.sum(1, keepdims=True) * C.sum(1, keepdims=True)
VARIANTS = {"no Kirchhoff correction": lambda C: C, "equal-weight Kirchhoff": lambda C: kcl(C, np.ones(4)), "precision-weighted Kirchhoff": lambda C: kcl(C, r)}

def scores(C, It):
    e = relative_error_percent(It, C)
    return {**{f"{t} median": np.median(e[:, j]) for j, t in enumerate(TARGET_COLUMNS)}, **{f"{t} p95": np.percentile(e[:, j], 95) for j, t in enumerate(TARGET_COLUMNS)},
            **{f"{t} mean": e[:, j].mean() for j, t in enumerate(TARGET_COLUMNS)}}
Cv = signs.predict(Xe["va"]) * 10 ** Lv
val = {k: scores(f(Cv), I["va"]) for k, f in VARIANTS.items()}
best = min(val, key=lambda k: sum(val[k][f"{t} median"] for t in ("id", "is", "ib")))   # Kirchhoff option picked on validation
Ct = VARIANTS[best](signs.predict(Xe["te"]) * 10 ** Lt)
rows = [{"split": "validation", "setup": k, **v} for k, v in val.items()] + [{"split": "TEST (final, used once)", "setup": best, **scores(Ct, I["te"])}]
print(f"networks: {n(F02)} x (id,is), {n(F1)} x ig-only, {n(F4)} x four-output; mix weight on specialists: id/is {w02:.1f}, ig {w1:.1f}; ib = Extra Trees")
print(f"Kirchhoff option chosen on validation: {best}")
df = pd.DataFrame(rows); df.to_csv(HERE / "final_results.csv", index=False); pd.set_option("display.width", 260)
for stat in ("median", "p95", "mean"):
    print(f"\n{stat} relative error (%):"); print(df[["split", "setup"] + [f"{t} {stat}" for t in TARGET_COLUMNS]].round(2).to_string(index=False))
