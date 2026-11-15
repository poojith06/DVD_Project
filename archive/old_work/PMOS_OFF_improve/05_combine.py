"""Combine: (1) average the chosen MLP networks, (2) per current blend with a tree model (weight picked on validation),
(3) signs from the repo's SignModel, (4) optional precision-weighted Kirchhoff correction, then report signed errors.
Usage: python 05_combine.py "<mlp glob>" <tree name> [--test]"""
import sys, glob, importlib
import numpy as np, pandas as pd
from common_setup import load, HERE, TARGET_COLUMNS, CURRENT_FLOOR
from common.surrogate import SignModel
from common.metrics import relative_error_percent
from sklearn.preprocessing import StandardScaler
ext_features = importlib.import_module("03_mlp_ensemble_feats").ext_features
mlp_glob, tree = sys.argv[1], sys.argv[2]; show_test = "--test" in sys.argv
X, Y, I, (tr, va, te), _ = load("remeasured")
sc = StandardScaler().fit(ext_features(tr)); Xe = {k: sc.transform(ext_features(d)) for k, d in (("tr", tr), ("va", va), ("te", te))}
files = sorted(glob.glob(str(HERE / "val_preds" / mlp_glob)))
mv = np.mean([np.load(f) for f in files], 0); mt = np.mean([np.load(f.replace("val_preds", "test_preds")) for f in files], 0)
tv, tt = np.load(HERE / "val_preds" / f"tree_{tree}_ext.npy"), np.load(HERE / "test_preds" / f"tree_{tree}_ext.npy")
signs = SignModel().fit(Xe["tr"], np.where(I["tr"] >= 0, 1.0, -1.0))
sv, st = signs.predict(Xe["va"]), signs.predict(Xe["te"])
magerr = lambda lp, It: 100 * np.abs(10 ** lp - np.abs(It)) / np.maximum(np.abs(It), CURRENT_FLOOR)
# blend weight per current, chosen on the validation median of the magnitude error
W = []
for j in range(4):
    cands = [(np.median(magerr(w * tv[:, j] + (1 - w) * mv[:, j], I["va"][:, j])), w) for w in np.linspace(0, 1, 21)]
    W.append(min(cands)[1])
W = np.array(W); bv = W * tv + (1 - W) * mv; bt = W * tt + (1 - W) * mt
r = np.array([np.median(magerr(bv[:, j], I["va"][:, j])) for j in range(4)]) / 100  # per-current precision estimate (validation)

def kcl(C, r):
    s2 = (r * np.abs(C)) ** 2; return C - s2 / s2.sum(1, keepdims=True) * C.sum(1, keepdims=True)

def report(name, C, It):
    e = relative_error_percent(It, C)
    return {"setup": name, **{f"{t}_median": np.median(e[:, j]) for j, t in enumerate(TARGET_COLUMNS)},
            **{f"{t}_p95": np.percentile(e[:, j], 95) for j, t in enumerate(TARGET_COLUMNS)}, **{f"{t}_mean": e[:, j].mean() for j, t in enumerate(TARGET_COLUMNS)}}
rows = []
for split_name, b, m, s, It in [("val", bv, mv, sv, I["va"])] + ([("TEST", bt, mt, st, I["te"])] if show_test else []):
    Cm = s * 10 ** m; Cb = s * 10 ** b
    rows += [report(f"{split_name}: MLP average only", Cm, It), report(f"{split_name}: + tree blend", Cb, It),
             report(f"{split_name}: + blend + equal-weight KCL", kcl(Cb, np.ones(4)), It),
             report(f"{split_name}: + blend + precision-weighted KCL", kcl(Cb, r), It)]
print(f"MLP networks averaged: {len(files)} ({mlp_glob}); tree: {tree}; blend weight on tree per current {dict(zip(TARGET_COLUMNS, W.round(2)))}")
print(f"per-current precision used for KCL weights: {dict(zip(TARGET_COLUMNS, (100*r).round(2)))} %")
df = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(df[["setup"] + [f"{t}_median" for t in TARGET_COLUMNS] + [f"{t}_p95" for t in TARGET_COLUMNS] + [f"{t}_mean" for t in TARGET_COLUMNS]].round(2).to_string(index=False))
