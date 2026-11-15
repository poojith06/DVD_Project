"""NMOS ON - MLP ensembles trained on the GPU, aiming for < 1 % error on every terminal current.

Data   : Datasets/data/nmos_on.csv, split 70/15/15 seed 42 (same as the repo and NMOS_ON_ridge/)
Inputs : repo physics features (ML models/common/features.py) + per-terminal extras: vgd, vgs, vgd/Vt, vgs/Vt, min(vd,vs), (vd-vs)/Vt, sign(vd-vs)
Target : log10|I| per current; sign restored by the physics rule ig > 0, ib < 0, sign(id) = sign(vd-vs), sign(is) = -sign(vd-vs)
Model  : E MLPs (256-128-64, SiLU) per output group, trained together on the GPU (gpu_ensemble.py), then averaged.
         Groups: (id, is) together, ig alone, ib alone. Huber loss in decades, AdamW, cosine LR, each network keeps its best validation epoch.
Metrics: relative error |pred - SPICE| / |SPICE|: MAE % (mean), median %, p95 %; R2 on log10|I| and on the current in amps.
Output : results/metrics.csv (all splits), results/ensemble_size.csv, preds/*.npy
Usage  : python train_mlp_ensemble.py [E] [epochs] [target] [features]      (defaults 16, 500, log, ext)
         features = raw  : only the 10 varying CSV columns (vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj)
         features = repo : the repo's build_features (raw + physics features)
         features = ext  : repo + vgd, vgs, vgd/Vt, vgs/Vt, min(vd,vs), (vd-vs)/Vt, sign(vd-vs)  (default)
         target = log : log10|I| with the physics sign rule (default)
         target = I   : the signed current in amps, standardised per current (for comparison)
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "ML models"))
from common.data import load_case, split
from common.features import build_features, BOLTZMANN_OVER_Q
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from gpu_ensemble import train_ensemble, DEVICE

E = int(sys.argv[1]) if len(sys.argv) > 1 else 16
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
TARGET = sys.argv[3] if len(sys.argv) > 3 else "log"
FEATS = sys.argv[4] if len(sys.argv) > 4 else "ext"
RAW = ["vdd", "temp", "w", "vd", "vs", "vth0", "toxe", "ndep", "u0", "xj"]
CURRENTS = ["id", "ig", "is", "ib"]
GROUPS = {"id+is": [0, 2], "ig": [1], "ib": [3]}

def features(d):
    if FEATS == "raw": return d[RAW]
    if FEATS == "repo": return build_features(d, "nmos")
    f = build_features(d, "nmos").copy(); vt = BOLTZMANN_OVER_Q * (d["temp"] + 273.15)
    f["vgd"] = d.vg - d.vd; f["vgs"] = d.vg - d.vs; f["vgd_vt"] = f.vgd / vt; f["vgs_vt"] = f.vgs / vt
    f["vmin"] = np.minimum(d.vd, d.vs); f["vds_vt"] = (d.vd - d.vs) / vt; f["sign_vds"] = np.sign(d.vd - d.vs)
    return f

def sign_rule(d):
    s = np.sign(d.vd - d.vs).replace(0, 1).to_numpy()
    return np.column_stack([s, np.ones(len(d)), -s, -np.ones(len(d))])

def metrics(true, pred):
    out = {}
    for j, c in enumerate(CURRENTS):
        t, p = true[:, j], pred[:, j]; e = 100 * np.abs(p - t) / np.abs(t)
        out[c] = {"MAE %": e.mean(), "median %": np.median(e), "p95 %": np.percentile(e, 95),
                  "R2 (log)": r2_score(np.log10(np.abs(t)), np.log10(np.maximum(np.abs(p), 1e-30))), "R2 (I)": r2_score(t, p)}
    return out

df = load_case("nmos_on", REPO / "Datasets" / "data")
tr, va, te = split(df, (0.70, 0.15, 0.15), 42)
SPLITS = {"train": tr, "validation": va, "test": te}
sc = StandardScaler().fit(features(tr)); X = {k: sc.transform(features(d)) for k, d in SPLITS.items()}
if TARGET == "log":
    Y = {k: np.log10(np.abs(d[CURRENTS].to_numpy())) for k, d in SPLITS.items()}
else:   # signed current, standardised with the training mean/std of each current
    MU, SD = tr[CURRENTS].to_numpy().mean(0), tr[CURRENTS].to_numpy().std(0)
    Y = {k: (d[CURRENTS].to_numpy() - MU) / SD for k, d in SPLITS.items()}
to_amps = (lambda d, p: sign_rule(d) * 10 ** p) if TARGET == "log" else (lambda d, p: p * SD + MU)
print(f"device: {DEVICE} | target: {TARGET} | features: {FEATS} | {E} networks per group | {EPOCHS} epochs | {X['train'].shape[1]} input features")

P = {k: np.zeros((E, len(d), 4)) for k, d in SPLITS.items()}     # log10|I| predictions of every network
for g, cols in GROUPS.items():
    t0 = time.time()
    pv, pt, px = train_ensemble(X["train"], Y["train"][:, cols], X["validation"], Y["validation"][:, cols], X["test"],
                                E=E, epochs=EPOCHS, seed=0, verbose=False, extra={"train": X["train"]})
    P["validation"][:, :, cols], P["test"][:, :, cols], P["train"][:, :, cols] = pv, pt, px["train"]
    print(f"  trained group {g:6s} ({E} networks) in {time.time()-t0:.0f}s", flush=True)
(HERE / "preds").mkdir(exist_ok=True); (HERE / "results").mkdir(exist_ok=True)
TAG = ("" if TARGET == "log" else "_targetI") + ("" if FEATS == "ext" else f"_feat{FEATS}")
for k in SPLITS: np.save(HERE / "preds" / f"{k}_{'log10I' if TARGET == 'log' else 'I_std'}{'' if FEATS == 'ext' else '_feat' + FEATS}_all_networks.npy", P[k])

rows = []
for k, d in SPLITS.items():
    pred = to_amps(d, P[k].mean(0))
    for c, m in metrics(d[CURRENTS].to_numpy(), pred).items(): rows.append({"split": k, "current": c, **m})
R = pd.DataFrame(rows); R.to_csv(HERE / "results" / f"metrics{TAG}.csv", index=False)
size_rows = []
for n in (1, 2, 4, 8, E):
    pred = to_amps(va, P["validation"][:n].mean(0))
    for c, m in metrics(va[CURRENTS].to_numpy(), pred).items(): size_rows.append({"networks averaged": n, "current": c, "median %": m["median %"], "p95 %": m["p95 %"], "MAE %": m["MAE %"]})
S = pd.DataFrame(size_rows); S.to_csv(HERE / "results" / f"ensemble_size{TAG}.csv", index=False)

pd.set_option("display.width", 200)
fmt = {"MAE %": "{:.3f}".format, "median %": "{:.3f}".format, "p95 %": "{:.2f}".format, "R2 (log)": "{:.5f}".format, "R2 (I)": "{:.5f}".format}
for k in SPLITS:
    print(f"\n=== {k.upper()} (ensemble of {E}) ==="); print(R[R.split == k].drop(columns="split").to_string(index=False, formatters=fmt))
print("\n=== validation median % vs number of networks averaged ===")
print(S.pivot(index="networks averaged", columns="current", values="median %")[CURRENTS].round(3).to_string())
