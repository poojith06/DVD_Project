"""NMOS ON - polynomial ridge regression, degree 1 to 5, for two targets: the current I and log10|I|.

Data   : Datasets/data/nmos_on.csv (10,000 rows; fixed netlist, all 5 process parameters vary)
Split  : 70/15/15 (7,000 / 1,500 / 1,500), seed 42 - same split as the repo's ML models/ code
Inputs : the 10 columns that vary in NMOS ON: vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj (vg = vdd, vb = 0 are fixed)
Model  : one ridge per current (id, ig, is, ib): standardise -> all polynomial terms up to degree d -> standardise -> ridge.
         Ridge penalty chosen per current by leave-one-out CV on the training rows only (RidgeCV).
Targets: "I"        = signed current in amps (standardised per current before fitting)
         "log10|I|" = log of the magnitude; sign restored by the physics rule ig > 0, ib < 0,
                      sign(id) = sign(vd - vs), sign(is) = -sign(vd - vs)  (true for 99.99 % of rows)
Metrics: relative error = |pred - SPICE| / |SPICE|;  MAE % = mean, median %, p95 %;  R2 (I) on amps, R2 (log) on log10|I|
Output : results/nmos_on_ridge_all_metrics.csv, results/nmos_on_ridge_alphas.csv, results/nmos_on_ridge_best.csv, printed tables
Run    : python nmos_on_ridge.py   (from this folder)
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "ML models"))
from common.data import load_case, split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

INPUTS = ["vdd", "temp", "w", "vd", "vs", "vth0", "toxe", "ndep", "u0", "xj"]
CURRENTS = ["id", "ig", "is", "ib"]
ALPHAS = np.logspace(-10, 3, 27)
METRICS = ["MAE %", "median %", "p95 %", "R2 (I)", "R2 (log)"]

df = load_case("nmos_on", REPO / "Datasets" / "data")
tr, va, te = split(df, (0.70, 0.15, 0.15), 42)
SPLITS = {"train": tr, "validation": va, "test": te}

def sign_rule(d):
    s = np.sign(d.vd - d.vs).replace(0, 1).to_numpy()
    return np.column_stack([s, np.ones(len(d)), -s, -np.ones(len(d))])

def metrics(true, pred):
    out = {}
    for j, c in enumerate(CURRENTS):
        t, p = true[:, j], pred[:, j]; e = 100 * np.abs(p - t) / np.abs(t)
        out[c] = {"MAE %": e.mean(), "median %": np.median(e), "p95 %": np.percentile(e, 95), "R2 (I)": r2_score(t, p),
                  "R2 (log)": r2_score(np.log10(np.abs(t)), np.log10(np.maximum(np.abs(p), 1e-30)))}
    return out

def fit_predict(degree, target):
    Y = tr[CURRENTS].to_numpy()
    if target == "I": mu, sd = Y.mean(0), Y.std(0); y = (Y - mu) / sd
    else: y = np.log10(np.abs(Y))
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False), StandardScaler(),
                          RidgeCV(alphas=ALPHAS, alpha_per_target=True)).fit(tr[INPUTS].to_numpy(), y)
    preds = {}
    for name, d in SPLITS.items():
        p = model.predict(d[INPUTS].to_numpy())
        preds[name] = p * sd + mu if target == "I" else sign_rule(d) * 10 ** p
    return preds, dict(zip(CURRENTS, np.atleast_1d(model[-1].alpha_))), model[2].n_features_in_

rows, alpha_rows = [], []
for target in ["I", "log10|I|"]:
    for degree in range(1, 6):
        t0 = time.time()
        preds, alphas, n_terms = fit_predict(degree, "I" if target == "I" else "log")
        for split_name, d in SPLITS.items():
            for cur, m in metrics(d[CURRENTS].to_numpy(), preds[split_name]).items():
                rows.append({"target": target, "degree": degree, "split": split_name, "current": cur, **m})
        alpha_rows.append({"target": target, "degree": degree, "poly terms": n_terms, "fit s": round(time.time() - t0, 1), **{f"alpha {c}": a for c, a in alphas.items()}})
        print(f"fitted {target:9s} degree {degree}: {n_terms:5d} polynomial terms, {time.time()-t0:5.1f} s", flush=True)
R, A = pd.DataFrame(rows), pd.DataFrame(alpha_rows)
(HERE / "results").mkdir(exist_ok=True)
R.to_csv(HERE / "results" / "nmos_on_ridge_all_metrics.csv", index=False); A.to_csv(HERE / "results" / "nmos_on_ridge_alphas.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("\n=== chosen ridge penalty (alpha) ===")
print(A.to_string(index=False, formatters={c: "{:.1e}".format for c in A.columns if c.startswith("alpha")}))
for split_name in ["train", "validation", "test"]:
    print(f"\n==================== {split_name.upper()} ====================")
    for metric in METRICS:
        t = R[R.split == split_name].pivot_table(index=["target", "degree"], columns="current", values=metric, sort=False)[CURRENTS]
        print(f"\n--- {metric} ({split_name}) ---"); print(t.to_string(float_format=(lambda v: f"{v:.4f}") if "R2" in metric else (lambda v: f"{v:,.2f}")))

best = []
for target in ["I", "log10|I|"]:
    for cur in CURRENTS:
        v = R[(R.target == target) & (R.current == cur) & (R.split == "validation")]
        deg = int(v.loc[v["median %"].idxmin(), "degree"])
        t = R[(R.target == target) & (R.current == cur) & (R.split == "test") & (R.degree == deg)].iloc[0]
        best.append({"target": target, "current": cur, "best degree": deg, **{f"test {m}": t[m] for m in METRICS}})
B = pd.DataFrame(best); B.to_csv(HERE / "results" / "nmos_on_ridge_best.csv", index=False)
print("\n=== best degree per target and current (lowest validation median error) - TEST results ===")
print(B.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
