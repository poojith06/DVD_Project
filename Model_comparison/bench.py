"""Benchmark harness: every regression model gets exactly the same data, features, target, signs and metrics.

Data    : the same files and split as Final_models (Datasets/data, 70/15/15, seed 42)
Inputs  : X = the repo's build_features (20 columns), standardised with the Final_models scaler; the raw DataFrame
          is also passed for models that need physical inputs (analytical model, PINN)
Target  : log10|I| for id, ig, is, ib  -> every model predicts a (rows, 4) array of decades
Signs   : identical for every model - Final_models' predict_sign (ON: physics rule; OFF: effective-drain rule +
          source classifier, fed with the model's own predicted magnitudes)
Tuning  : on the validation split only; test is never used for any choice
Metrics : relative error |pred - SPICE| / |SPICE| (repo's relative_error_percent): MAE %, median %, p95 %;
          R2 on log10|I| and on amps; sign accuracy
Usage   : python bench.py <model> [case ...]          e.g. python bench.py random_forest
          models are listed in models.py (REGISTRY); results -> results/<model>/<case>_metrics.csv
Model interface: m = REGISTRY[name](case, polarity); m.fit(data); m.predict(X, df) -> (rows, 4) log10|I|
          data = {"train": (X, Y, df), "validation": (X, Y, df)}
"""
import sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "Final_models")); sys.path.insert(0, str(REPO / "ML models"))
import train_final as tf
from common.config import CASES
from common.features import build_features
from common.surrogate import log_magnitude
from common.metrics import relative_error_percent
from sklearn.metrics import r2_score

CURRENTS = tf.CURRENTS
ALL_CASES = list(CASES)

def load(case):
    """splits (DataFrames), X, Y per split and the Final_models bundle (scaler + sign models)."""
    S = tf.splits(case); bundle = tf.load(case); pol = CASES[case][0]
    X = {k: bundle["scaler"].transform(build_features(d, pol)) for k, d in S.items()}
    Y = {k: log_magnitude(d[CURRENTS].to_numpy()) for k, d in S.items()}
    return S, X, Y, bundle

def score(case, S, bundle, logpred, model_name, signed=None):
    """signed: optional {split: signed currents} for models that predict signs themselves (component model)."""
    rows = []
    for k, d in S.items():
        true = d[CURRENTS].to_numpy()
        pred = signed[k] if signed is not None else tf.predict_sign(bundle, d, logpred[k]) * 10 ** logpred[k]
        err = relative_error_percent(true, pred); lt, lp = log_magnitude(true), log_magnitude(pred)
        for j, c in enumerate(CURRENTS):
            rows.append({"model": model_name, "case": case, "split": k, "current": c,
                         "MAE %": err[:, j].mean(), "median %": np.median(err[:, j]), "p95 %": np.percentile(err[:, j], 95),
                         "R2 (log)": r2_score(lt[:, j], lp[:, j]), "R2 (I)": r2_score(true[:, j], pred[:, j]),
                         "sign acc %": 100 * (np.sign(pred[:, j]) == np.where(true[:, j] >= 0, 1, -1)).mean()})
    return pd.DataFrame(rows)

def run(model_name, cases):
    from models import REGISTRY
    out = HERE / "results" / model_name; out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        t0 = time.time(); S, X, Y, bundle = load(case)
        m = REGISTRY[model_name](case, CASES[case][0])
        m.fit({k: (X[k], Y[k], S[k]) for k in ("train", "validation")})
        t_fit = time.time() - t0
        logpred = {k: np.asarray(m.predict(X[k], S[k]), float) for k in S}
        signed = {k: np.asarray(m.predict_signed(X[k], S[k]), float) for k in S} if hasattr(m, "predict_signed") else None
        R = score(case, S, bundle, logpred, model_name, signed); R["fit s"] = round(t_fit, 1)
        R.to_csv(out / f"{case}_metrics.csv", index=False)
        np.savez_compressed(out / f"{case}_logpred.npz", **logpred)
        info = getattr(m, "info", {})
        (out / f"{case}_info.json").write_text(json.dumps(info, indent=1, default=str))
        v = R[R.split == "validation"]; t = R[R.split == "test"]
        print(f"[{model_name}] {case:8s} fit {t_fit:6.0f}s | val median {' '.join(f'{x:.2f}' for x in v['median %'])}"
              f" | test median {' '.join(f'{x:.2f}' for x in t['median %'])} | test p95 {' '.join(f'{x:.1f}' for x in t['p95 %'])}"
              f" | {info}", flush=True)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:] or ALL_CASES)
