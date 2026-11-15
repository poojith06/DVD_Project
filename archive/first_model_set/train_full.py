"""Retrain the chosen model of each case on ALL 10,000 rows of its dataset, for use in Part B and Part C.

Part A (model choice on validation, errors on test) comes from the 70/15/15 models in <model>/saved_models.
After that choice, the same model type with the same settings is retrained on every row:
  - neural networks: same recipe and 500 epochs; with no validation set left, the epoch is picked on the
    training loss (the cosine learning rate ends near 0, so this is in practice the last epoch)
  - GPR: the kernel each current chose on validation in the 70 % run is kept; hyper-parameters refitted
    (log marginal likelihood on 2,000 rows), then conditioned on all 10,000 rows
  - feature scaler and sign model refitted on all rows
The gate and circuit data (Part B, Part C) are never used.

    python train_full.py                                     # GPR for ON, resnet_mlp for OFF
    python train_full.py --choose nmos_off=hybrid_analytic_mlp --cases nmos_off

Writes full_data/saved_models/<case>.joblib and full_data/fit_check.csv. Then:
    python evaluate_gates.py --models-dir full_data/saved_models --out full_data/gate_results
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common.config import CASES, DATA_DIR, MODEL_SEED, ROOT, SEED, SPLIT, TARGET_COLUMNS
from common.data import load_case, split
from common.metrics import log_magnitude, summarise
from common.registry import MODEL_TYPES
from common.signs import SignModel
from common.surrogate import TransistorSurrogate, fit_scaler
from gpr.model import ExactGP, GPRModel

DEFAULT_CHOICE = {"nmos_on": "gpr", "pmos_on": "gpr", "nmos_off": "resnet_mlp", "pmos_off": "resnet_mlp"}
OUT = ROOT / "full_data"
EPOCHS = None


def fit_full(case, model_key, df, X, Y):
    if model_key == "gpr":
        # keep the kernel chosen on validation in the 70 % run
        kernels = TransistorSurrogate.load(ROOT / "gpr" / "saved_models" / f"{case}.joblib").estimator.kernels
        est = GPRModel(case, seed=MODEL_SEED)
        est.kernels = list(kernels)
        est.gps = [ExactGP(k, MODEL_SEED).fit(X, Y[:, j]) for j, k in enumerate(kernels)]
        print(f"  GPR kernels (from the 70 % run): {kernels}", flush=True)
        return est
    est = MODEL_TYPES[model_key](case, seed=MODEL_SEED, params={"epochs": EPOCHS} if EPOCHS else None)
    # no held-out rows: the "validation" argument is the training data itself
    return est.fit(X, Y, X, Y, df, df)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--choose", nargs="+", default=[], metavar="CASE=MODEL")
    parser.add_argument("--epochs", type=int, help="override the network epochs (quick test runs only)")
    parser.add_argument("--out", type=Path, default=ROOT / "full_data")
    args = parser.parse_args()
    global EPOCHS, OUT
    EPOCHS, OUT = args.epochs, args.out
    choice = dict(DEFAULT_CHOICE)
    for item in args.choose:
        case, _, model = item.partition("=")
        choice[case] = model
    (OUT / "saved_models").mkdir(parents=True, exist_ok=True)

    rows = []
    for case in args.cases:
        model_key = choice[case]
        df = load_case(case, DATA_DIR)
        print(f"\n=== {case}: {model_key} on all {len(df)} rows", flush=True)
        surrogate = TransistorSurrogate(case, fit_scaler(df, case), None)
        X = surrogate.transform(df)
        Y = log_magnitude(df[TARGET_COLUMNS].to_numpy())
        start = time.perf_counter()
        est = fit_full(case, model_key, df, X, Y)
        if not hasattr(est, "predict_signed"):
            surrogate.sign_model = SignModel(case).fit(X, df, df[TARGET_COLUMNS].to_numpy(), est.predict(X, df))
        surrogate.estimator = est
        seconds = time.perf_counter() - start
        surrogate.save(OUT / "saved_models" / f"{case}.joblib")

        # Check: error on the rows it was trained on, next to the 70 % model's error on its own training rows.
        # (All rows are seen in training now, so this is a fit check, not a test result.)
        old = TransistorSurrogate.load(ROOT / model_key / "saved_models" / f"{case}.joblib")
        train70 = split(df, SPLIT, SEED)[0]
        new_all = summarise(df[TARGET_COLUMNS].to_numpy(), surrogate.predict_array(df))
        old_tr = summarise(train70[TARGET_COLUMNS].to_numpy(), old.predict_array(train70))
        for label, m in (("100 % model, all rows", new_all), ("70 % model, its train rows", old_tr)):
            print(f"  {label:28s} median % id/ig/is/ib: "
                  + " / ".join(f"{m[f'{c}_median_error_pct']:.3f}" for c in TARGET_COLUMNS)
                  + "   p95 %: " + " / ".join(f"{m[f'{c}_p95_error_pct']:.2f}" for c in TARGET_COLUMNS), flush=True)
            rows.append({"case": case, "model": model_key, "check": label, **m})
        print(f"  trained in {seconds:.0f}s -> {OUT / 'saved_models' / f'{case}.joblib'}", flush=True)

    path = OUT / "fit_check.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["case"].isin(new["case"])], new], ignore_index=True)
    new.to_csv(path, index=False)


if __name__ == "__main__":
    main()
