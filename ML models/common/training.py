"""Train one model type on the four cases and report its validation and test results.

Used by each model folder's train.py. Models are trained one at a time; choosing the
best model for each case afterwards is done by select_best.py.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import (CASES, CURRENT_FLOOR, DATA_DIR, SEED, SPLIT, TARGET_COLUMNS)
from common.data import kirchhoff_residual, load_case, split
from common.metrics import summarise
from common.registry import MODEL_TYPES
from common.surrogate import TransistorSurrogate, fit_shared_parts, log_magnitude


def parse_args(output_dir):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path(output_dir) / "results")
    parser.add_argument("--models-dir", type=Path, default=Path(output_dir) / "saved_models")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--split", nargs=3, type=float, default=SPLIT,
                        metavar=("TRAIN", "VAL", "TEST"), help="e.g. 0.8 0.1 0.1")
    parser.add_argument("--mlp-epochs", type=int, help="override the MLP epoch count")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def parity_plot(true, pred, case, model_name, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for k, (ax, name) in enumerate(zip(axes, TARGET_COLUMNS)):
        x = np.maximum(np.abs(true[:, k]), CURRENT_FLOOR)
        y = np.maximum(np.abs(pred[:, k]), CURRENT_FLOOR)
        ax.loglog(x, y, ".", markersize=2, alpha=0.5)
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.loglog(limits, limits, "k--", linewidth=1)
        ax.set_title(f"{name}")
        ax.set_xlabel("SPICE |I| (A)")
        ax.set_ylabel("Predicted |I| (A)")
    fig.suptitle(f"{case}: {model_name}, test set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_case(model_key, case, args):
    """Train one model on one case, save it, and return (validation row, test row)."""
    df = load_case(case, args.data_dir)
    residual = kirchhoff_residual(df)
    print(f"\n=== {case}: {len(df)} rows | dataset KCL residual / largest current: "
          f"median {residual.median():.1e}, max {residual.max():.1e}")

    train, val, test = split(df, args.split, args.seed)
    scaler, sign_model, X_train, signs_train = fit_shared_parts(train, case)
    y_train = log_magnitude(train[TARGET_COLUMNS].to_numpy())
    y_val = log_magnitude(val[TARGET_COLUMNS].to_numpy())

    params = {"epochs": args.mlp_epochs} if model_key == "mlp" and args.mlp_epochs else None
    estimator = MODEL_TYPES[model_key](params, seed=args.seed)
    surrogate = TransistorSurrogate(case, scaler, estimator, sign_model)
    X_val = surrogate.transform(val)

    start = time.perf_counter()
    estimator.fit(X_train, y_train, X_val, y_val, signs_train)
    seconds = time.perf_counter() - start
    surrogate.calibrate_smear(X_val, y_val)

    val_scores = summarise(val[TARGET_COLUMNS].to_numpy(), surrogate.predict_scaled(X_val))
    X_test = surrogate.transform(test)
    true_test = test[TARGET_COLUMNS].to_numpy()
    pred_test = surrogate.predict_scaled(X_test)
    test_signs = np.where(true_test >= 0, 1.0, -1.0)
    test_scores = {
        **summarise(true_test, pred_test),
        "sign_accuracy": float((sign_model.predict(X_test) == test_signs).mean()),
        "max_abs_kcl_residual_A": float(np.abs(pred_test.sum(axis=1)).max()),
    }
    print(f"  {model_key} ({seconds:.1f}s)")
    print(f"    validation: R2(log) {val_scores['r2_log']:.4f}  "
          f"median {val_scores['median_error_pct']:.3f}%  p95 {val_scores['p95_error_pct']:.3f}%")
    print(f"    test:       R2(log) {test_scores['r2_log']:.4f}  "
          f"median {test_scores['median_error_pct']:.3f}%  p95 {test_scores['p95_error_pct']:.3f}%")

    surrogate.save(args.models_dir / f"{case}.joblib")
    parity_plot(true_test, pred_test, case, model_key, args.results_dir / f"parity_{case}.png")
    base = {"case": case, "model": model_key}
    return ({**base, "train_seconds": round(seconds, 1), **val_scores}, {**base, **test_scores})


def main(model_key, output_dir):
    args = parse_args(output_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    val_rows, test_rows = [], []
    for case in args.cases:
        val_row, test_row = train_case(model_key, case, args)
        val_rows.append(val_row)
        test_rows.append(test_row)

    pd.DataFrame(val_rows).to_csv(args.results_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_csv(args.results_dir / "test_metrics.csv", index=False)
    print(f"\nSaved models to {args.models_dir}\nSaved results to {args.results_dir}")
