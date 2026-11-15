"""Train one model type on the four cases and report its train, validation and test results.

Used by each model folder's train.py. Models are trained one at a time; choosing the
best model for each case afterwards is done by select_best.py.
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # silences a harmless joblib warning on Windows

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import CASES, CURRENT_FLOOR, DATA_DIR, MODEL_SEED, SEED, SPLIT, TARGET_COLUMNS
from common.data import kirchhoff_residual, load_case, split
from common.metrics import log_magnitude, summarise
from common.registry import MODEL_TYPES
from common.signs import SignModel
from common.surrogate import TransistorSurrogate, fit_scaler


def parse_args(output_dir):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path(output_dir) / "results")
    parser.add_argument("--models-dir", type=Path, default=Path(output_dir) / "saved_models")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--split", nargs=3, type=float, default=SPLIT,
                        metavar=("TRAIN", "VAL", "TEST"), help="e.g. 0.8 0.1 0.1")
    parser.add_argument("--seed", type=int, default=SEED, help="data split seed")
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED,
                        help="network initialisation / batch order / GPR subset seed")
    parser.add_argument("--epochs", type=int, help="override the number of training epochs (neural networks)")
    return parser.parse_args()


def parity_plot(true, pred, case, model_name, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for k, (ax, name) in enumerate(zip(axes, TARGET_COLUMNS)):
        x = np.maximum(np.abs(true[:, k]), CURRENT_FLOOR)
        y = np.maximum(np.abs(pred[:, k]), CURRENT_FLOOR)
        ax.loglog(x, y, ".", markersize=2, alpha=0.5)
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.loglog(limits, limits, "k--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("SPICE |I| (A)")
        ax.set_ylabel("Predicted |I| (A)")
    fig.suptitle(f"{case}: {model_name}, test set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_case(model_key, case, args):
    """Train one model on one case, save it, and return {split: metrics row}."""
    df = load_case(case, args.data_dir)
    residual = kirchhoff_residual(df)
    print(f"\n=== {case}: {len(df)} rows | dataset KCL residual / largest current: "
          f"median {residual.median():.1e}, max {residual.max():.1e}", flush=True)

    splits = dict(zip(("train", "validation", "test"), split(df, args.split, args.seed)))
    surrogate = TransistorSurrogate(case, fit_scaler(splits["train"], case), None)
    X = {k: surrogate.transform(d) for k, d in splits.items()}
    Y = {k: log_magnitude(d[TARGET_COLUMNS].to_numpy()) for k, d in splits.items()}

    params = {"epochs": args.epochs} if args.epochs else None
    estimator = MODEL_TYPES[model_key](case, seed=args.model_seed, params=params)
    start = time.perf_counter()
    estimator.fit(X["train"], Y["train"], X["validation"], Y["validation"], splits["train"], splits["validation"])
    if not hasattr(estimator, "predict_signed"):
        surrogate.sign_model = SignModel(case).fit(
            X["train"], splits["train"], splits["train"][TARGET_COLUMNS].to_numpy(),
            estimator.predict(X["train"], splits["train"]))
    seconds = time.perf_counter() - start
    surrogate.estimator = estimator

    rows, preds = {}, {}
    for k, d in splits.items():
        preds[k] = surrogate.predict_array(d)
        rows[k] = {"case": case, "model": model_key, "split": k,
                   **summarise(d[TARGET_COLUMNS].to_numpy(), preds[k])}
    rows["validation"]["train_seconds"] = round(seconds, 1)

    print(f"  {model_key} ({seconds:.0f}s)   median % id / ig / is / ib   (p95 %)")
    for k in ("validation", "test"):
        r = rows[k]
        print(f"    {k:10s}  " + " / ".join(f"{r[f'{c}_median_error_pct']:.3f}" for c in TARGET_COLUMNS)
              + "   (" + " / ".join(f"{r[f'{c}_p95_error_pct']:.2f}" for c in TARGET_COLUMNS) + ")", flush=True)

    surrogate.save(args.models_dir / f"{case}.joblib")
    parity_plot(splits["test"][TARGET_COLUMNS].to_numpy(), preds["test"], case, model_key,
                args.results_dir / f"parity_{case}.png")
    return rows


def _merge(path, new_rows):
    """Write the rows for the cases just trained, keeping rows of other cases from an earlier run."""
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["case"].isin(new["case"])], new], ignore_index=True)
    new.sort_values("case").to_csv(path, index=False)


def main(model_key, output_dir):
    args = parse_args(output_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    collected = {"train": [], "validation": [], "test": []}
    for case in args.cases:
        rows = train_case(model_key, case, args)
        for k in collected:
            collected[k].append(rows[k])

    for k, name in (("train", "train_metrics.csv"), ("validation", "validation_metrics.csv"),
                    ("test", "test_metrics.csv")):
        _merge(args.results_dir / name, collected[k])
    print(f"\nSaved models to {args.models_dir}\nSaved results to {args.results_dir}")
