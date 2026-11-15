"""Choose the best model type per case on the VALIDATION set and write the Part A results.

Ranking: lowest mean validation median relative error over the four currents (mean p95 breaks ties).

    python select_best.py                 # uses runs/final
    python select_best.py --tag final --choose nmos_on=gpr

Output (results/):
  model_comparison_validation.csv   4 model types x 4 cases: R2 (log), median and p95 error on validation
  chosen_models_test.csv            chosen model per case: test median / p95 error
  <case>/model.joblib, parity.png, validation_comparison.csv, test_metrics.csv
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from config import CASES, MODEL_TYPES, RESULTS_DIR, RUNS_DIR, TARGET_COLUMNS

RANK = ["mean_median_error_pct", "mean_p95_error_pct"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="final")
    p.add_argument("--choose", nargs="+", default=[], metavar="CASE=MODEL")
    args = p.parse_args()
    overrides = dict(item.split("=", 1) for item in args.choose)
    run = RUNS_DIR / args.tag
    val, test = [], []
    for m in MODEL_TYPES:
        r = run / m / "results"
        if (r / "validation_metrics.csv").exists():
            val.append(pd.read_csv(r / "validation_metrics.csv"))
            test.append(pd.read_csv(r / "test_metrics.csv"))
    if not val:
        sys.exit(f"No trained models in {run}")
    val, test = pd.concat(val, ignore_index=True), pd.concat(test, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # R2 is computed on log10|I| (decades) per current, plus the mean of the four; errors are relative errors in %
    cols = ["case", "model"] + [f"{c}_{m}" for m in ("r2_log", "median_error_pct", "p95_error_pct")
                                for c in TARGET_COLUMNS] + ["mean_r2_log"] + RANK
    val.sort_values(["case"] + RANK)[cols].to_csv(RESULTS_DIR / "model_comparison_validation.csv", index=False)

    chosen = []
    for case in CASES:
        cand = val[val.case == case].sort_values(RANK)
        if cand.empty:
            continue
        model = overrides.get(case, cand.iloc[0]["model"])
        d = RESULTS_DIR / case
        d.mkdir(exist_ok=True)
        shutil.copyfile(run / model / "saved_models" / f"{case}.joblib", d / "model.joblib")
        shutil.copyfile(run / model / "results" / f"parity_{case}.png", d / "parity.png")
        cand.to_csv(d / "validation_comparison.csv", index=False)
        t = test[(test.case == case) & (test.model == model)]
        t.to_csv(d / "test_metrics.csv", index=False)
        chosen.append(t.iloc[0])
        print(f"{case}: {model}" + (" (chosen by you)" if case in overrides else " (best validation score)"))
    show = ["case", "model"] + [f"{c}_median_error_pct" for c in TARGET_COLUMNS] + RANK
    pd.DataFrame(chosen)[["case", "model"] + [f"{c}_{m}" for m in ("median_error_pct", "p95_error_pct")
                                              for c in TARGET_COLUMNS]].to_csv(RESULTS_DIR / "chosen_models_test.csv",
                                                                               index=False)
    print("\nValidation (median relative error %):")
    print(val.sort_values(["case"] + RANK)[show].to_string(index=False, float_format="{:.3f}".format))
    print("\nChosen models, test set:")
    print(pd.DataFrame(chosen)[show].to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
