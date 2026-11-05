"""Choose the best individually trained model for each case and write the final results.

Run this after training the models one by one with mlp/train.py, xgboost_regressor/train.py
and so on. For each case it compares the models' validation results and picks the one with
the lowest validation 95th-percentile error (median error breaks ties). The choice goes in
DVD_Project/results/<case>/ together with that case's comparison table and test results.

    python select_best.py
    python select_best.py --choose pmos_off=mlp nmos_on=xgb     # override the choice
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common.config import CASES, MODEL_FILENAME, RESULTS_DIR, ROOT
from common.registry import MODEL_FOLDERS


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect every trained model's validation and test rows."""
    validation, test = [], []
    for folder in MODEL_FOLDERS.values():
        results = root / folder / "results"
        if (results / "validation_metrics.csv").exists():
            validation.append(pd.read_csv(results / "validation_metrics.csv"))
            test.append(pd.read_csv(results / "test_metrics.csv"))
    if not validation:
        sys.exit("No trained models found. Run a model's train.py first, "
                 "for example: python mlp/train.py")
    return pd.concat(validation, ignore_index=True), pd.concat(test, ignore_index=True)


def parse_choices(items) -> dict:
    choices = {}
    for item in items or []:
        case, _, model = item.partition("=")
        if case not in CASES or model not in MODEL_FOLDERS:
            sys.exit(f"Bad --choose value {item!r}. Use case=model, with case in "
                     f"{list(CASES)} and model in {list(MODEL_FOLDERS)}.")
        choices[case] = model
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--choose", nargs="+", metavar="CASE=MODEL",
                        help="use this model for a case instead of the best validation score")
    parser.add_argument("--root", type=Path, default=ROOT, help="folder holding the model folders")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    overrides = parse_choices(args.choose)

    validation, test = load_results(args.root)

    fmt = lambda v: f"{v:.4f}"
    chosen_rows = []
    for case in CASES:
        candidates = validation[validation["case"] == case]
        if candidates.empty:
            print(f"{case}: no trained model, skipped")
            continue
        if case in overrides:
            if overrides[case] not in set(candidates["model"]):
                sys.exit(f"{case}: model {overrides[case]!r} has not been trained for this case")
            model = overrides[case]
            reason = "chosen by you"
        else:
            model = candidates.sort_values(["p95_error_pct", "median_error_pct"]).iloc[0]["model"]
            reason = "lowest validation p95 error"

        folder = args.root / MODEL_FOLDERS[model]
        case_dir = args.results_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "saved_models" / f"{case}.joblib", case_dir / MODEL_FILENAME)
        shutil.copyfile(folder / "results" / f"parity_{case}.png", case_dir / "parity.png")
        candidates.sort_values("p95_error_pct").to_csv(case_dir / "validation_comparison.csv", index=False)
        chosen = test[(test["case"] == case) & (test["model"] == model)]
        chosen.to_csv(case_dir / "test_metrics.csv", index=False)
        chosen_rows.append(chosen.iloc[0])
        print(f"{case}: {model} ({reason}) -> {case_dir}")

    print("\nValidation comparison:")
    print(validation[["case", "model", "r2_log", "median_error_pct", "p95_error_pct"]]
          .sort_values(["case", "p95_error_pct"]).to_string(index=False, float_format=fmt))
    print("\nChosen models on the test set:")
    print(pd.DataFrame(chosen_rows)[["case", "model", "r2_log", "median_error_pct",
                                     "p95_error_pct", "sign_accuracy"]]
          .to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
