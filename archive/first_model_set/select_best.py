"""Choose the best trained model for each case and write the final results.

Run this after training the models one by one with components_mlp/train.py, gpr/train.py and so on.
For each case it compares the models' VALIDATION results and picks the one with the lowest mean median
relative error over the four currents (mean p95 error breaks ties) - the same ranking as the benchmark in
Model_comparison/. The choice goes in ML_Models/results/<case>/ together with that case's comparison table
and test results.

    python select_best.py
    python select_best.py --choose pmos_on=gpr nmos_on=gpr     # override the choice
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common.config import CASES, MODEL_FILENAME, RESULTS_DIR, ROOT, TARGET_COLUMNS

MODEL_FOLDERS = {m: m for m in ("components_mlp", "gpr", "hybrid_analytic_mlp", "resnet_mlp")}
RANK = ["mean_median_error_pct", "mean_p95_error_pct"]


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
                 "for example: python resnet_mlp/train.py")
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
    medians = [f"{c}_median_error_pct" for c in TARGET_COLUMNS]
    chosen_rows = []
    for case in CASES:
        candidates = validation[validation["case"] == case]
        if candidates.empty:
            print(f"{case}: no trained model, skipped")
            continue
        if case in overrides:
            if overrides[case] not in set(candidates["model"]):
                sys.exit(f"{case}: model {overrides[case]!r} has not been trained for this case")
            model, reason = overrides[case], "chosen by you"
        else:
            model = candidates.sort_values(RANK).iloc[0]["model"]
            reason = "lowest validation mean median error"

        folder = args.root / MODEL_FOLDERS[model]
        case_dir = args.results_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "saved_models" / f"{case}.joblib", case_dir / MODEL_FILENAME)
        shutil.copyfile(folder / "results" / f"parity_{case}.png", case_dir / "parity.png")
        candidates.sort_values(RANK).to_csv(case_dir / "validation_comparison.csv", index=False)
        chosen = test[(test["case"] == case) & (test["model"] == model)]
        chosen.to_csv(case_dir / "test_metrics.csv", index=False)
        chosen_rows.append(chosen.iloc[0])
        print(f"{case}: {model} ({reason}) -> {case_dir}")

    show = ["case", "model"] + medians + ["mean_median_error_pct", "mean_p95_error_pct"]
    print("\nValidation comparison (median relative error %):")
    print(validation[show].sort_values(["case"] + RANK).to_string(index=False, float_format="{:.3f}".format))
    print("\nChosen models on the test set:")
    print(pd.DataFrame(chosen_rows)[show].to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
