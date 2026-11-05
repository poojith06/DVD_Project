"""Predict terminal currents with a saved model.

    python predict.py --case nmos_off --input samples.csv --output currents.csv
    python predict.py --case nmos_off --input s.csv --output c.csv --models-dir mlp/saved_models

By default this uses the chosen model in DVD_Project/results/<case>/.
Other scripts can import load_surrogate and call .predict(dataframe) directly.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common.config import CASES, MODEL_FILENAME, RESULTS_DIR
from common.surrogate import TransistorSurrogate


def load_surrogate(case: str, models_dir=None) -> TransistorSurrogate:
    """Load the chosen model for a case, or <models_dir>/<case>.joblib if a folder is given.

    A model folder's own saved_models/ (for example mlp/saved_models) is a models_dir.
    """
    if models_dir is None:
        return TransistorSurrogate.load(RESULTS_DIR / case / MODEL_FILENAME)
    return TransistorSurrogate.load(Path(models_dir) / f"{case}.joblib")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, choices=list(CASES))
    parser.add_argument("--input", required=True, type=Path, help="CSV with the 12 input columns")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models-dir", type=Path,
                        help="use <models-dir>/<case>.joblib instead of the chosen model")
    args = parser.parse_args()

    load_surrogate(args.case, args.models_dir).predict(pd.read_csv(args.input)).to_csv(
        args.output, index=False
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
