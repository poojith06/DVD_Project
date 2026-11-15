"""Load a case's CSV, check it is usable, and split it."""
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import CASES, INPUT_COLUMNS, SEED, SPLIT, TARGET_COLUMNS


def load_case(case: str, data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / f"{case}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset {path}. Expected one CSV per case with columns: "
            f"{', '.join(INPUT_COLUMNS + TARGET_COLUMNS)}"
        )
    df = pd.read_csv(path)

    missing = [c for c in INPUT_COLUMNS + TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    columns = INPUT_COLUMNS + TARGET_COLUMNS
    finite = np.isfinite(df[columns].to_numpy(dtype=float)).all(axis=1)
    if not finite.all():
        print(f"  [{case}] dropping {(~finite).sum()} rows with NaN or inf values")
    df = df.loc[finite, columns].reset_index(drop=True)

    _check_rails(df, case)
    return df


def _check_rails(df: pd.DataFrame, case: str) -> None:
    """Catch a CSV saved under the wrong case name, which would give wrong features."""
    _, gate_rail, body_rail = CASES[case]
    for column, rail in (("vg", gate_rail), ("vb", body_rail)):
        expected = df["vdd"] if rail == "vdd" else 0.0
        if not np.allclose(df[column], expected, atol=1e-6):
            raise ValueError(f"{case}: {column} should equal {rail} on every row")


def kirchhoff_residual(df: pd.DataFrame) -> pd.Series:
    """|sum of terminal currents| relative to the largest terminal current in each row."""
    currents = df[TARGET_COLUMNS].to_numpy()
    return pd.Series(
        np.abs(currents.sum(axis=1)) / np.abs(currents).max(axis=1).clip(min=1e-300),
        index=df.index,
    )


def split(df: pd.DataFrame, fractions=SPLIT, seed: int = SEED):
    """Return (train, val, test) from one fixed-seed shuffle."""
    order = np.random.default_rng(seed).permutation(len(df))
    n_train = int(round(fractions[0] * len(df)))
    n_val = int(round(fractions[1] * len(df)))
    return (
        df.iloc[order[:n_train]].reset_index(drop=True),
        df.iloc[order[n_train:n_train + n_val]].reset_index(drop=True),
        df.iloc[order[n_train + n_val:]].reset_index(drop=True),
    )
