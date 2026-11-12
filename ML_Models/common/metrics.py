"""Accuracy metrics for signed terminal currents (same definitions as the benchmark in Model_comparison/)."""
import numpy as np
from sklearn.metrics import r2_score

from common.config import CURRENT_FLOOR, TARGET_COLUMNS


def log_magnitude(currents: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.abs(currents), CURRENT_FLOOR))


def relative_error_percent(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """|pred - true| / |true| in percent. A wrong sign therefore costs at least 100%.

    If both currents are at the abstol floor, SPICE cannot tell them apart, so the
    error counts as zero. The 2x margin allows a floor-level prediction just above the floor.
    """
    errors = 100.0 * np.abs(pred - true) / np.maximum(np.abs(true), CURRENT_FLOOR)
    at_floor = (np.abs(true) <= CURRENT_FLOOR) & (np.abs(pred) <= 2 * CURRENT_FLOOR)
    return np.where(at_floor, 0.0, errors)


def summarise(true: np.ndarray, pred: np.ndarray) -> dict:
    """Per terminal: MAE, median and p95 of the relative error (%), R2 on log10|I|, sign accuracy (%).

    The mean_* values average the four terminals; mean_median_error_pct is the ranking score
    used by select_best.py. R2 is on log10|I| because in amps the large channel current
    would hide the much smaller gate and body currents.
    """
    errors = relative_error_percent(true, pred)
    log_true, log_pred = log_magnitude(true), log_magnitude(pred)
    result = {}
    for k, name in enumerate(TARGET_COLUMNS):
        result[f"{name}_mae_pct"] = float(errors[:, k].mean())
        result[f"{name}_median_error_pct"] = float(np.median(errors[:, k]))
        result[f"{name}_p95_error_pct"] = float(np.percentile(errors[:, k], 95))
        result[f"{name}_r2_log"] = (float(r2_score(log_true[:, k], log_pred[:, k]))
                                    if np.ptp(log_true[:, k]) > 0 else float("nan"))
        result[f"{name}_sign_acc_pct"] = float(
            100 * (np.sign(pred[:, k]) == np.where(true[:, k] >= 0, 1, -1)).mean())
    for metric in ("mae_pct", "median_error_pct", "p95_error_pct", "r2_log", "sign_acc_pct"):
        result[f"mean_{metric}"] = float(np.nanmean([result[f"{n}_{metric}"] for n in TARGET_COLUMNS]))
    return result
