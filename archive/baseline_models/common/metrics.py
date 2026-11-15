"""Accuracy metrics for signed terminal currents."""
import numpy as np
from sklearn.metrics import r2_score

from common.config import CURRENT_FLOOR, TARGET_COLUMNS
from common.surrogate import log_magnitude


def relative_error_percent(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """|pred - true| / |true| in percent. A wrong sign therefore costs at least 100%.

    If both currents are at the abstol floor, SPICE cannot tell them apart, so the
    error counts as zero. The 2x margin allows for the smearing factor, which can lift
    a floor-level prediction just above the floor.
    """
    errors = 100.0 * np.abs(pred - true) / np.maximum(np.abs(true), CURRENT_FLOOR)
    at_floor = (np.abs(true) <= CURRENT_FLOOR) & (np.abs(pred) <= 2 * CURRENT_FLOOR)
    return np.where(at_floor, 0.0, errors)


def summarise(true: np.ndarray, pred: np.ndarray) -> dict:
    """R2 and error percentiles, pooled over all four terminals and per terminal.

    R2 is measured on log10|I|. In linear amps the ON channel current would dominate
    the score and hide errors in the much smaller gate and body currents. The pooled
    R2 averages the per-terminal values, because pooling the raw values would reward
    the fixed offset between terminals instead of real accuracy. A terminal that sits
    at the floor on every row has no variance, so its R2 is undefined (NaN) and it is
    left out of the average.
    """
    errors = relative_error_percent(true, pred)
    log_true, log_pred = log_magnitude(true), log_magnitude(pred)
    per_terminal_r2 = [
        r2_score(log_true[:, k], log_pred[:, k]) if np.ptp(log_true[:, k]) > 0 else np.nan
        for k in range(true.shape[1])
    ]

    result = {
        "r2_log": float(np.nanmean(per_terminal_r2)),
        "median_error_pct": float(np.median(errors)),
        "p95_error_pct": float(np.percentile(errors, 95)),
    }
    for k, name in enumerate(TARGET_COLUMNS):
        result[f"{name}_r2_log"] = float(per_terminal_r2[k])
        result[f"{name}_median_error_pct"] = float(np.median(errors[:, k]))
        result[f"{name}_p95_error_pct"] = float(np.percentile(errors[:, k], 95))
    return result
