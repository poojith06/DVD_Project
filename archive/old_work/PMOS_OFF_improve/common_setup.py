"""Shared setup for the PMOS OFF improvement study: same data split and features as the repo's ML models/ code."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "ML models"))
from common.config import SEED, SPLIT, TARGET_COLUMNS, CURRENT_FLOOR
from common.data import load_case, split
from common.features import build_features
from sklearn.preprocessing import StandardScaler

DATA = {"remeasured": REPO / "Datasets_hires" / "data", "original": REPO / "Datasets" / "data"}

def load(which="remeasured"):
    df = load_case("pmos_off", DATA[which]); tr, va, te = split(df, SPLIT, SEED)
    sc = StandardScaler().fit(build_features(tr, "pmos"))
    X = {k: sc.transform(build_features(d, "pmos")) for k, d in (("tr", tr), ("va", va), ("te", te))}
    I = {k: d[TARGET_COLUMNS].to_numpy() for k, d in (("tr", tr), ("va", va), ("te", te))}
    Y = {k: np.log10(np.maximum(np.abs(v), CURRENT_FLOOR)) for k, v in I.items()}
    return X, Y, I, (tr, va, te), sc

def mag_err(log_pred, I_true):
    """relative error (%) of the predicted MAGNITUDE, |10^pred - |I|| / |I| (sign handled separately)"""
    a = np.maximum(np.abs(I_true), CURRENT_FLOOR)
    return 100 * np.abs(10.0 ** log_pred - a) / a

def summ(log_pred, I_true):
    e = mag_err(log_pred, I_true)
    return {t: (np.median(e[:, j]), np.percentile(e[:, j], 95), e[:, j].mean()) for j, t in enumerate(TARGET_COLUMNS)}
