"""Data, features, metrics, current signs and the saved-model wrapper (everything except the models)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from config import (ACTIVE_COLUMNS, CASES, CURRENT_FLOOR, INPUT_COLUMNS, RAIL_TAU, SEED, SPLIT,
                    TARGET_COLUMNS)

# ----------------------------------------------------------------------------- data

def load_case(case: str, data_dir: Path) -> pd.DataFrame:
    """Read <data_dir>/<case>.csv, drop non-finite rows, check vg/vb match the case."""
    df = pd.read_csv(Path(data_dir) / f"{case}.csv")
    missing = [c for c in INPUT_COLUMNS + TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{case}.csv is missing columns: {missing}")
    columns = INPUT_COLUMNS + TARGET_COLUMNS
    finite = np.isfinite(df[columns].to_numpy(float)).all(axis=1)
    if not finite.all():
        print(f"  [{case}] dropping {(~finite).sum()} rows with NaN or inf values")
    df = df.loc[finite, columns].reset_index(drop=True)
    _, gate_rail, body_rail = CASES[case]
    for column, rail in (("vg", gate_rail), ("vb", body_rail)):
        expected = df["vdd"] if rail == "vdd" else 0.0
        if not np.allclose(df[column], expected, atol=1e-6):
            raise ValueError(f"{case}: {column} should equal {rail} on every row")
    return df


def kirchhoff_residual(df: pd.DataFrame) -> pd.Series:
    """|id + ig + is + ib| / largest |terminal current| per row."""
    c = df[TARGET_COLUMNS].to_numpy()
    return pd.Series(np.abs(c.sum(1)) / np.abs(c).max(1).clip(min=1e-300), index=df.index)


def split(df: pd.DataFrame, fractions=SPLIT, seed: int = SEED):
    """(train, validation, test) from one fixed-seed shuffle."""
    order = np.random.default_rng(seed).permutation(len(df))
    n_tr, n_va = int(round(fractions[0] * len(df))), int(round(fractions[1] * len(df)))
    parts = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
    return tuple(df.iloc[p].reset_index(drop=True) for p in parts)


def rail_distance(df: pd.DataFrame) -> np.ndarray:
    """dist(vd) + dist(vs), with dist(v) = distance from v to the nearer of 0 and VDD (volts)."""
    vdd = df["vdd"].to_numpy()
    dist = lambda v: np.minimum(v, vdd - v).clip(min=0)
    return dist(df["vd"].to_numpy()) + dist(df["vs"].to_numpy())


def rail_weights(df: pd.DataFrame, strength: float) -> np.ndarray:
    """Training-loss weight per row: 1 + strength * exp(-r / RAIL_TAU), normalised to mean 1."""
    w = 1.0 + strength * np.exp(-rail_distance(df) / RAIL_TAU)
    return w / w.mean()

# ----------------------------------------------------------------------------- features

BOLTZMANN_OVER_Q = 8.617333e-5  # V/K


def build_features(df: pd.DataFrame, polarity: str) -> pd.DataFrame:
    """The 10 varying inputs + 10 physics features (effective source, Vgs/Vbs, |Vds|, 1/T, subthreshold
    term, oxide fields, log W, log(1 - exp(-|Vds|/VT)))."""
    vd, vs, vg, vb = df["vd"], df["vs"], df["vg"], df["vb"]
    kelvin = df["temp"] + 273.15
    vt = BOLTZMANN_OVER_Q * kelvin
    vds = vd - vs
    src = np.minimum(vd, vs) if polarity == "nmos" else np.maximum(vd, vs)   # effective source
    vgs = vg - src
    f = df[ACTIVE_COLUMNS].copy()
    f["vds"] = vds
    f["abs_vds"] = vds.abs()
    f["vgs_eff"] = vgs
    f["vbs_eff"] = vb - src
    f["inv_temp"] = 1.0 / kelvin
    f["phi_sub"] = (vgs.abs() - df["vth0"].abs()) / vt
    f["eox_d"] = (vg - vd).abs() / df["toxe"]
    f["eox_s"] = (vg - vs).abs() / df["toxe"]
    f["log_w"] = np.log10(df["w"])
    f["log_vds_term"] = np.log10((-np.expm1(-vds.abs() / vt)).clip(lower=1e-12))
    return f

# ----------------------------------------------------------------------------- metrics

def log_magnitude(currents: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.abs(currents), CURRENT_FLOOR))


def relative_error_percent(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """|pred - true| / |true| in %, zero when both sit at the abstol floor."""
    err = 100.0 * np.abs(pred - true) / np.maximum(np.abs(true), CURRENT_FLOOR)
    at_floor = (np.abs(true) <= CURRENT_FLOOR) & (np.abs(pred) <= 2 * CURRENT_FLOOR)
    return np.where(at_floor, 0.0, err)


def summarise(true: np.ndarray, pred: np.ndarray) -> dict:
    """Per current: MAE, median, p95 relative error (%), R2 on log10|I|, sign accuracy; mean_* over the four."""
    err = relative_error_percent(true, pred)
    lt, lp = log_magnitude(true), log_magnitude(pred)
    out = {}
    for k, c in enumerate(TARGET_COLUMNS):
        out[f"{c}_mae_pct"] = float(err[:, k].mean())
        out[f"{c}_median_error_pct"] = float(np.median(err[:, k]))
        out[f"{c}_p95_error_pct"] = float(np.percentile(err[:, k], 95))
        out[f"{c}_r2_log"] = float(r2_score(lt[:, k], lp[:, k])) if np.ptp(lt[:, k]) > 0 else float("nan")
        out[f"{c}_sign_acc_pct"] = float(100 * (np.sign(pred[:, k]) == np.where(true[:, k] >= 0, 1, -1)).mean())
    for m in ("mae_pct", "median_error_pct", "p95_error_pct", "r2_log", "sign_acc_pct"):
        out[f"mean_{m}"] = float(np.nanmean([out[f"{c}_{m}"] for c in TARGET_COLUMNS]))
    return out

# ----------------------------------------------------------------------------- signs

class SignModel:
    """Sign of each current (the models predict log10|I|).
    ig, ib: fixed sign per case if it never changes on the training rows, else a classifier.
    ON:  sign(id) = sign(vd - vs), is opposite (channel current flows from the higher terminal).
    OFF: the effective drain has a fixed sign (+ NMOS, - PMOS); the effective source's sign comes from a
         gradient-boosting classifier on the features + the KCL margin log10(|ig| + |ib|) - log10|I_drain|."""

    def __init__(self, case):
        self.case, self.polarity, self.on = case, CASES[case][0], case.endswith("_on")

    def _off_parts(self, df, log_pred):
        d_is_drain = ((df.vd > df.vs) if self.polarity == "nmos" else (df.vd < df.vs)).to_numpy()
        a = 10.0 ** log_pred
        margin = np.log10(a[:, 1] + a[:, 3]) - np.log10(np.where(d_is_drain, a[:, 0], a[:, 2]))
        return d_is_drain, margin

    def fit(self, X, df, true_currents, log_pred):
        signs = np.where(true_currents >= 0, 1.0, -1.0)
        self.other = {}
        for k in (1, 3):
            classes = np.unique(signs[:, k])
            self.other[k] = (float(classes[0]) if len(classes) == 1
                             else HistGradientBoostingClassifier(random_state=SEED).fit(X, signs[:, k]))
        if not self.on:
            d_is_drain, margin = self._off_parts(df, log_pred)
            source_true = np.where(d_is_drain, signs[:, 2], signs[:, 0])
            self.source_clf = HistGradientBoostingClassifier(random_state=SEED).fit(np.c_[X, margin], source_true)
        return self

    def predict(self, X, df, log_pred):
        s = np.zeros((len(df), 4))
        for k, m in self.other.items():
            s[:, k] = m if isinstance(m, float) else m.predict(X)
        if self.on:
            vds = np.sign(df["vd"] - df["vs"]).replace(0, 1).to_numpy()
            s[:, 0], s[:, 2] = vds, -vds
        else:
            drain = 1.0 if self.polarity == "nmos" else -1.0
            d_is_drain, margin = self._off_parts(df, log_pred)
            source = self.source_clf.predict(np.c_[X, margin])
            s[:, 0] = np.where(d_is_drain, drain, source)
            s[:, 2] = np.where(d_is_drain, source, drain)
        return s

# ----------------------------------------------------------------------------- saved model

class TransistorSurrogate:
    """Feature scaler + regression model + sign model of one case: 12 inputs -> 4 signed currents (A)."""

    def __init__(self, case, scaler, estimator=None, sign_model=None):
        self.case, self.polarity = case, CASES[case][0]
        self.scaler, self.estimator, self.sign_model = scaler, estimator, sign_model

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(build_features(df, self.polarity))

    def predict_array(self, df: pd.DataFrame) -> np.ndarray:
        X = self.transform(df)
        log_pred = self.estimator.predict(X, df)
        return self.sign_model.predict(X, df, log_pred) * 10.0 ** log_pred

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.predict_array(df), columns=TARGET_COLUMNS, index=df.index)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> "TransistorSurrogate":
        return joblib.load(path)


def fit_scaler(train: pd.DataFrame, case: str) -> StandardScaler:
    return StandardScaler().fit(build_features(train, CASES[case][0]))
