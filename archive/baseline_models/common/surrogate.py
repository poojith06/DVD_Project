"""TransistorSurrogate: the complete 12-inputs to 4-signed-currents model for one case."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from common.config import CASES, CURRENT_FLOOR, INPUT_COLUMNS, SEED, TARGET_COLUMNS
from common.features import build_features


def log_magnitude(currents: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.abs(currents), CURRENT_FLOOR))


def enforce_kirchhoff(currents: np.ndarray) -> np.ndarray:
    """Make each row sum to zero, correcting each terminal in proportion to I^2.

    Almost all of the correction lands on the largest currents, so small gate and
    body currents are not swamped by a small relative error in the channel current.
    """
    squares = currents ** 2
    weights = squares / squares.sum(axis=1, keepdims=True)
    return currents - weights * currents.sum(axis=1, keepdims=True)


class SignModel:
    """Predicts the sign of each terminal current, since log|I| discards it."""

    def __init__(self, seed=SEED):
        self.seed = seed

    def fit(self, X, signs):
        self.models = []
        for k in range(signs.shape[1]):
            classes = np.unique(signs[:, k])
            if len(classes) == 1:  # this terminal never changes sign
                self.models.append(float(classes[0]))
            else:
                self.models.append(
                    HistGradientBoostingClassifier(random_state=self.seed).fit(X, signs[:, k])
                )
        return self

    def predict(self, X):
        n_rows = len(X)
        return np.column_stack([
            np.full(n_rows, m) if isinstance(m, float) else m.predict(X)
            for m in self.models
        ])


class TransistorSurrogate:
    """Holds the feature scaler, regressor, sign model and bias correction together."""

    def __init__(self, case, scaler, estimator, sign_model):
        self.case = case
        self.polarity = CASES[case][0]
        self.scaler = scaler
        self.estimator = estimator
        self.sign_model = sign_model
        self.smear = np.ones(len(TARGET_COLUMNS))

    def calibrate_smear(self, X_val, y_val):
        """Duan's smearing factor per terminal, so summed leakage is not biased low.

        10**residual is very sensitive to outliers, and points near vd == vs can miss
        by decades. Residuals are limited to their 1st-99th percentile so that a few
        such points cannot shift the factor for every prediction.
        """
        residual = y_val - self.estimator.predict(X_val)
        low, high = np.percentile(residual, [1, 99], axis=0)
        self.smear = np.mean(10.0 ** np.clip(residual, low, high), axis=0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        return self.scaler.transform(build_features(df, self.polarity))

    def predict_scaled(self, X: np.ndarray, kirchhoff: bool = True) -> np.ndarray:
        # The floor also stops a badly extrapolated prediction reaching exactly 0 A,
        # which would make the Kirchhoff correction divide by zero.
        magnitudes = np.maximum(self.smear * 10.0 ** self.estimator.predict(X), CURRENT_FLOOR)
        currents = self.sign_model.predict(X) * magnitudes
        return enforce_kirchhoff(currents) if kirchhoff else currents

    def predict(self, df: pd.DataFrame, kirchhoff: bool = True) -> pd.DataFrame:
        """Predict signed currents in amps, with columns id, ig, is, ib."""
        return pd.DataFrame(
            self.predict_scaled(self.transform(df), kirchhoff),
            columns=TARGET_COLUMNS, index=df.index,
        )

    def save(self, path: Path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> "TransistorSurrogate":
        return joblib.load(path)


def fit_shared_parts(train: pd.DataFrame, case: str):
    """Fit what every model type shares for a case, using the training split only."""
    features = build_features(train, CASES[case][0])
    scaler = StandardScaler().fit(features)
    X_train = scaler.transform(features)
    signs = np.where(train[TARGET_COLUMNS].to_numpy() >= 0, 1.0, -1.0)
    return scaler, SignModel().fit(X_train, signs), X_train, signs
