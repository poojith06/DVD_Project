"""TransistorSurrogate: the complete 12-inputs -> 4 signed currents model for one case."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common.config import CASES, INPUT_COLUMNS, TARGET_COLUMNS
from common.features import build_features


class TransistorSurrogate:
    """Holds the feature scaler, the regression model and the sign model of one case together."""

    def __init__(self, case, scaler, estimator, sign_model=None):
        self.case = case
        self.polarity = CASES[case][0]
        self.scaler = scaler
        self.estimator = estimator
        self.sign_model = sign_model  # None for models that predict signed currents themselves

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        return self.scaler.transform(build_features(df, self.polarity))

    def predict_array(self, df: pd.DataFrame) -> np.ndarray:
        """Signed currents in amps, shape (rows, 4) in the order id, ig, is, ib."""
        X = self.transform(df)
        if hasattr(self.estimator, "predict_signed"):
            return self.estimator.predict_signed(X, df)
        log_pred = self.estimator.predict(X, df)
        return self.sign_model.predict(X, df, log_pred) * 10.0 ** log_pred

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict signed currents in amps, with columns id, ig, is, ib."""
        return pd.DataFrame(self.predict_array(df), columns=TARGET_COLUMNS, index=df.index)

    def save(self, path: Path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> "TransistorSurrogate":
        return joblib.load(path)


def fit_scaler(train: pd.DataFrame, case: str) -> StandardScaler:
    """Feature scaler fitted on the training split only."""
    return StandardScaler().fit(build_features(train, CASES[case][0]))
