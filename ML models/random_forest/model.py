"""Random Forest: one forest per terminal, so large terminals cannot dominate the splits."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from common.config import SEED

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": 0.8,
}


class RandomForestModel:
    name = "rf"

    def __init__(self, params=None, seed=SEED):
        self.params = {**RF_PARAMS, **(params or {})}
        self.seed = seed

    def fit(self, X_train, y_train, X_val=None, y_val=None, signs_train=None):
        self.models = [
            RandomForestRegressor(**self.params, random_state=self.seed, n_jobs=-1).fit(
                np.asarray(X_train), y_train[:, k]
            )
            for k in range(y_train.shape[1])
        ]
        return self

    def predict(self, X):
        return np.column_stack([m.predict(np.asarray(X)) for m in self.models])
