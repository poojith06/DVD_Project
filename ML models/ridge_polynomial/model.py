"""Ridge polynomial: polynomial expansion plus ridge, with the penalty chosen per terminal."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common.config import SEED

RIDGE_PARAMS = {
    "degree": 3,
    "alphas": (1e-3, 1e2, 50),  # start, stop, count on a log scale
}


class RidgePolynomialModel:
    name = "ridge"

    def __init__(self, params=None, seed=SEED):
        self.params = {**RIDGE_PARAMS, **(params or {})}

    def fit(self, X_train, y_train, X_val=None, y_val=None, signs_train=None):
        start, stop, count = self.params["alphas"]
        self.pipeline = make_pipeline(
            PolynomialFeatures(self.params["degree"], include_bias=False),
            StandardScaler(),
            RidgeCV(
                alphas=np.logspace(np.log10(start), np.log10(stop), count),
                alpha_per_target=True,
            ),
        ).fit(np.asarray(X_train), y_train)
        return self

    def predict(self, X):
        return self.pipeline.predict(np.asarray(X))
