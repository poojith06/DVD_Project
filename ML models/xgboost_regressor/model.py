"""XGBoost: one boosted ensemble per terminal, stopped early on the validation set."""
import numpy as np

from common.config import SEED

XGB_PARAMS = {
    "n_estimators": 1200,
    "early_stopping_rounds": 50,
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 3,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    # Not reg:pseudohubererror: its Hessian vanishes on targets spanning many decades,
    # the leaf values explode, and training stops at iteration 0. See explanation.md.
    "objective": "reg:absoluteerror",
}


class XGBoostModel:
    name = "xgb"

    def __init__(self, params=None, seed=SEED):
        self.params = {**XGB_PARAMS, **(params or {})}
        self.seed = seed

    def fit(self, X_train, y_train, X_val, y_val, signs_train=None):
        from xgboost import XGBRegressor  # imported here so other models work without it

        self.models = []
        for k in range(y_train.shape[1]):
            model = XGBRegressor(**self.params, random_state=self.seed, n_jobs=-1)
            model.fit(
                np.asarray(X_train), y_train[:, k],
                eval_set=[(np.asarray(X_val), y_val[:, k])], verbose=False,
            )
            self.models.append(model)
        return self

    def predict(self, X):
        return np.column_stack([m.predict(np.asarray(X)) for m in self.models]).astype(float)
