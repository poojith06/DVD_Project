"""Hybrid model: analytical compact model + MLP that learns the remaining error (residual).

    log10|I| = analytical(df) + MLP(features, analytical prediction)

1. analytic.py: closed-form device equations (EKV channel, BSIM4-style gate tunnelling, junction leakage, GIDL)
   with ~20-30 constants per case, fitted by least squares on the training rows
2. one MLP (256-128-64 SiLU) per output group (id, is), ig, ib learns log10|I| - analytical; its inputs are the
   20 scaled features plus the standardised analytical prediction (common/nets.py recipe)
Idea from physics + ML hybrid compact models (Kao/Hu 2022, PRISM-UDE).
"""
import numpy as np

from common.config import CASES, GROUPS, MODEL_SEED
from common.nets import mlp, train_net
from hybrid_analytic_mlp.analytic import AnalyticModel


class HybridAnalyticMLPModel:
    name = "hybrid_analytic_mlp"

    def __init__(self, case, seed=MODEL_SEED, params=None):
        self.case, self.seed, self.params = case, seed, params

    def _inputs(self, X, a):
        return np.hstack([X, (a - self.a_mean) / self.a_std])

    def fit(self, X_train, y_train, X_val, y_val, df_train, df_val):
        self.analytic = AnalyticModel(self.case, CASES[self.case][0]).fit(df_train, y_train, df_val, y_val)
        a_train, a_val = self.analytic.predict(df_train), self.analytic.predict(df_val)
        self.a_mean, self.a_std = a_train.mean(0), a_train.std(0) + 1e-9
        I_train, I_val = self._inputs(X_train, a_train), self._inputs(X_val, a_val)
        r_train, r_val = y_train - a_train, y_val - a_val
        self.nets = {
            group: train_net(mlp, I_train, r_train[:, cols], I_val, r_val[:, cols], self.seed + i, self.params)
            for i, (group, cols) in enumerate(GROUPS.items())
        }
        return self

    def predict(self, X, df):
        a = self.analytic.predict(df)
        out, inputs = a.copy(), self._inputs(X, a)
        for group, cols in GROUPS.items():
            out[:, cols] += self.nets[group].predict(inputs)
        return out
