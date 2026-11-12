"""Sign of each terminal current (the models predict log10|I|, which has no sign).

Same method as Final_models (used for every model in the benchmark):
  ig, ib     one fixed sign per case if the training rows never change sign, else a classifier
  ON cases   id/is: physics rule - the channel current flows from the higher of vd/vs to the lower,
             so sign(id) = sign(vd - vs) and sign(is) = -sign(vd - vs)
  OFF cases  the effective drain (higher of vd/vs for NMOS, lower for PMOS) always has the sign +1 (NMOS) / -1 (PMOS);
             the effective source's sign comes from a gradient-boosting classifier whose inputs are the features plus
             the KCL margin log10(|ig| + |ib|) - log10|I_eff_drain| (> 0: gate + body leakage exceed the drain current)
The components_mlp model predicts signed currents itself and does not use this.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from common.config import CASES, SEED


class SignModel:
    def __init__(self, case):
        self.case, self.polarity = case, CASES[case][0]
        self.on = case.endswith("_on")

    def _off_parts(self, df, log_pred):
        d_is_drain = ((df.vd > df.vs) if self.polarity == "nmos" else (df.vd < df.vs)).to_numpy()
        A = 10.0 ** log_pred
        margin = np.log10(A[:, 1] + A[:, 3]) - np.log10(np.where(d_is_drain, A[:, 0], A[:, 2]))
        return d_is_drain, margin

    def fit(self, X, df, true_currents, log_pred):
        """X: scaled features, true_currents: (rows, 4) SPICE currents, log_pred: the model's own log10|I| (train rows)."""
        signs = np.where(true_currents >= 0, 1.0, -1.0)
        self.other = {}
        for k in (1, 3):                                    # ig, ib
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
            drain_sign = 1.0 if self.polarity == "nmos" else -1.0
            d_is_drain, margin = self._off_parts(df, log_pred)
            source = self.source_clf.predict(np.c_[X, margin])
            s[:, 0] = np.where(d_is_drain, drain_sign, source)
            s[:, 2] = np.where(d_is_drain, source, drain_sign)
        return s
