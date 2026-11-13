"""Near-crossing-robust error of the component model(s) (and of mlp_final for comparison).

robust error % = 100 * |pred - SPICE| / (sum over BSIM4 components of |contribution| to that terminal)
The denominator does not go to zero where id/is cross zero (channel and leakage cancel), unlike |SPICE|.
Needs: `python bench.py components_mlp` (saves component_data/models/<case>.pt) and results/mlp_final/<case>_logpred.npz.
Output: results/components_mlp/robust_metric.csv (+ printed table).
Usage : python component_data/eval_robust.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; MC = HERE.parent
sys.path.insert(0, str(MC))
import bench, model_components as mc
from bench import tf, CURRENTS

rows = []
for case in bench.ALL_CASES:
    S, X, Y, bundle = bench.load(case)
    models = {"components_mlp": mc.ComponentModel.load(HERE / "models" / f"{case}.pt")}
    if (HERE / "models" / f"{case}_vf.pt").exists():
        models["components_mlp_vf"] = mc.ComponentModelVF.load(HERE / "models" / f"{case}_vf.pt")
    pol = models["components_mlp"].polarity
    ref = np.load(MC / "results" / "mlp_final" / f"{case}_logpred.npz")
    for split in ("validation", "test"):
        d = S[split]; true = d[CURRENTS].to_numpy()
        C, _ = mc.components_for(case, d)
        denom = np.abs(mc.coefficients(d, pol) * C[:, None, :]).sum(2)          # (rows, 4)
        preds = {**{n: m.predict_signed(X[split], d) for n, m in models.items()},
                 "mlp_final": tf.predict_sign(bundle, d, ref[split]) * 10 ** ref[split]}
        for name, P in preds.items():
            err = 100 * np.abs(P - true) / denom
            for j, c in enumerate(CURRENTS):
                rows.append({"model": name, "case": case, "split": split, "current": c, "robust mean %": err[:, j].mean(),
                             "robust median %": np.median(err[:, j]), "robust p95 %": np.percentile(err[:, j], 95),
                             "robust max %": err[:, j].max()})
R = pd.DataFrame(rows)
R.to_csv(MC / "results" / "components_mlp" / "robust_metric.csv", index=False)
pd.set_option("display.width", 200)
print(R.pivot_table(index=["case", "split", "model"], columns="current", values=["robust median %", "robust p95 %"], sort=False).round(3))
