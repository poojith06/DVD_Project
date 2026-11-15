"""EXP-09: why the gate leakage is computed with Kirchhoff's current law on the supply super-node.
Compares, for the final models, (a) the plain sum of the predicted currents into the terminals tied to the vdd node
(the literal reading of 'use the predicted currents to calculate the gate leakage') with (b) the super-node method.
Evaluation only. Output: results/analysis/report_numbers/plain_sum_check.csv"""
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
import numpy as np
import pandas as pd

import evaluate_gates as eg

F = REPO / "results"
models = eg.load_models(F / "models")
METHODS = ("plain sum of the currents into the vdd terminals", "super-node (Kirchhoff) method")
rows, spice_all, est_all = [], [], {m: [] for m in METHODS}
for gate in eg.GATES:
    df = pd.read_csv(eg.GATE_DATA / f"{gate}.csv")
    v, vdd, n = eg.node_voltages(df), df["vdd"].to_numpy(), len(df)
    plain = np.zeros(n)
    for _, pol, d, g, s, b, w in eg.GATES[gate]:
        p = "n_" if pol == "nmos" else "p_"
        x = pd.DataFrame({"vdd": df.vdd, "temp": df.temp, "w": w, "vd": v[d], "vs": v[s], "vg": v[g], "vb": v[b],
                          **{k: df[p + k] for k in eg.PROCESS}})
        on = v[g] > vdd / 2 if pol == "nmos" else v[g] < vdd / 2
        cur = np.zeros((n, 4))
        for state, mask in (("on", on), ("off", ~on)):
            if mask.any():
                cur[mask] = models[f"{pol}_{state}"].predict(x[mask]).to_numpy()
        for k, node in enumerate((d, g, s, b)):
            if node == "vdd":
                plain += cur[:, k]          # current into a terminal tied to vdd = current drawn from the supply
    kcl = eg.predict_gate(gate, df, models)
    spice_all.append(df.leak.to_numpy())
    for method, est in zip(METHODS, (plain, kcl)):
        e = 100 * (est - df.leak) / df.leak
        est_all[method].append(est)
        rows.append({"gate": gate, "method": method, "mean_accuracy_pct": 100 - e.abs().mean(),
                     "median_error_pct": e.abs().median(), "max_abs_error_pct": e.abs().max(),
                     "rows_with_negative_leakage": int((est < 0).sum()), "rows": n})
spice = np.concatenate(spice_all)
for method in METHODS:
    est = np.concatenate(est_all[method])
    e = 100 * (est - spice) / spice
    rows.append({"gate": "ALL", "method": method, "mean_accuracy_pct": 100 - np.abs(e).mean(),
                 "median_error_pct": np.median(np.abs(e)), "max_abs_error_pct": np.abs(e).max(),
                 "rows_with_negative_leakage": int((est < 0).sum()), "rows": len(est)})
t = pd.DataFrame(rows)
t.to_csv(F / "analysis" / "report_numbers" / "plain_sum_check.csv", index=False)
pd.set_option("display.width", 200)
print(t.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
