"""EXP-17: accuracy of the final models under the four definitions of "mean accuracy" (the project document does not
define it). Reads the saved predictions in results (nothing is re-evaluated or re-trained).

 (1) 100 - mean |e|            e = (ML - SPICE) / SPICE, mean over all rows
 (2) 100 - median |e|
 (3) 100 - |mean e|            signed errors cancel
 (4) mean over groups of 100 - |median(ML) / median(SPICE) - 1|   groups: the 18 gate cases (Part B), the 64 vectors (Part C)
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
F = REPO / "results"


def defs(spice, est, group):
    e = 100 * (est - spice) / spice
    df = pd.DataFrame({"g": group.to_numpy(), "s": spice.to_numpy(), "m": est.to_numpy()})
    g = df.groupby("g")
    ratio = 100 * (g.m.median() / g.s.median() - 1).abs()
    return {"(1) 100 - mean |e|": 100 - e.abs().mean(), "(2) 100 - median |e|": 100 - e.abs().median(),
            "(3) 100 - |mean e|": 100 - abs(e.mean()), "(4) mean over groups of 100 - |median ratio - 1|": 100 - ratio.mean()}


b = pd.read_csv(F / "partB" / "gate_predictions.csv", dtype={"inputs": str})
c = pd.read_csv(F / "partC" / "cla3_predictions.csv")
out = pd.DataFrame({
    "Part B, ML (3,600 rows, 18 groups)": defs(b.leak_spice, b.leak_ml, b.gate + "_" + b.inputs),
    "Part C, ML (12,800 rows, 64 groups)": defs(c.leak_spice, c.leak_ml, c.vector),
    "Part C, ceiling (perfect gate models)": defs(c.leak_spice, c.leak_ceiling, c.vector)})
out.index.name = "definition"
out.to_csv(F / "analysis" / "report_numbers" / "accuracy_definitions.csv")
print(out.round(2).to_string())
