"""Error analysis numbers for the report (final models; reads the saved predictions only):
 - Part B and Part C: mean |error| by temperature range and by NMOS / PMOS vth0 quartile, ML and (Part C) ceiling
 - Part B: largest and smallest case errors, worst rows
 - Part C: worst rows
Output: results/analysis/report_numbers/error_breakdown_*.csv and error_breakdown.txt"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
F = REPO / "results"
OUT = F / "analysis" / "report_numbers"
lines = []
say = lambda s="": (print(s), lines.append(s))


def breakdown(df, err_cols):
    out = []
    bins = {"temperature": pd.cut(df.temp, [-40, 0, 40, 85, 125], include_lowest=True),
            "NMOS vth0 quartile": pd.qcut(df.n_vth0, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]),
            "PMOS |vth0| quartile": pd.qcut(df.p_vth0.abs(), 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]),
            "VDD tercile": pd.qcut(df.vdd, 3, labels=["low VDD", "mid VDD", "high VDD"])}
    for name, b in bins.items():
        for key, g in df.groupby(b, observed=True):
            row = {"variable": name, "range": str(key), "rows": len(g)}
            for c in err_cols:
                row[f"mean_abs_error_pct_{c}"] = g[c].abs().mean()
                row[f"bias_pct_{c}"] = g[c].median()
            out.append(row)
    return pd.DataFrame(out)


b = pd.read_csv(F / "partB" / "gate_predictions.csv", dtype={"inputs": str})
c = pd.read_csv(F / "partC" / "cla3_predictions.csv")
c["error_ml_pct"] = c.error_ml_pct.astype(float)
bb = breakdown(b.rename(columns={"error_pct": "ml"}), ["ml"])
bc = breakdown(c.rename(columns={"error_ml_pct": "ml", "error_ceiling_pct": "ceiling"}), ["ml", "ceiling"])
bb.to_csv(OUT / "error_breakdown_partB.csv", index=False)
bc.to_csv(OUT / "error_breakdown_partC.csv", index=False)
pd.set_option("display.width", 200)
say("Part B (3,600 rows): mean |error| (%) and median signed error (%) by condition")
say(bb.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
say("\nPart C (12,800 rows): ML and ceiling")
say(bc.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
t = pd.read_csv(F / "partB" / "gate_results.csv", dtype={"inputs": str}).sort_values("mean_accuracy_pct")
say("\nPart B cases from least to most accurate (mean accuracy %):")
say(t[["gate", "inputs", "mean_accuracy_pct", "median_error_pct", "p95_error_pct", "max_error_pct", "bias_pct"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
w = b.reindex(b.error_pct.abs().sort_values(ascending=False).index).head(8)
say("\nPart B worst rows:")
say(w[["gate", "inputs", "corner", "temp", "vdd", "n_vth0", "p_vth0", "leak_spice", "leak_ml", "error_pct"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
w = c.reindex(c.error_ml_pct.abs().sort_values(ascending=False).index).head(8)
say("\nPart C worst rows:")
say(w[["vector", "corner", "temp", "vdd", "leak_spice", "leak_ml", "error_ml_pct", "error_ceiling_pct"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
# how many rows exceed 5 %
say(f"\nrows with |error| > 5 %: Part B {int((b.error_pct.abs() > 5).sum())} of {len(b)}; Part C {int((c.error_ml_pct.abs() > 5).sum())} of {len(c)}")
say(f"rows with |error| > 1 %: Part B {int((b.error_pct.abs() > 1).sum())}; Part C {int((c.error_ml_pct.abs() > 1).sum())}")
(OUT / "error_breakdown.txt").write_text("\n".join(lines), encoding="utf-8")
