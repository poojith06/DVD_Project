"""Collect results/<model>/<case>_metrics.csv into one table and pick the best model per case.

Ranking uses VALIDATION only (test is reported, never used to choose):
  score = mean over the 4 currents of the validation median %   (lower is better)
  ties / close calls: look at the validation p95 % as well.
Output: results/summary_long.csv (every row), results/summary_<split>.csv (one line per model and case), printed tables.
Usage : python summarize.py
"""
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)

files = sorted((HERE / "results").glob("*/*_metrics.csv"))
L = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
L.to_csv(HERE / "results" / "summary_long.csv", index=False)
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]; CUR = ["id", "ig", "is", "ib"]

def wide(split):
    d = L[L.split == split]
    W = d.pivot_table(index=["case", "model"], columns="current", values=["median %", "p95 %", "MAE %"])
    W = W.reindex(columns=pd.MultiIndex.from_product([["median %", "p95 %", "MAE %"], CUR]))
    W[("score", "mean median %")] = W["median %"].mean(1); W[("score", "mean p95 %")] = W["p95 %"].mean(1)
    return W

V, T = wide("validation"), wide("test")
for s, W in (("validation", V), ("test", T)):
    flat = W.copy(); flat.columns = [f"{a} {b}" for a, b in W.columns]; flat.to_csv(HERE / "results" / f"summary_{s}.csv")

for case in CASES:
    if case not in V.index.get_level_values(0): continue
    v = V.loc[case].sort_values(("score", "mean median %")); t = T.loc[case].reindex(v.index)
    tab = pd.DataFrame({
        "val median id/ig/is/ib": v["median %"].round(2).astype(str).agg(" / ".join, axis=1),
        "val mean median": v[("score", "mean median %")].round(3),
        "val mean p95": v[("score", "mean p95 %")].round(2),
        "test median id/ig/is/ib": t["median %"].round(2).astype(str).agg(" / ".join, axis=1),
        "test p95 id/ig/is/ib": t["p95 %"].round(1).astype(str).agg(" / ".join, axis=1),
        "test MAE id/ig/is/ib": t["MAE %"].round(2).astype(str).agg(" / ".join, axis=1),
    })
    print(f"\n=== {case.upper()}  (sorted by validation mean median %; best = first row) ===")
    print(tab.to_string())
