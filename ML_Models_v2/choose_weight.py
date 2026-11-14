"""Choose the near-rail training-weight strength from the tuning runs, on VALIDATION data only.

The tuning runs train resnet_mlp on the two OFF cases (whose drain/source currents set the gate leakage) with
several strengths: runs/tune_A<strength>/. Rule: lowest near-rail validation mean error of id and is (OFF cases),
among the strengths whose normal validation score (mean median error over the 4 currents) is at most 10 % worse
than without weighting. Writes runs/chosen_rail_weight.txt.

    python choose_weight.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from config import RUNS_DIR

rows = []
for run in sorted(RUNS_DIR.glob("tune_A*")):
    f = run / "resnet_mlp" / "results" / "validation_metrics.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows.append({"rail_weight": float(d.rail_weight.iloc[0]),
                     "val_mean_median_pct": d.mean_median_error_pct.mean(),
                     "near_rail_id_is_mean_pct": d[["nr_id_mae_pct", "nr_is_mae_pct"]].to_numpy().mean(),
                     **{f"{c}_nr_id_is": d.loc[d.case == c, ["nr_id_mae_pct", "nr_is_mae_pct"]].to_numpy().mean()
                        for c in d.case}})
if not rows:
    sys.exit("No tuning runs found (runs/tune_A*/resnet_mlp).")
t = pd.DataFrame(rows).sort_values("rail_weight")
base = t.loc[t.rail_weight == 0, "val_mean_median_pct"]
limit = 1.10 * (base.iloc[0] if len(base) else t.val_mean_median_pct.min())
ok = t[t.val_mean_median_pct <= limit]
best = ok.sort_values("near_rail_id_is_mean_pct").iloc[0]
print(t.round(4).to_string(index=False))
print(f"\nchosen rail weight: {best.rail_weight:g} (normal validation score limit {limit:.4f} %)")
t.to_csv(RUNS_DIR / "rail_weight_tuning.csv", index=False)
(RUNS_DIR / "chosen_rail_weight.txt").write_text(f"{best.rail_weight:g}\n")
