"""Choose the resnet_mlp settings for the OFF cases from the tuning runs, on VALIDATION data only.

Candidates: runs/tune_A0 (the default settings, rail weight 0) and runs/rn_<name>/ (one changed setting each), all
trained on nmos_off and pmos_off with the 80:10:10 split. One setting is chosen for both OFF cases.
Rule (fixed before the runs): the lowest near-rail validation mean error of id and is (rows with both terminals
within 0.2 V of a rail, the operating points logic gates use), averaged over the two OFF cases, among the settings
whose normal validation score (mean median error over the four currents) is at most 5 % worse than the default.
Writes runs/resnet_tuning.csv and runs/chosen_resnet.txt (the run tag).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from config import RUNS_DIR

rows = []
for run in [RUNS_DIR / "tune_A0", *sorted(RUNS_DIR.glob("rn_*"))]:
    f = run / "resnet_mlp" / "results" / "validation_metrics.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    d = d[d.case.isin(["nmos_off", "pmos_off"])]
    if len(d) < 2:
        continue
    rows.append({"run": run.name, "settings": d.get("model_params", pd.Series(["default"])).iloc[0]
                 if "model_params" in d else "default",
                 "val_score": d.mean_median_error_pct.mean(),
                 "near_rail_id_is": d[["nr_id_mae_pct", "nr_is_mae_pct"]].to_numpy().mean(),
                 **{f"{c}_near_rail_id_is": d.loc[d.case == c, ["nr_id_mae_pct", "nr_is_mae_pct"]].to_numpy().mean()
                    for c in ("nmos_off", "pmos_off")},
                 **{f"{c}_val_score": d.loc[d.case == c, "mean_median_error_pct"].iloc[0] for c in ("nmos_off", "pmos_off")}})
t = pd.DataFrame(rows)
if "tune_A0" not in set(t.run):
    sys.exit("default run runs/tune_A0 missing")
limit = 1.05 * t.loc[t.run == "tune_A0", "val_score"].iloc[0]
best = t[t.val_score <= limit].sort_values("near_rail_id_is").iloc[0]
pd.set_option("display.width", 220)
print(t.round(4).to_string(index=False))
print(f"\nnormal validation score limit: {limit:.4f} %  ->  chosen: {best.run} ({best.settings})")
t.to_csv(RUNS_DIR / "resnet_tuning.csv", index=False)
(RUNS_DIR / "chosen_resnet.txt").write_text(best.run + "\n")
