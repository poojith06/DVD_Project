"""EXP-11 (reporting only): gate and CLA accuracy when ONE model family is used for all four transistor cases.
These numbers were NOT used to choose the final models (the choice is made on the validation split only); they show how
much the family matters at gate level. Two sets of saved models are evaluated:
  80:10:10 (ml/runs/final):  gpr, hybrid_analytic_mlp, resnet_mlp (default settings), pinn
  70:15:15 (first model set):                gpr, hybrid_analytic_mlp, resnet_mlp, components_mlp
Uses the same scripts as the submitted results (evaluate_gates.py, ml/evaluate_cla.py). Evaluation only.
Output: results/analysis/report_numbers/family_gate_accuracy.csv
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "analysis" / "report_numbers"
PY = sys.executable
sets = []
for fam in ("gpr", "hybrid_analytic_mlp", "resnet_mlp", "pinn"):
    sets.append(("80:10:10 (ml)", fam, REPO / "ml" / "runs" / "final" / fam / "saved_models",
                 REPO / "ml" / "evaluate_gates.py"))
for fam in ("gpr", "hybrid_analytic_mlp", "resnet_mlp", "components_mlp"):
    sets.append(("70:15:15 (first model set)", fam, REPO / "archive" / "first_model_set" / fam / "saved_models", REPO / "archive" / "first_model_set" / "evaluate_gates.py"))
rows = []
tmp = Path(tempfile.mkdtemp(prefix="family_"))
for split, fam, mdir, script in sets:
    if not (mdir / "nmos_on.joblib").exists():
        print("missing", mdir)
        continue
    o = tmp / f"{split[:2]}_{fam}"
    subprocess.run([PY, str(script), "--models-dir", str(mdir), "--out", str(o / "gate")], check=True, capture_output=True)
    pred = pd.read_csv(o / "gate" / "gate_predictions.csv", dtype={"inputs": str})
    subprocess.run([PY, str(REPO / "ml" / "evaluate_cla.py"), "--predictions", str(o / "gate" / "gate_predictions.csv"),
                    "--out", str(o / "cla")], check=True, capture_output=True)
    s = pd.read_csv(o / "cla" / "summary.csv", index_col=0).value
    e = 100 * (pred.leak_ml - pred.leak_spice) / pred.leak_spice
    pred["err"] = e.abs()
    per_gate = (100 - pred.groupby("gate").err.mean()).to_dict()
    n00 = pred[(pred.gate == "nand2") & (pred.inputs == "00")]
    rows.append({"split": split, "family (all four cases)": fam, "partB_mean_acc_pct": 100 - pred.err.mean(),
                 "partB_median_acc_pct": 100 - pred.err.median(), "nand2_00_mean_acc_pct": 100 - n00.err.mean(),
                 **{f"partB_{g}_mean_acc_pct": per_gate[g] for g in ("inv", "nand2", "nor2", "and2", "or2")},
                 "partC_mean_acc_pct": float(s["partC_ml_acc_100_minus_mean_abs_pct"]),
                 "partC_median_acc_pct": float(s["partC_ml_acc_100_minus_median_abs_pct"])})
    print(rows[-1], flush=True)
t = pd.DataFrame(rows)
t.to_csv(OUT / "family_gate_accuracy.csv", index=False)
pd.set_option("display.width", 250)
print(t.round(2).to_string(index=False))
