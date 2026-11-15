"""Facts about the saved models for the report (read from the saved model files and run metrics; nothing is re-trained).
Output: results/analysis/report_numbers/model_facts.txt and training_times.csv"""
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
import numpy as np
import pandas as pd

from pipeline import TransistorSurrogate

F = REPO / "results"
lines = []
say = lambda s="": (print(s), lines.append(s))
say("Final models (results/models)")
for case in ("nmos_on", "nmos_off", "pmos_on", "pmos_off"):
    m = TransistorSurrogate.load(F / "models" / f"{case}.joblib")
    est = m.estimator
    name = type(est).__name__
    detail = ""
    if name == "GPRModel":
        detail = f"kernels per current (id, ig, is, ib): {est.kernels}; training rows {len(est.gps[0].Xt)}"
    elif name == "HybridAnalyticMLPModel":
        a = est.analytic
        detail = (f"analytical model with {len(a.names)} fitted constants; MLP groups {list(est.nets)}; "
                  f"hidden widths {[l.out_features for l in est.nets['id+is'].net if hasattr(l, 'out_features')]}")
    elif name == "ResNetMLPModel":
        n = est.nets["id+is"].net
        detail = f"width {n.inp.out_features}, residual blocks {len(n.blocks)}; groups {list(est.nets)}"
    elif name == "PINNModel":
        detail = f"chosen variants {est.chosen_variants}"
    nparams = ""
    if hasattr(est, "nets"):
        nparams = sum(p.numel() for g in est.nets.values() for p in g.net.parameters())
        detail += f"; network parameters {nparams}"
    say(f"  {case}: {name}; {detail}")
    say(f"      sign model: classifiers for {[k for k in m.sign_model.other]} (ig, ib), source classifier: {hasattr(m.sign_model, 'source_clf')}")

say("\nTraining time (seconds, validation_metrics.csv of each run) - runs/final (all 4 types, 80:10:10)")
rows = []
for fam in ("gpr", "hybrid_analytic_mlp", "resnet_mlp", "pinn"):
    d = pd.read_csv(REPO / "ml" / "runs" / "final" / fam / "results" / "validation_metrics.csv")
    for _, r in d.iterrows():
        rows.append({"model": fam, "case": r.case, "train_seconds": r.train_seconds})
for tag in ("rn_b5", "tune_A0"):
    d = pd.read_csv(REPO / "ml" / "runs" / tag / "resnet_mlp" / "results" / "validation_metrics.csv")
    for _, r in d.iterrows():
        rows.append({"model": f"resnet_mlp ({tag})", "case": r.case, "train_seconds": r.train_seconds})
t = pd.DataFrame(rows)
t.to_csv(F / "analysis" / "report_numbers" / "training_times.csv", index=False)
say(t.pivot(index="model", columns="case", values="train_seconds").round(0).to_string())
say("(times depend on what else was running on the GPU / CPU at the time; the runs shared the machine)")

say("\nSplit sizes (80:10:10, seed 42):")
for case in ("nmos_on", "nmos_off", "pmos_on", "pmos_off"):
    n = len(pd.read_csv(REPO / "partA_transistor" / "data" / f"{case}.csv"))
    say(f"  {case}: {n} rows -> train {int(round(0.8 * n))}, validation {int(round(0.1 * n))}, test {n - int(round(0.8 * n)) - int(round(0.1 * n))}")
(F / "analysis" / "report_numbers" / "model_facts.txt").write_text("\n".join(lines), encoding="utf-8")
