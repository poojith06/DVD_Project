"""EXP-18: does `set rndseed=16` give the same 200 corners on the ngspice available here (WSL) as in the saved data
(generated with ngspice-44.2 on Windows)? Runs the unchanged Part B INV netlist in a scratch folder and compares its
corner columns with the saved partB_gates/data/inv.csv. Nothing in the repo is modified."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simlib import REPO, run_folder

SIM = HERE / "sim" / "corners"
if SIM.exists():
    shutil.rmtree(SIM)
(SIM / "netlists").mkdir(parents=True)
(SIM / "data").mkdir()
shutil.copy(REPO / "model" / "22nm_HP.pm", SIM / "netlists" / "22nm_HP.pm")
shutil.copy(REPO / "partB_gates" / "netlists" / "inv_ngspice.cir", SIM / "netlists" / "inv_ngspice.cir")
ver = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-c", "ngspice --version 2>&1 | head -3"], capture_output=True, text=True).stdout
print("ngspice available here:", " ".join(ver.split())[:80])
run_folder(SIM / "netlists")
new = pd.read_csv(SIM / "data" / "inv.csv")
old = pd.read_csv(REPO / "partB_gates" / "data" / "inv.csv")
cols = ["vdd", "temp", "n_vth0", "n_toxe", "n_ndep", "n_u0", "n_xj", "p_vth0", "p_toxe", "p_ndep", "p_u0", "p_xj"]
a = new.groupby("corner")[cols].first()
b = old.groupby("corner")[cols].first()
rel = ((a - b).abs() / b.abs()).max(axis=1)
print(f"corners regenerated: {len(a)}, saved: {len(b)}")
print(f"corners identical to the saved ones (relative difference < 1e-5 in every column): {int((rel < 1e-5).sum())} of {len(b)}")
print(f"median relative difference over the corners: {rel.median():.3f}; first corner saved vdd {b.vdd.iloc[0]}, regenerated {a.vdd.iloc[0]}")
