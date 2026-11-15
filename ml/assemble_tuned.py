"""Build runs/final_tuned = runs/final with the resnet_mlp OFF models (nmos_off, pmos_off) replaced by the tuned run
chosen on validation (runs/chosen_resnet.txt, from choose_resnet.py). Everything else is copied unchanged, so
select_best.py --tag final_tuned compares the four model types exactly as before, with the tuned resnet for OFF."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from config import RUNS_DIR

chosen = (RUNS_DIR / "chosen_resnet.txt").read_text().strip()
src, dst = RUNS_DIR / "final", RUNS_DIR / "final_tuned"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
tuned, target = RUNS_DIR / chosen / "resnet_mlp", dst / "resnet_mlp"
for case in ("nmos_off", "pmos_off"):
    shutil.copyfile(tuned / "saved_models" / f"{case}.joblib", target / "saved_models" / f"{case}.joblib")
    shutil.copyfile(tuned / "results" / f"parity_{case}.png", target / "results" / f"parity_{case}.png")
for split in ("train", "validation", "test"):
    old = pd.read_csv(target / "results" / f"{split}_metrics.csv")
    new = pd.read_csv(tuned / "results" / f"{split}_metrics.csv")
    new = new[new.case.isin(["nmos_off", "pmos_off"])]
    pd.concat([old[~old.case.isin(new.case)], new], ignore_index=True).sort_values("case").to_csv(
        target / "results" / f"{split}_metrics.csv", index=False)
print(f"runs/final_tuned: resnet_mlp OFF models from runs/{chosen}; everything else from runs/final")
