"""Seed check for the 5 strongest single models: components_mlp_vf, hybrid_analytic_mlp, resnet_mlp, gpr, pinn.

Same harness as Model_comparison/bench.py (data = Datasets/data incl. the new PMOS OFF, split 70/15/15 seed 42, features,
log10 target, Final_models sign method, metrics). Only the models' random seeds change; the data split is fixed.
  seed 0 = the main benchmark run (Model_comparison/results, copied here by `python run_seeds.py copy0`)
  seed k = every seed the models set is shifted by 1000*k: torch.manual_seed (network init, batch order),
           torch.Generator().manual_seed (PINN init), np.random.default_rng (GPR hyper-parameter subset)
Output: seed<k>/results/<model>/<case>_metrics.csv (+ logpred), summary_*.csv from `python run_seeds.py summary`.
Nothing outside this folder is written (component-model weights go to seed<k>/component_models/).
Usage: python run_seeds.py copy0
       python run_seeds.py run <k> <model ...> [--cases nmos_on ...]
       python run_seeds.py summary
"""
import sys, shutil
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; MC = HERE.parent
sys.path.insert(0, str(MC))
MODELS = ["components_mlp_vf", "hybrid_analytic_mlp", "resnet_mlp", "gpr", "pinn"]
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]

def patch_seeds(k):
    import torch
    off = 1000 * k
    _ms = torch.manual_seed; torch.manual_seed = lambda s: _ms(int(s) + off)
    class Gen(torch.Generator):
        def manual_seed(self, s): return super().manual_seed(int(s) + off)
    torch.Generator = Gen
    _rng = np.random.default_rng
    np.random.default_rng = lambda s=None: _rng(int(s) + off if isinstance(s, (int, np.integer)) else s)   # ints only

def run(k, models, cases):
    patch_seeds(k)
    import bench, model_components
    out = HERE / f"seed{k}"; bench.HERE = out
    _save = model_components.ComponentModel.save
    def save(self, path):
        (out / "component_models").mkdir(parents=True, exist_ok=True); _save(self, out / "component_models" / Path(path).name)
    model_components.ComponentModel.save = save
    for m in models:
        try: bench.run(m, cases)
        except Exception as e: print(f"[{m}] FAILED: {e!r}", flush=True)

def copy0():
    for m in MODELS:
        dst = HERE / "seed0" / "results" / m; dst.mkdir(parents=True, exist_ok=True)
        for f in (MC / "results" / m).glob("*_*"): shutil.copy2(f, dst / f.name)

def summary():
    import pandas as pd
    L = pd.concat([pd.read_csv(f).assign(seed=int(f.parts[-4][4:]))
                   for f in sorted(HERE.glob("seed*/results/*/*_metrics.csv"))], ignore_index=True)
    L.to_csv(HERE / "summary_long.csv", index=False)
    g = L.groupby(["case", "model", "split", "current"])
    S = g[["median %", "p95 %", "MAE %", "R2 (log)", "sign acc %"]].agg(["mean", "std", "min", "max"])
    S["n seeds"] = g.size(); S.to_csv(HERE / "summary_mean_std.csv")
    print(S.xs("test", level="split")[[("median %", "mean"), ("median %", "std"), ("p95 %", "mean"), ("MAE %", "mean"), ("n seeds", "")]].round(3).to_string())

if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "copy0": copy0()
    elif cmd == "run":
        cases = CASES
        if "--cases" in a: i = a.index("--cases"); cases = a[i + 1:]; a = a[:i]
        run(int(a[0]), a[1:] or MODELS, cases)
    elif cmd == "summary": summary()
