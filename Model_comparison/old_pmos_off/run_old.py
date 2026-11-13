"""Benchmark on the OLD PMOS OFF dataset (ig/ib read from the voltage sources, as committed in 8147451).

Same harness, models, split (70/15/15 seed 42), features, target, sign method and metrics as Model_comparison/bench.py.
Only the data changes. Nothing outside this folder is written:
  Datasets/data/pmos_off.csv   old CSV (git show HEAD:Datasets/data/pmos_off.csv)
  final_bundle/                Final_models recipe trained on the old data (scaler, sign models, 3 MLPs) - used by
                               bench for the signs and as the mlp_final model
  component_data/              BSIM4 component currents for the old rows (replayed in ngspice) - components_mlp(_vf)
  results/<model>/pmos_off_*   per-model results, results/summary_*.csv
Usage: python run_old.py prep               # train the Final_models bundle + replay components in ngspice
       python run_old.py bench <model ...>  # run models (ensemble_top3 last, it reads the other results)
       python run_old.py summary
"""
import sys
from pathlib import Path
OLD = Path(__file__).resolve().parent; MC = OLD.parent; REPO = MC.parent
sys.path.insert(0, str(MC)); sys.path.insert(0, str(MC / "component_data"))
import bench, train_final as tf, models_research, model_components
from common.data import load_case, split

CASE = "pmos_off"
def old_splits(case):
    tr, va, te = split(load_case(case, OLD / "Datasets" / "data"), (0.70, 0.15, 0.15), 42)
    return {"train": tr, "validation": va, "test": te}

tf.splits = old_splits                  # data
tf.HERE = OLD / "final_bundle"          # Final_models bundle (models/, results/) for the old data
bench.HERE = OLD                        # results/<model>/...
models_research.HERE = OLD              # ensemble_top3 reads results/ here
model_components.REPO = OLD             # components_for reads OLD/Datasets/data
model_components.CDIR = OLD / "component_data"

def prep():
    tf.HERE.mkdir(exist_ok=True)
    tf.run(CASE)
    import make_replay as mr, subprocess, time
    mr.HERE, mr.REPO = OLD / "component_data", OLD
    n = mr.write_netlist(CASE)
    cir = mr.HERE / f"{CASE}_replay.cir"     # include the model card from the real Datasets/netlists
    cir.write_text(cir.read_text().replace(".include ../../Datasets/netlists/22nm_HP.pm",
                   ".include ../../../Datasets/netlists/22nm_HP.pm"), newline="\n")
    wsl = "/mnt/c/IIITH Academic/SEM 5/Digital VLSI Design/repo/Model_comparison/old_pmos_off/component_data"
    t0 = time.time()
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-c",
                    f"cd '{wsl}' && OMP_WAIT_POLICY=passive ngspice -b {CASE}_replay.cir > {CASE}_replay.log 2>&1"], check=True)
    df = mr.parse(CASE, n)
    print(f"replay: {n} rows in {time.time()-t0:.0f}s", flush=True)

def summary():
    src = (MC / "summarize.py").read_text().replace("HERE = Path(__file__).resolve().parent", f"HERE = Path(r'{OLD}')")
    exec(compile(src, "summarize.py", "exec"), {"__name__": "__main__"})

if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "prep": prep()
    elif cmd == "bench":
        for m in args:
            try: bench.run(m, [CASE])
            except Exception as e: print(f"[{m}] FAILED: {e!r}", flush=True)
    elif cmd == "summary": summary()
