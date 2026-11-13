"""Run the voltage-factor variant (model_components.ComponentModelVF) exactly like bench.run, without touching models.py.
Usage: python component_data/run_variant.py [case ...]   -> results/components_mlp_vf/<case>_metrics.csv (bench.score format)"""
import sys, time, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import bench, model_components as mc
from common.config import CASES

name = "components_mlp_vf"; out = bench.HERE / "results" / name; out.mkdir(parents=True, exist_ok=True)
for case in sys.argv[1:] or bench.ALL_CASES:
    t0 = time.time(); S, X, Y, bundle = bench.load(case)
    m = mc.ComponentModelVF(case, CASES[case][0]).fit({k: (X[k], Y[k], S[k]) for k in ("train", "validation")})
    t_fit = time.time() - t0
    logpred = {k: m.predict(X[k], S[k]) for k in S}; signed = {k: m.predict_signed(X[k], S[k]) for k in S}
    R = bench.score(case, S, bundle, logpred, name, signed); R["fit s"] = round(t_fit, 1)
    R.to_csv(out / f"{case}_metrics.csv", index=False); np.savez_compressed(out / f"{case}_logpred.npz", **logpred)
    (out / f"{case}_info.json").write_text(json.dumps(m.info, indent=1, default=str))
    v, t = R[R.split == "validation"], R[R.split == "test"]
    print(f"[{name}] {case:8s} fit {t_fit:5.0f}s | val median {' '.join(f'{x:.2f}' for x in v['median %'])}"
          f" | test median {' '.join(f'{x:.2f}' for x in t['median %'])} | test p95 {' '.join(f'{x:.1f}' for x in t['p95 %'])}", flush=True)
