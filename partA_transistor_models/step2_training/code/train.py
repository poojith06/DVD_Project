"""Train model types on the four cases (80:10:10 split) and write train / validation / test metrics.

    python train.py                                         # all 4 model types, all 4 cases
    python train.py --models resnet_mlp --cases nmos_off pmos_off --rail-weight 10 --tag tune_A10
    python train.py --rail-weight 10 --tag final

Output: runs/<tag>/<model>/saved_models/<case>.joblib
        runs/<tag>/<model>/results/{train,validation,test}_metrics.csv, parity_<case>.png
Validation metrics also hold the "near-rail" check (nr_* columns): the validation rows with both vd and vs within
NEAR_RAIL_V (together) of a rail, i.e. the operating points that logic gates use.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (CASES, CURRENT_FLOOR, DATA_DIR, MODEL_SEED, MODEL_TYPES, NEAR_RAIL_V, RUNS_DIR, SEED,
                    SPLIT, TARGET_COLUMNS)
from models import MODELS
from pipeline import (SignModel, TransistorSurrogate, fit_scaler, kirchhoff_residual, load_case, log_magnitude,
                      rail_distance, rail_weights, split, summarise)


def parity_plot(true, pred, title, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for k, (ax, name) in enumerate(zip(axes, TARGET_COLUMNS)):
        x, y = np.maximum(np.abs(true[:, k]), CURRENT_FLOOR), np.maximum(np.abs(pred[:, k]), CURRENT_FLOOR)
        ax.loglog(x, y, ".", markersize=2, alpha=0.5)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.loglog(lim, lim, "k--", linewidth=1)
        ax.set(title=name, xlabel="SPICE |I| (A)", ylabel="Predicted |I| (A)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_case(model_key, case, args, out_dir):
    df = load_case(case, DATA_DIR)
    res = kirchhoff_residual(df)
    tr, va, te = split(df, SPLIT, SEED)
    print(f"\n=== {model_key} / {case}: {len(tr)} train, {len(va)} validation, {len(te)} test rows "
          f"(KCL residual median {res.median():.1e}); rail weight {args.rail_weight:g}", flush=True)
    sur = TransistorSurrogate(case, fit_scaler(tr, case))
    X = {k: sur.transform(d) for k, d in (("train", tr), ("validation", va), ("test", te))}
    Y = {k: log_magnitude(d[TARGET_COLUMNS].to_numpy()) for k, d in (("train", tr), ("validation", va), ("test", te))}
    w = rail_weights(tr, args.rail_weight) if args.rail_weight > 0 else None
    params = dict(args.model_params)
    if args.epochs:
        params["epochs"] = args.epochs
    params = params or None
    est = MODELS[model_key](case, seed=args.model_seed, params=params)
    t0 = time.perf_counter()
    est.fit(X["train"], Y["train"], X["validation"], Y["validation"], tr, va, w_train=w)
    sur.estimator = est
    sur.sign_model = SignModel(case).fit(X["train"], tr, tr[TARGET_COLUMNS].to_numpy(), est.predict(X["train"], tr))
    seconds = time.perf_counter() - t0

    rows, preds = {}, {}
    for k, d in (("train", tr), ("validation", va), ("test", te)):
        preds[k] = sur.predict_array(d)
        rows[k] = {"case": case, "model": model_key, "split": k, "rail_weight": args.rail_weight,
                   "model_params": args.model_params_text or "default",
                   **summarise(d[TARGET_COLUMNS].to_numpy(), preds[k])}
    near = rail_distance(va) < NEAR_RAIL_V
    nr = summarise(va[TARGET_COLUMNS].to_numpy()[near], preds["validation"][near])
    rows["validation"].update({"nr_rows": int(near.sum()), **{f"nr_{k}": v for k, v in nr.items()}})
    rows["validation"]["train_seconds"] = round(seconds, 1)

    v = rows["validation"]
    print(f"  done in {seconds:.0f}s   validation median % id/ig/is/ib: "
          + " / ".join(f"{v[f'{c}_median_error_pct']:.3f}" for c in TARGET_COLUMNS)
          + f"   near-rail ({near.sum()} rows) mean % id/ig/is/ib: "
          + " / ".join(f"{v[f'nr_{c}_mae_pct']:.3f}" for c in TARGET_COLUMNS), flush=True)
    t = rows["test"]
    print("  test median % id/ig/is/ib: " + " / ".join(f"{t[f'{c}_median_error_pct']:.3f}" for c in TARGET_COLUMNS),
          flush=True)
    sur.save(out_dir / "saved_models" / f"{case}.joblib")
    parity_plot(te[TARGET_COLUMNS].to_numpy(), preds["test"], f"{case}: {model_key}, test set",
                out_dir / "results" / f"parity_{case}.png")
    return rows


def merge(path, new_rows):
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["case"].isin(new["case"])], new], ignore_index=True)
    new.sort_values("case").to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_TYPES, default=MODEL_TYPES)
    p.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    p.add_argument("--rail-weight", type=float, default=0.0, help="near-rail training weight strength (0 = off)")
    p.add_argument("--tag", default="final", help="run name: outputs go to runs/<tag>/")
    p.add_argument("--model-seed", type=int, default=MODEL_SEED)
    p.add_argument("--epochs", type=int, help="override network epochs (quick tests only)")
    p.add_argument("--force", action="store_true", help="retrain cases already claimed in this run tag")
    p.add_argument("--model-params", dest="model_params_text", default="",
                   help="model settings as key=value pairs, e.g. width=512,blocks=5,lr=5e-4,batch_size=64")
    args = p.parse_args()
    num = lambda v: int(v) if v.lstrip("-").isdigit() else float(v)
    args.model_params = {k.strip(): num(v.strip()) for k, v in
                         (kv.split("=") for kv in args.model_params_text.split(",") if kv.strip())}

    for model_key in args.models:
        out_dir = RUNS_DIR / args.tag / model_key
        (out_dir / "results").mkdir(parents=True, exist_ok=True)
        for case in args.cases:
            # claim the case so a second run of the same tag (e.g. one started on the CPU in parallel) skips it
            claim = out_dir / f".claim_{case}"
            try:
                if args.force and claim.exists():
                    claim.unlink()
                claim.open("x").close()
            except FileExistsError:
                print(f"{model_key} / {case}: already trained or running in runs/{args.tag} - skipped", flush=True)
                continue
            rows = train_case(model_key, case, args, out_dir)
            for k in ("train", "validation", "test"):   # written after every case: a crash keeps finished cases
                merge(out_dir / "results" / f"{k}_metrics.csv", [rows[k]])
        print(f"\n{model_key}: saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
