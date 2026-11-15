"""Part B (steps 4-5): gate leakage from the four chosen transistor models vs SPICE.

For every row of partB_gates/step3_spice_reference/data/<gate>.csv (5 gates, 18 input cases, 200 corners):
  1. each transistor's terminal voltages come from the recorded SPICE node voltages (inputs at 0 / VDD, internal
     node v_int, first-stage output v_n, output v_out; NMOS body 0, PMOS body VDD), widths as in the netlists
  2. ON / OFF from its gate voltage (NMOS ON if Vg > VDD/2, PMOS ON if Vg < VDD/2) -> chosen model of that case
     (partA_transistor_models/step2_training/saved_models/<case>.joblib) predicts id, ig, is, ib
  3. gate leakage = current drawn from VDD (as recorded by SPICE, -i(vsup)), from the predicted currents with
     Kirchhoff's current law: S = vdd + every node joined to it through ON PMOS; leakage = sum of the predicted
     currents into all terminals on S. A transistor with drain, source and body on S adds id + is + ib = -ig, so
     the channel current of ON transistors (~0 V across them) cancels.
Error = (ML - SPICE) / SPICE; bias = median signed error; mean accuracy = 100 % - mean |error|.

    python partB_gates/step4_ml_prediction/evaluate_gates.py [--models-dir DIR] [--out DIR]
Output (default pipeline_out/gate_results/; the submitted results are in partB_gates/step5_results/ and are never
overwritten by default): gate_predictions.csv (all rows: SPICE and ML leakage, SPICE output voltage, internal node
voltages from SPICE and ML), gate_results.csv (18 cases), gate_parity.png.
The ML-side internal node voltages (v_int_ml, v_n_ml) are copies of the recorded SPICE node voltages: the spec's step 4
feeds those recorded voltages to the transistor models.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
# shared modules (config, pipeline, models) and the pickled transistor models live in step2_training/code
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "partA_transistor_models" / "step2_training" / "code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import GATE_DATA, PIPELINE_OUT, SAVED_MODELS
from pipeline import TransistorSurrogate

# (name, type, drain, gate, source, body, W) from partB_gates/step3_spice_reference/netlists/<gate>_ngspice.cir.
# int = v_int (x in NAND2/AND2, y in NOR2/OR2), n = v_n (first-stage output of AND2/OR2).
GATES = {
    "inv": [("Mp", "pmos", "out", "a", "vdd", "vdd", 88e-9), ("Mn", "nmos", "out", "a", "0", "0", 44e-9)],
    "nand2": [("Mpa", "pmos", "out", "a", "vdd", "vdd", 88e-9), ("Mpb", "pmos", "out", "b", "vdd", "vdd", 88e-9),
              ("Mna", "nmos", "out", "a", "int", "0", 88e-9), ("Mnb", "nmos", "int", "b", "0", "0", 88e-9)],
    "nor2": [("Mpa", "pmos", "int", "a", "vdd", "vdd", 176e-9), ("Mpb", "pmos", "out", "b", "int", "vdd", 176e-9),
             ("Mna", "nmos", "out", "a", "0", "0", 44e-9), ("Mnb", "nmos", "out", "b", "0", "0", 44e-9)],
    "and2": [("Mpa", "pmos", "n", "a", "vdd", "vdd", 88e-9), ("Mpb", "pmos", "n", "b", "vdd", "vdd", 88e-9),
             ("Mna", "nmos", "n", "a", "int", "0", 88e-9), ("Mnb", "nmos", "int", "b", "0", "0", 88e-9),
             ("Mpi", "pmos", "out", "n", "vdd", "vdd", 88e-9), ("Mni", "nmos", "out", "n", "0", "0", 44e-9)],
    "or2": [("Mpa", "pmos", "int", "a", "vdd", "vdd", 176e-9), ("Mpb", "pmos", "n", "b", "int", "vdd", 176e-9),
            ("Mna", "nmos", "n", "a", "0", "0", 44e-9), ("Mnb", "nmos", "n", "b", "0", "0", 44e-9),
            ("Mpi", "pmos", "out", "n", "vdd", "vdd", 88e-9), ("Mni", "nmos", "out", "n", "0", "0", 44e-9)],
}
PROCESS = ["vth0", "toxe", "ndep", "u0", "xj"]


def load_models(models_dir=None):
    """Models from DIR/<case>.joblib (saved_models layout) or DIR/<case>/model.joblib (select_best.py layout)."""
    d = Path(models_dir) if models_dir else SAVED_MODELS

    def path(c):
        flat = d / f"{c}.joblib"
        return flat if flat.exists() else d / c / "model.joblib"
    return {c: TransistorSurrogate.load(path(c)) for c in ("nmos_on", "nmos_off", "pmos_on", "pmos_off")}


def node_voltages(df):
    vdd = df["vdd"].to_numpy()
    v = {"0": np.zeros(len(df)), "vdd": vdd, "a": df["A"].to_numpy() * vdd, "out": df["v_out"].to_numpy()}
    if "B" in df:
        v["b"] = df["B"].to_numpy() * vdd
    if "v_int" in df:
        v["int"] = df["v_int"].to_numpy()
    if "v_n" in df:
        v["n"] = df["v_n"].to_numpy()
    return v


def predict_gate(gate, df, models):
    """ML leakage drawn from VDD for every row of one gate's CSV (Kirchhoff method, see the docstring)."""
    v, vdd, n = node_voltages(df), df["vdd"].to_numpy(), len(df)
    devices = []
    for _, pol, d, g, s, b, w in GATES[gate]:
        p = "n_" if pol == "nmos" else "p_"
        x = pd.DataFrame({"vdd": df["vdd"], "temp": df["temp"], "w": w, "vd": v[d], "vs": v[s], "vg": v[g],
                          "vb": v[b], **{k: df[p + k] for k in PROCESS}})
        on = v[g] > vdd / 2 if pol == "nmos" else v[g] < vdd / 2
        cur = np.zeros((n, 4))                                  # id, ig, is, ib, positive into the terminal
        for state, mask in (("on", on), ("off", ~on)):
            if mask.any():
                cur[mask] = models[f"{pol}_{state}"].predict(x[mask]).to_numpy()
        devices.append((pol, (d, g, s, b), on, cur))
    in_s = {node: np.zeros(n, bool) for node in v}
    in_s["vdd"][:] = True
    for _ in devices:                                           # grow S from vdd through ON PMOS
        for pol, (d, g, s, b), on, _ in devices:
            if pol == "pmos":
                join = on & (in_s[d] | in_s[s])
                in_s[d] |= join
                in_s[s] |= join
    leak = np.zeros(n)
    for _, (d, g, s, b), _, cur in devices:
        inside = in_s[d] & in_s[s] & in_s[b]
        leak += np.where(inside & ~in_s[g], -cur[:, 1], 0.0)
        for k, node in enumerate((d, g, s, b)):
            leak += np.where(~inside & in_s[node], cur[:, k], 0.0)
    return leak


def results_table(pred, group_cols):
    rows = []
    for key, g in pred.groupby(group_cols, sort=False):
        e = g["error_pct"]
        rows.append({**dict(zip(group_cols, key if isinstance(key, tuple) else (key,))), "corners": len(g),
                     "spice_median_A": g["leak_spice"].median(), "ml_median_A": g["leak_ml"].median(),
                     "bias_pct": e.median(), "median_error_pct": e.abs().median(),
                     "p95_error_pct": e.abs().quantile(0.95), "mean_error_pct": e.abs().mean(),
                     "max_error_pct": e.abs().max(), "mean_accuracy_pct": 100 - e.abs().mean()})
    return pd.DataFrame(rows)


def parity(pred, label_col, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for lab, g in pred.groupby(label_col, sort=False):
        ax.loglog(g["leak_spice"], g["leak_ml"], ".", markersize=3, alpha=0.6, label=str(lab).upper())
    lo = min(pred.leak_spice.min(), pred.leak_ml.min()) / 1.5
    hi = max(pred.leak_spice.max(), pred.leak_ml.max()) * 1.5
    ax.loglog([lo, hi], [lo, hi], "k--", linewidth=1, label="ML = SPICE")
    ax.set(xlabel="SPICE leakage (A)", ylabel="ML leakage (A)", title=title)
    ax.legend(markerscale=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_line(e):
    e = np.abs(e)
    return (f"mean accuracy {100 - e.mean():.2f} %, median accuracy {100 - np.median(e):.2f} %, "
            f"p95 error {np.percentile(e, 95):.2f} %, max {e.max():.2f} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models_dir", nargs="?",
                    help="folder with <case>.joblib or <case>/model.joblib (default: step2_training/saved_models)")
    ap.add_argument("--models-dir", dest="models_dir_opt", help="same as the positional argument")
    ap.add_argument("--out", type=Path, help="output folder (default: pipeline_out/gate_results)")
    args = ap.parse_args()
    out = args.out or (PIPELINE_OUT / "gate_results")
    out.mkdir(parents=True, exist_ok=True)
    models = load_models(args.models_dir_opt or args.models_dir)
    parts = []
    for gate in GATES:
        df = pd.read_csv(GATE_DATA / f"{gate}.csv")
        p = df.rename(columns={"leak": "leak_spice"})
        p.insert(0, "gate", gate)
        p.insert(1, "inputs", df["A"].astype(int).astype(str) + (df["B"].astype(int).astype(str) if "B" in df else ""))
        p["leak_ml"] = predict_gate(gate, df, models)
        p["error_pct"] = 100 * (p.leak_ml - p.leak_spice) / p.leak_spice
        # Spec Section 5: "the internal node voltages from both". v_int / v_n / v_out are the recorded SPICE values.
        # Following step 4 of the spec, the ML side uses these recorded SPICE node voltages as the terminal voltages of
        # the transistors (it does not solve for them), so the ML internal-node columns are copies of the SPICE ones.
        for node in ("v_int", "v_n"):
            if node in p:
                p[f"{node}_ml"] = p[node]
        parts.append(p)
    pred = pd.concat(parts, ignore_index=True)
    # name of the internal node recorded in v_int (spec Section 5: x between the series NMOS, y between the series PMOS)
    pred["int_node"] = pred["gate"].map({"nand2": "x", "and2": "x", "nor2": "y", "or2": "y"}).fillna("")
    process = [c for c in pred.columns if c.startswith(("n_", "p_")) and c not in ("n_node",)]
    order = (["gate", "inputs", "A"] + (["B"] if "B" in pred else []) + ["corner", "vdd", "temp"] + process
             + ["leak_spice", "leak_ml", "error_pct", "v_out", "int_node"]
             + [c for c in ("v_int", "v_int_ml", "v_n", "v_n_ml") if c in pred])
    pred = pred[order + [c for c in pred.columns if c not in order]]
    pred.to_csv(out / "gate_predictions.csv", index=False)
    table = results_table(pred, ["gate", "inputs"])
    table.to_csv(out / "gate_results.csv", index=False)
    parity(pred, "gate", f"Gate leakage, {len(pred)} simulations (18 cases x 200 corners)", out / "gate_parity.png")
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nper gate:", {g: round(100 - d.error_pct.abs().mean(), 2) for g, d in pred.groupby("gate", sort=False)})
    print(f"ALL {len(pred)} gate simulations: {summary_line(pred.error_pct.to_numpy())}")


if __name__ == "__main__":
    main()
