"""Part B steps 4-5: predict gate leakage with the transistor models and compare with SPICE.

For every row of Gates_Part-B/data/<gate>.csv (gate, input combination, corner), the recorded SPICE node
voltages (inputs, internal nodes, output) give each transistor's terminal voltages. The transistor is
classified ON or OFF from its gate voltage and the matching chosen model (results/<case>/model.joblib)
predicts its four terminal currents. The gate leakage is the current drawn from VDD, the quantity the SPICE
netlists record (leak = -i(vsup)), built from the predicted currents with Kirchhoff's current law
(see predict_gate). Error = (ML - SPICE) / SPICE; bias = median signed error; mean accuracy = 100 - mean |error|.

    python evaluate_gates.py                     # chosen models -> gate_results/
    python evaluate_gates.py --models-dir gpr/saved_models --out gate_results_gpr

Outputs: gate_predictions.csv (every row: SPICE and ML leakage, node voltages), gate_results.csv (the 18-row
table of deliverable 6) and gate_parity.png (deliverable 7).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # silences a harmless joblib warning on Windows
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import ROOT
from predict import load_surrogate

GATE_DATA = ROOT.parent / "Gates_Part-B" / "data"
L = 22e-9

# Transistor netlists copied from Gates_Part-B/netlists/<gate>_ngspice.cir:
# (name, type, drain, gate, source, body, width). Node names are CSV columns or the rails 0 / vdd.
# The CSV calls the internal node v_int (x for NAND2/AND2, y for NOR2/OR2) and the first-stage
# output of AND2/OR2 v_n.
GATES = {
    "inv": [
        ("Mp", "pmos", "out", "a", "vdd", "vdd", 88e-9),
        ("Mn", "nmos", "out", "a", "0", "0", 44e-9),
    ],
    "nand2": [
        ("Mpa", "pmos", "out", "a", "vdd", "vdd", 88e-9),
        ("Mpb", "pmos", "out", "b", "vdd", "vdd", 88e-9),
        ("Mna", "nmos", "out", "a", "int", "0", 88e-9),
        ("Mnb", "nmos", "int", "b", "0", "0", 88e-9),
    ],
    "nor2": [
        ("Mpa", "pmos", "int", "a", "vdd", "vdd", 176e-9),
        ("Mpb", "pmos", "out", "b", "int", "vdd", 176e-9),
        ("Mna", "nmos", "out", "a", "0", "0", 44e-9),
        ("Mnb", "nmos", "out", "b", "0", "0", 44e-9),
    ],
    "and2": [
        ("Mpa", "pmos", "n", "a", "vdd", "vdd", 88e-9),
        ("Mpb", "pmos", "n", "b", "vdd", "vdd", 88e-9),
        ("Mna", "nmos", "n", "a", "int", "0", 88e-9),
        ("Mnb", "nmos", "int", "b", "0", "0", 88e-9),
        ("Mpi", "pmos", "out", "n", "vdd", "vdd", 88e-9),
        ("Mni", "nmos", "out", "n", "0", "0", 44e-9),
    ],
    "or2": [
        ("Mpa", "pmos", "int", "a", "vdd", "vdd", 176e-9),
        ("Mpb", "pmos", "n", "b", "int", "vdd", 176e-9),
        ("Mna", "nmos", "n", "a", "0", "0", 44e-9),
        ("Mnb", "nmos", "n", "b", "0", "0", 44e-9),
        ("Mpi", "pmos", "out", "n", "vdd", "vdd", 88e-9),
        ("Mni", "nmos", "out", "n", "0", "0", 44e-9),
    ],
}
PROCESS = ["vth0", "toxe", "ndep", "u0", "xj"]
TERMINALS = {"drain": "id", "gate": "ig", "source": "is", "body": "ib"}


def node_voltages(df: pd.DataFrame) -> dict:
    """Map every node name used in GATES to its voltage per row (SPICE values)."""
    v = {"0": np.zeros(len(df)), "vdd": df["vdd"].to_numpy(),
         "a": df["A"].to_numpy() * df["vdd"].to_numpy(), "out": df["v_out"].to_numpy()}
    if "B" in df:
        v["b"] = df["B"].to_numpy() * df["vdd"].to_numpy()
    if "v_int" in df:
        v["int"] = df["v_int"].to_numpy()
    if "v_n" in df:
        v["n"] = df["v_n"].to_numpy()
    return v


def predict_gate(gate: str, df: pd.DataFrame, surrogates: dict) -> tuple[np.ndarray, np.ndarray]:
    """ML supply leakage for every row of one gate's CSV, by two methods.

    supply_sum: sum of the predicted currents into every terminal tied to the vdd node (the literal reading).
    kcl:        Kirchhoff's current law on the "supply super-node" S = vdd plus every node joined to it
                through ON PMOS. The current drawn from VDD equals the sum of the currents into all terminals
                at nodes of S. A transistor with drain, source and body all in S (an ON PMOS in the pull-up
                network) adds id + is + ib = -ig, so its channel current, which has ~0 V across it and
                cannot be predicted reliably, cancels out. Only OFF-transistor currents leaving S and gate
                currents remain.
    """
    v = node_voltages(df)
    vdd = df["vdd"].to_numpy()
    n = len(df)
    devices = []
    for _, polarity, d, g, s, b, w in GATES[gate]:
        p = "n_" if polarity == "nmos" else "p_"
        x = pd.DataFrame({"vdd": df["vdd"], "temp": df["temp"], "w": w,
                          "vd": v[d], "vs": v[s], "vg": v[g], "vb": v[b],
                          **{k: df[p + k] for k in PROCESS}})
        on = v[g] > vdd / 2 if polarity == "nmos" else v[g] < vdd / 2
        currents = np.zeros((n, 4))  # id, ig, is, ib, positive into the terminal
        for state, mask in (("on", on), ("off", ~on)):
            if mask.any():
                currents[mask] = surrogates[f"{polarity}_{state}"].predict(x[mask]).to_numpy()
        devices.append((polarity, (d, g, s, b), on, currents))

    supply_sum = np.zeros(n)
    for _, nodes, _, currents in devices:
        for k, node in enumerate(nodes):
            if node == "vdd":
                supply_sum += currents[:, k]

    # Super-node S per row: grow from vdd through ON PMOS (drain <-> source).
    in_s = {node: np.zeros(n, bool) for node in v}
    in_s["vdd"][:] = True
    for _ in range(len(devices)):
        for polarity, (d, g, s, b), on, _ in devices:
            if polarity == "pmos":
                join = on & (in_s[d] | in_s[s])
                in_s[d] |= join
                in_s[s] |= join
    kcl = np.zeros(n)
    for _, nodes, _, currents in devices:
        d, g, s, b = nodes
        inside = in_s[d] & in_s[s] & in_s[b]
        # transistor entirely inside S: id + is + ib = -ig (zero if its gate node is in S as well)
        kcl += np.where(inside & ~in_s[g], -currents[:, 1], 0.0)
        # any other transistor: add its currents into the terminals that sit on S
        for k, node in enumerate(nodes):
            kcl += np.where(~inside & in_s[node], currents[:, k], 0.0)
    return kcl, supply_sum


def input_label(row) -> str:
    return f"{int(row['A'])}" + (f"{int(row['B'])}" if "B" in row and not pd.isna(row["B"]) else "")


def results_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (gate, inputs), g in pred.groupby(["gate", "inputs"], sort=False):
        signed = 100 * (g["leak_ml"] - g["leak_spice"]) / g["leak_spice"]
        err = signed.abs()
        rows.append({"gate": gate, "inputs": inputs, "corners": len(g),
                     "spice_median_A": g["leak_spice"].median(), "ml_median_A": g["leak_ml"].median(),
                     "bias_pct": signed.median(), "mean_error_pct": err.mean(),
                     "median_error_pct": err.median(), "p95_error_pct": err.quantile(0.95),
                     "max_error_pct": err.max()})
    return pd.DataFrame(rows)


def parity_plot(pred: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for gate, g in pred.groupby("gate", sort=False):
        ax.loglog(g["leak_spice"], g["leak_ml"], ".", markersize=3, alpha=0.6, label=gate.upper())
    lo = min(pred["leak_spice"].min(), pred["leak_ml"].min()) / 1.5
    hi = max(pred["leak_spice"].max(), pred["leak_ml"].max()) * 1.5
    ax.loglog([lo, hi], [lo, hi], "k--", linewidth=1, label="ML = SPICE")
    ax.set_xlabel("SPICE leakage (A)")
    ax.set_ylabel("ML leakage (A)")
    ax.set_title(f"Gate leakage, {len(pred)} gate simulations (18 cases x 200 corners)")
    ax.legend(markerscale=4)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, help="use <dir>/<case>.joblib instead of the chosen models")
    parser.add_argument("--data-dir", type=Path, default=GATE_DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "gate_results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    surrogates = {c: load_surrogate(c, args.models_dir)
                  for c in ("nmos_on", "nmos_off", "pmos_on", "pmos_off")}
    parts = []
    for gate in GATES:
        df = pd.read_csv(args.data_dir / f"{gate}.csv")
        out = df.copy()
        out.insert(0, "gate", gate)
        out.insert(1, "inputs", df.apply(input_label, axis=1))
        out = out.rename(columns={"leak": "leak_spice"})
        out["leak_ml"], out["leak_ml_supply_sum"] = predict_gate(gate, df, surrogates)
        out["error_pct"] = 100 * (out["leak_ml"] - out["leak_spice"]) / out["leak_spice"]
        out["error_pct_supply_sum"] = 100 * (out["leak_ml_supply_sum"] - out["leak_spice"]) / out["leak_spice"]
        parts.append(out)
    pred = pd.concat(parts, ignore_index=True)
    pred.to_csv(args.out / "gate_predictions.csv", index=False)

    table = results_table(pred)
    table.to_csv(args.out / "gate_results.csv", index=False)
    parity_plot(pred, args.out / "gate_parity.png")

    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    for label, col in (("KCL super-node (used for the table and plot)", "error_pct"),
                       ("plain supply-terminal sum (for comparison)", "error_pct_supply_sum")):
        err = pred[col].abs()
        print(f"\n{label}, all {len(pred)} gate simulations: mean |error| {err.mean():.2f} % "
              f"(mean accuracy {100 - err.mean():.2f} %), median {err.median():.2f} %, p95 {err.quantile(0.95):.2f} %")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
