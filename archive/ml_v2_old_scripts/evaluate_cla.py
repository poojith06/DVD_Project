"""Part C (steps 7-8): 3-bit CLA leakage = sum of the ML gate leakages, compared with the full SPICE netlist.

For each input vector the logic value at the inputs of every gate (33 gates: 27 NAND2, 3 INV, 2 AND2, 1 OR2,
from PartC_3bit CLA adder/netlists/cla3.sub) is worked out; the gate model of Part B gives that gate's leakage at
the same corner (results/gate_results/gate_predictions.csv - Part B and Part C share the 200 corners), and the
33 gate leakages are summed. As the spec says, this assumes every gate input sits at exactly 0 or VDD and ignores
loading between gates.

    python evaluate_cla.py            (after evaluate_gates.py)
Output: results/cla_results/ cla_predictions.csv (12,800 rows), cla_results.csv (64 vectors), cla_parity.png
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from config import CLA_DIR, REPO, RESULTS_DIR
from evaluate_gates import parity, results_table, summary_line

LOGIC = {"nand2": lambda a, b: 1 - (a & b), "inv": lambda a: 1 - a, "and2": lambda a, b: a & b,
         "or2": lambda a, b: a | b, "nor2": lambda a, b: 1 - (a | b)}


def cla_gates():
    """(type, output, inputs) for every gate call in cla3.sub (from git if the file is missing locally)."""
    path = CLA_DIR / "netlists" / "cla3.sub"
    text = path.read_text() if path.exists() else subprocess.run(
        ["git", "-C", str(REPO), "show", "HEAD:PartC_3bit CLA adder/netlists/cla3.sub"],
        capture_output=True, text=True, check=True).stdout
    gates = []
    for line in text.splitlines():
        t = line.split()
        if t and t[0].lower().startswith("x") and t[-1].lower() in LOGIC:
            gates.append((t[-1].lower(), t[1], t[2:-3]))       # ports: out, inputs..., vdd, gnd, TYPE
    return gates


def main():
    out = RESULTS_DIR / "cla_results"
    out.mkdir(parents=True, exist_ok=True)
    gates = cla_gates()
    cla = pd.read_csv(CLA_DIR / "data" / "cla3.csv")
    gp = pd.read_csv(RESULTS_DIR / "gate_results" / "gate_predictions.csv")
    gp["inp"] = gp.A.astype(int).astype(str) + gp.B.fillna(-1).astype(int).astype(str).replace("-1", "")
    ml = gp.set_index(["gate", "inp", "corner"])["leak_ml"].to_dict()
    sp = gp.set_index(["gate", "inp", "corner"])["leak_spice"].to_dict()

    vectors = {}
    for (A, B), d in cla.groupby(["A", "B"]):
        r = d.iloc[0]
        v = {f"a{i}": int(r[f"A{i}"]) for i in range(3)} | {f"b{i}": int(r[f"B{i}"]) for i in range(3)}
        cases = []
        for typ, o, ins in gates:
            bits = [v[x] for x in ins]
            v[o] = LOGIC[typ](*bits)
            cases.append((typ, "".join(map(str, bits))))
        for node, col in (("s0", "v_s0"), ("s1", "v_s1"), ("s2", "v_s2"), ("cout", "v_cout")):
            if v[node] != int(r[col] > r["vdd"] / 2):
                sys.exit(f"logic mismatch at A={A}, B={B}, {node}")
        vectors[(A, B)] = cases

    keys = list(zip(cla.A, cla.B, cla.corner))
    pred = cla[["corner", "A", "B", "vdd", "temp"]].copy()
    pred["vector"] = pred.A.astype(str) + "+" + pred.B.astype(str)
    pred["leak_spice"] = cla.leak
    pred["leak_ml"] = [sum(ml[(t, i, c)] for t, i in vectors[(A, B)]) for A, B, c in keys]
    pred["leak_gate_sum_spice"] = [sum(sp[(t, i, c)] for t, i in vectors[(A, B)]) for A, B, c in keys]
    pred["error_pct"] = 100 * (pred.leak_ml - pred.leak_spice) / pred.leak_spice
    pred["gate_sum_error_pct"] = 100 * (pred.leak_gate_sum_spice - pred.leak_spice) / pred.leak_spice
    pred.to_csv(out / "cla_predictions.csv", index=False)
    table = results_table(pred, ["vector"])
    table.to_csv(out / "cla_results.csv", index=False)
    pred["all"] = "CLA"
    parity(pred, "all", f"3-bit CLA leakage, {len(pred)} simulations (64 vectors x 200 corners)",
           out / "cla_parity.png")
    print(f"{len(gates)} gates; logic matches the SPICE outputs for all 64 vectors")
    print(f"ML (sum of ML gate leakages)     : {summary_line(pred.error_pct.to_numpy())}")
    print(f"for reference, SPICE gates summed: {summary_line(pred.gate_sum_error_pct.to_numpy())}")


if __name__ == "__main__":
    main()
