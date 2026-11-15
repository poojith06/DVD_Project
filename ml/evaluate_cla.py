"""Part C (spec steps 7-8, deliverable 8): 3-bit CLA leakage = plain sum of the Part B ML gate leakages.

1. Gate list from the CLA netlist (cla3_ngspice.cir top level + cla3.sub): instance, type, input nets in pin order,
   output net. Each gate subcircuit uses Part B's pin convention (first input = A, second = B; same transistor
   connections as partB_gates/netlists), so "NAND2 inputs 01" is the same physical case as in Part B.
2. Gates sorted topologically; every net's logic value computed per input vector; outputs checked against the
   correct 3-bit sum and against the SPICE output voltages in cla3.csv (threshold VDD/2). Any mismatch stops.
3. For each (vector, corner) every gate's ML leakage is looked up in the Part B gate predictions, keyed by
   (gate type, input bits, corner) - Part B and Part C share the same 200 corners. A missing key stops the script.
4. leak_ml = sum of the gate leakages. leak_ceiling = the same sum of the Part B SPICE gate leakages (perfect gate
   models; error analysis only - it shows what the summing itself costs: loading, non-rail inputs, pin currents).
5. error = (leak - leak_spice) / leak_spice, compared with cla3.csv.

    python evaluate_cla.py --predictions gate_results/gate_predictions.csv --out ../results/phase2_comparison/set_b

Outputs in --out: cla3_gates.csv, cla3_predictions.csv, cla3_results.csv (overall + per vector), cla3_parity.png,
summary.csv; the accuracy under four definitions and an error breakdown are printed.
"""
import argparse
import os
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CLA_DIR = Path(os.environ.get("CLA_DIR", REPO / "partC_cla"))   # folder with netlists/ and data/cla3.csv
LOGIC = {"inv": lambda a: 1 - a, "nand2": lambda a, b: 1 - (a & b), "nor2": lambda a, b: 1 - (a | b),
         "and2": lambda a, b: a & b, "or2": lambda a, b: a | b}
PRIMARY = ["a0", "a1", "a2", "b0", "b1", "b2"]
OUTPUTS = {"s0": "v_s0", "s1": "v_s1", "s2": "v_s2", "cout": "v_cout"}


# ----------------------------------------------------------------------------- 1. gate list
def read_netlist_text(rel):
    path = CLA_DIR / "netlists" / rel
    if path.exists():
        return path.read_text()
    # tracked in git but missing in the working tree (restore with git checkout); read the committed copy
    out = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:partC_cla/netlists/{rel}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"cannot find netlists/{rel} locally or in git")
    print(f"note: netlists/{rel} is missing locally - read from git HEAD")
    return out.stdout


def gate_list():
    top = read_netlist_text("cla3_ngspice.cir")
    sub = read_netlist_text("cla3.sub")
    inst = re.search(r"(?im)^\s*x\w+\s+(.*?)\s+CLA3\s*$", top).group(1).split()
    ports = re.search(r"(?im)^\s*\.subckt\s+CLA3\s+(.*)$", sub).group(1).split()
    rename = dict(zip(ports, inst))                  # subckt port -> top-level net (identity here, checked)
    rows = []
    for line in sub.splitlines():
        t = line.split()
        if not t or not t[0].lower().startswith("x"):
            continue
        typ = t[-1].lower()
        if typ not in LOGIC:
            sys.exit(f"unknown gate type in cla3.sub: {line}")
        nets = [rename.get(n, n) for n in t[1:-1]]
        out, ins = nets[0], nets[1:-2]               # ports: out, inputs..., vdd, gnd
        rows.append({"instance": t[0], "type": typ, "inputs": " ".join(ins), "pin_A": ins[0],
                     "pin_B": ins[1] if len(ins) > 1 else "", "output": out})
    return pd.DataFrame(rows)


def topological(gates):
    driver = {r.output: i for i, r in gates.iterrows()}
    deps = {i: [driver[n] for n in r.inputs.split() if n in driver] for i, r in gates.iterrows()}
    indeg = {i: len(d) for i, d in deps.items()}
    users = {i: [] for i in deps}
    for i, d in deps.items():
        for j in d:
            users[j].append(i)
    q, order = deque(i for i, k in indeg.items() if k == 0), []
    while q:
        i = q.popleft()
        order.append(i)
        for u in users[i]:
            indeg[u] -= 1
            if indeg[u] == 0:
                q.append(u)
    if len(order) != len(gates):
        sys.exit("combinational loop in the CLA netlist")
    return gates.loc[order].reset_index(drop=True)


# ----------------------------------------------------------------------------- metrics
def metrics(spice, est):
    e = 100 * (est - spice) / spice
    return {"spice_median_A": float(np.median(spice)), "est_median_A": float(np.median(est)),
            "bias_pct": float(np.median(e)), "median_abs_error_pct": float(np.median(np.abs(e))),
            "p95_abs_error_pct": float(np.percentile(np.abs(e), 95)), "mean_abs_error_pct": float(np.mean(np.abs(e))),
            "max_abs_error_pct": float(np.max(np.abs(e)))}


def accuracy_definitions(df, col, group="vector"):
    e = 100 * (df[col] - df.leak_spice) / df.leak_spice
    g = df.assign(e=e).groupby(group)
    ratio = (g[col].median() / g.leak_spice.median() - 1).abs() * 100
    return {"acc_100_minus_mean_abs_pct": 100 - e.abs().mean(),
            "acc_100_minus_median_abs_pct": 100 - e.abs().median(),
            "acc_100_minus_abs_mean_signed_pct": 100 - abs(e.mean()),
            "acc_mean_over_groups_median_ratio_pct": 100 - ratio.mean()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True,
                   help="Part B gate predictions CSV (gate, A, B, corner, leak_spice, leak_ml), from evaluate_gates.py")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. gates
    gates = topological(gate_list())
    gates.to_csv(args.out / "cla3_gates.csv", index=False)
    counts = gates.type.value_counts().to_dict()
    print(f"1. {len(gates)} gates: {counts}")
    if counts != {"nand2": 27, "inv": 3, "and2": 2, "or2": 1}:
        sys.exit("unexpected gate counts")

    # 2. logic per vector
    cla = pd.read_csv(CLA_DIR / "data" / "cla3.csv")
    vec_cases = {}
    for (A, B), d in cla.groupby(["A", "B"]):
        r = d.iloc[0]
        v = {f"a{i}": int(r[f"A{i}"]) for i in range(3)} | {f"b{i}": int(r[f"B{i}"]) for i in range(3)}
        if 4 * v["a2"] + 2 * v["a1"] + v["a0"] != A or 4 * v["b2"] + 2 * v["b1"] + v["b0"] != B:
            sys.exit(f"vector bits do not match A={A}, B={B}")
        cases = []
        for g in gates.itertuples():
            bits = [v[n] for n in g.inputs.split()]
            v[g.output] = LOGIC[g.type](*bits)
            cases.append((g.type, "".join(map(str, bits))))
        total = v["s0"] + 2 * v["s1"] + 4 * v["s2"] + 8 * v["cout"]
        if total != A + B:
            sys.exit(f"logic gives {total} for {A}+{B}")
        for net, col in OUTPUTS.items():
            if not ((d[col] > d.vdd / 2).astype(int) == v[net]).all():
                sys.exit(f"SPICE output {net} disagrees with the logic for A={A}, B={B}")
        vec_cases[(A, B)] = cases
    print(f"2. logic: all {len(vec_cases)} vectors give the correct 3-bit sum and match the SPICE outputs (VDD/2)")

    # 3-4. lookup and sum
    gp = pd.read_csv(args.predictions)
    gp["inp"] = gp.A.astype(int).astype(str) + gp.B.fillna(-1).astype(int).astype(str).str.replace("-1", "")
    ml = gp.set_index(["gate", "inp", "corner"]).leak_ml.to_dict()
    sp = gp.set_index(["gate", "inp", "corner"]).leak_spice.to_dict()
    leak_ml, leak_ceil = np.zeros(len(cla)), np.zeros(len(cla))
    for k, (A, B, c) in enumerate(zip(cla.A, cla.B, cla.corner)):
        for typ, inp in vec_cases[(A, B)]:
            key = (typ, inp, c)
            if key not in ml:
                sys.exit(f"missing Part B prediction for {key}")
            leak_ml[k] += ml[key]
            leak_ceil[k] += sp[key]
    print(f"3-4. summed {len(gates)} gate leakages for {len(cla)} (vector, corner) rows from {args.predictions}")

    # 5-6. compare and write
    pred = pd.DataFrame({"vector": cla.A.astype(str) + "+" + cla.B.astype(str), "A": cla.A, "B": cla.B,
                         "corner": cla.corner, "vdd": cla.vdd, "temp": cla.temp, "n_vth0": cla.n_vth0,
                         "p_vth0": cla.p_vth0, "leak_spice": cla.leak, "leak_ml": leak_ml, "leak_ceiling": leak_ceil})
    pred["error_ml_pct"] = 100 * (pred.leak_ml - pred.leak_spice) / pred.leak_spice
    pred["error_ceiling_pct"] = 100 * (pred.leak_ceiling - pred.leak_spice) / pred.leak_spice
    pred.to_csv(args.out / "cla3_predictions.csv", index=False)

    rows = [{"vector": "ALL", "rows": len(pred), **{f"ml_{k}": v for k, v in metrics(pred.leak_spice, pred.leak_ml).items()},
             **{f"ceiling_{k}": v for k, v in metrics(pred.leak_spice, pred.leak_ceiling).items()}}]
    for vec, d in pred.groupby("vector", sort=False):
        rows.append({"vector": vec, "rows": len(d), **{f"ml_{k}": v for k, v in metrics(d.leak_spice, d.leak_ml).items()},
                     **{f"ceiling_{k}": v for k, v in metrics(d.leak_spice, d.leak_ceiling).items()}})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "cla3_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.loglog(pred.leak_spice, pred.leak_ml, ".", markersize=2, alpha=0.4, label="ML (sum of ML gate leakages)")
    lo, hi = pred[["leak_spice", "leak_ml"]].min().min() / 1.3, pred[["leak_spice", "leak_ml"]].max().max() * 1.3
    ax.loglog([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set(xlabel="SPICE CLA leakage (A)", ylabel="ML CLA leakage (A)",
           title=f"3-bit CLA: {len(pred)} simulations (64 vectors x 200 corners)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "cla3_parity.png", dpi=150)
    plt.close(fig)

    # Part B accuracy of the same gate predictions, for the comparison
    eb = 100 * (gp.leak_ml - gp.leak_spice) / gp.leak_spice
    acc_ml, acc_ceil = accuracy_definitions(pred, "leak_ml"), accuracy_definitions(pred, "leak_ceiling")
    gp["case"] = gp.gate + "_" + gp.inp
    acc_b = accuracy_definitions(gp.rename(columns={}), "leak_ml", group="case")
    worst = pred.loc[pred.error_ml_pct.abs().idxmax()]
    summary = {"predictions": str(args.predictions), "partB_mean_acc_pct": 100 - eb.abs().mean(),
               **{f"partB_{k}": v for k, v in acc_b.items()},
               **{f"partC_ml_{k}": v for k, v in acc_ml.items()}, **{f"partC_ceiling_{k}": v for k, v in acc_ceil.items()},
               **{f"partC_ml_{k}": v for k, v in metrics(pred.leak_spice, pred.leak_ml).items()},
               **{f"partC_ceiling_{k}": v for k, v in metrics(pred.leak_spice, pred.leak_ceiling).items()},
               "worst_vector": worst.vector, "worst_corner": int(worst.corner), "worst_error_pct": worst.error_ml_pct,
               "worst_temp": worst.temp}
    pd.Series(summary).to_csv(args.out / "summary.csv", header=["value"])

    print("\nMean accuracy (Part C, 12,800 rows; per-vector medians for the last definition):")
    print(f"  {'definition':48s} {'ML':>8s} {'ceiling':>8s}")
    for k in acc_ml:
        print(f"  {k:48s} {acc_ml[k]:8.2f} {acc_ceil[k]:8.2f}")
    m = metrics(pred.leak_spice, pred.leak_ml)
    print(f"\nPart B (same gate predictions): mean accuracy {100 - eb.abs().mean():.2f} %   "
          f"Part C: {acc_ml['acc_100_minus_mean_abs_pct']:.2f} %")
    print(f"Part C ML: bias {m['bias_pct']:+.2f} %, median |e| {m['median_abs_error_pct']:.2f} %, "
          f"p95 |e| {m['p95_abs_error_pct']:.2f} %, worst {worst.error_ml_pct:+.2f} % (vector {worst.vector}, "
          f"corner {int(worst.corner)}, T {worst.temp:.0f} C)")
    print("\nError breakdown (mean |error| %, ML / ceiling):")
    bins = {"temperature (C)": pd.cut(pred.temp, [-40, 0, 40, 85, 125], include_lowest=True),
            "NMOS vth0 quartile": pd.qcut(pred.n_vth0, 4, labels=["low", "mid-low", "mid-high", "high"]),
            "PMOS |vth0| quartile": pd.qcut(pred.p_vth0.abs(), 4, labels=["low", "mid-low", "mid-high", "high"])}
    for name, b in bins.items():
        g = pred.groupby(b, observed=True)
        s = "  ".join(f"{k}: {g.error_ml_pct.apply(lambda x: x.abs().mean())[k]:.2f}/"
                      f"{g.error_ceiling_pct.apply(lambda x: x.abs().mean())[k]:.2f}" for k in g.groups)
        print(f"  {name:22s} {s}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
