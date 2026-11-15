"""Loading analysis for the report (EXP-12, EXP-19): the leakage recorded in Part B is the current drawn from VDD
(-i(vsup)). In the isolated gate simulations the inputs are ideal voltage sources, which also supply (inputs at 1) or
absorb (inputs at 0) a small gate current. In a circuit those currents flow through the driving gates instead.

Re-simulates every Part B row (corner values taken from the saved CSV, applied with one altermod line per parameter,
as in the original netlists) and records the current of the input sources next to the supply current.
Then (1) tabulates how large these pin currents are relative to the recorded leakage, per gate case, and
(2) adds the pin current of every input that is driven by another gate and is at logic 1 to the sum of the Part B SPICE
gate leakages of the CLA ("ceiling with pin currents"). Evaluation only: no data or netlist is changed.

    python validation_checks/pin_currents.py [--out results/analysis/loading_analysis]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simlib import PARAMS, REPO, printed, run_folder

sys.path.insert(0, str(REPO / "ml"))
import evaluate_cla as ec

GATES = ["inv", "nand2", "nor2", "and2", "or2"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "results" / "analysis" / "loading_analysis")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    SIM = HERE / "sim" / "pins"
    SIM.mkdir(parents=True, exist_ok=True)
    for f in SIM.glob("*"):
        f.unlink()
    (SIM / "22nm_HP.pm").write_text((REPO / "model" / "22nm_HP.pm").read_text(), newline="\n")
    data = {}
    for g in GATES:
        text = (REPO / "partB_gates" / "netlists" / f"{g}_ngspice.cir").read_text().replace("\r", "")
        circuit = text[:text.index(".control")]
        d = pd.read_csv(REPO / "partB_gates" / "data" / f"{g}.csv")
        two = g != "inv"
        t = circuit + ".control\n"
        for _, r in d.iterrows():
            t += f"alter vsup dc = {r.vdd!r}\noption temp = {r.temp!r}\n"
            for dev in "np":
                for q in PARAMS:
                    t += f"altermod {dev}mos {q} = {r[dev + '_' + q]!r}\n"
            t += f"alter va dc = {r.A * r.vdd!r}\n" + (f"alter vb dc = {r.B * r.vdd!r}\n" if two else "") + "op\n"
            t += "print -i(vsup) i(va)" + (" i(vb)" if two else "") + "\n"
        t += ".endc\n.end\n"
        (SIM / f"{g}.cir").write_text(t, newline="\n")
        data[g] = d
    print("re-simulating 3,600 Part B rows ...", flush=True)
    outs = run_folder(SIM)
    rows = []
    for g in GATES:
        d = data[g]
        k = 3 if g != "inv" else 2
        v = np.array([x for _, x in printed(outs[g])]).reshape(-1, k)
        assert len(v) == len(d), f"{g}: {len(v)} results for {len(d)} rows"
        out = pd.DataFrame({"gate": g, "corner": d.corner, "A": d.A, "B": d["B"] if "B" in d else np.nan,
                            "leak": d.leak, "leak_resim": v[:, 0], "pin_a": -v[:, 1], "pin_b": -v[:, 2] if k == 3 else np.nan})
        rows.append(out)
        print(f"  {g}: max |resim/csv - 1| = {np.max(np.abs(out.leak_resim / out.leak - 1)):.1e}", flush=True)
    pins = pd.concat(rows, ignore_index=True)
    pins["inputs"] = pins.A.astype(int).astype(str) + pins.B.fillna(-1).astype(int).astype(str).str.replace("-1", "")
    # pin current = current delivered by the input source into the circuit (positive: source supplies it, input at 1)
    pins["supplied_by_inputs"] = np.where(pins.A == 1, pins.pin_a, 0.0) + np.where(pins.B == 1, pins.pin_b, 0.0)
    pins["absorbed_by_inputs"] = -(np.where(pins.A == 0, pins.pin_a, 0.0) + np.where(pins.B == 0, pins.pin_b, 0.0))
    pins.to_csv(args.out / "pin_currents.csv", index=False)

    tab = []
    for (g, inp), d in pins.groupby(["gate", "inputs"], sort=False):
        s = d.supplied_by_inputs / d.leak
        a = d.absorbed_by_inputs / d.leak
        tab.append({"gate": g, "inputs": inp, "median_supplied_over_leak_pct": 100 * s.median(),
                    "max_supplied_over_leak_pct": 100 * s.max(), "median_absorbed_over_leak_pct": 100 * a.median(),
                    "max_absorbed_over_leak_pct": 100 * a.max()})
    tab = pd.DataFrame(tab)
    tab.to_csv(args.out / "pin_current_shares.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nInput-pin currents relative to the recorded supply leakage (%):")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # ---- CLA ceiling with pin currents
    gates = ec.topological(ec.gate_list())
    cla = pd.read_csv(ec.CLA_DIR / "data" / "cla3.csv")
    PRIMARY = {f"{x}{i}" for x in "ab" for i in range(3)}
    pa = pins.set_index(["gate", "inputs", "corner"])["pin_a"].to_dict()
    pb = pins.set_index(["gate", "inputs", "corner"])["pin_b"].to_dict()
    leak = pins.set_index(["gate", "inputs", "corner"])["leak"].to_dict()
    add_pins = np.zeros(len(cla))
    base = np.zeros(len(cla))
    vec_cache = {}
    for (A, B), d in cla.groupby(["A", "B"]):
        r = d.iloc[0]
        v = {f"a{i}": int(r[f"A{i}"]) for i in range(3)} | {f"b{i}": int(r[f"B{i}"]) for i in range(3)}
        lst = []
        for g in gates.itertuples():
            ins = g.inputs.split()
            bits = [v[n] for n in ins]
            v[g.output] = ec.LOGIC[g.type](*bits)
            lst.append((g.type, "".join(map(str, bits)), [(n not in PRIMARY) and b == 1 for n, b in zip(ins, bits)]))
        vec_cache[(A, B)] = lst
    for i, (A, B, c) in enumerate(zip(cla.A, cla.B, cla.corner)):
        for typ, inp, flags in vec_cache[(A, B)]:
            base[i] += leak[(typ, inp, c)]
            if flags[0]:
                add_pins[i] += pa[(typ, inp, c)]
            if len(flags) > 1 and flags[1]:
                add_pins[i] += pb[(typ, inp, c)]
    spice = cla.leak.to_numpy()
    res = []
    for label, est in (("sum of Part B SPICE gate leakages (ceiling, as reported)", base),
                       ("ceiling + pin currents of internally driven inputs at logic 1", base + add_pins)):
        e = 100 * (est - spice) / spice
        res.append({"estimate": label, "mean_accuracy_pct": 100 - np.abs(e).mean(), "median_accuracy_pct": 100 - np.median(np.abs(e)),
                    "bias_pct": np.median(e), "p95_abs_error_pct": np.percentile(np.abs(e), 95)})
    res = pd.DataFrame(res)
    res.to_csv(args.out / "cla_ceiling_with_pins.csv", index=False)
    print("\nCLA (12,800 rows): perfect gate models, with and without the pin currents:")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
