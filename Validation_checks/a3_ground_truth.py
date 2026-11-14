"""Check A3: ground truth without altermod. 20 random rows of every Part A CSV, every Part B gate CSV and cla3.csv
are re-simulated with the process values written DIRECTLY into the .model card text (one card per row), and the
currents / leakage / node voltages are compared with the CSV values (which carry 6 significant digits)."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simlib import OPTIONS, PARAMS, REPO, card_with, corner_values, printed, run_folder

SIM = Path(__file__).resolve().parent / "sim" / "a3"
SIM.mkdir(parents=True, exist_ok=True)
for f in SIM.glob("*"):
    f.unlink()
rng = np.random.default_rng(0)
N = 20
jobs = []           # (kind, name, csv row, expected list)

# ---------------- Part A: single transistor, every terminal its own source
for case in ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]:
    d = pd.read_csv(REPO / "Datasets" / "data" / f"{case}.csv")
    pol = case[:4]
    for i in rng.choice(len(d), N, replace=False):
        r = d.iloc[i]
        name = f"A_{case}_{i}"
        vals = {"nmos": {}, "pmos": {}}
        vals[pol] = {p: float(r[p]) for p in PARAMS}
        (SIM / f"{name}.pm").write_text(card_with({"nmos": vals["nmos"], "pmos": vals["pmos"]}), newline="\n")
        if case == "pmos_off":
            # the dataset reads PMOS OFF ig / ib from BSIM4's internal currents (the source branch currents are
            # rounded to ~2.2e-16 A), so the check must read them the same way (same formulas as the dataset netlist)
            readout = ("let cig = -(@m1[igs] + @m1[igd] + @m1[igb] + @m1[igcs] + @m1[igcd])\n"
                       "let cib = -(@m1[ibd] + @m1[ibs]) + @m1[igb] + @m1[isub] + @m1[igidl] + @m1[igisl]\n"
                       "print i(vd) cig i(vs) cib\n")
            sign = [-1.0, 1.0, -1.0, 1.0]            # id, is = -i(source); ig, ib are already "into the terminal"
        else:
            readout, sign = "print i(vd) i(vg) i(vs) i(vb)\n", [-1.0] * 4
        net = (f"* {name}\n{OPTIONS}\n.include {name}.pm\n.temp {r.temp!r}\n"
               f"Vd d 0 dc {r.vd!r}\nVg g 0 dc {r.vg!r}\nVs s 0 dc {r.vs!r}\nVb b 0 dc {r.vb!r}\n"
               f"M1 d g s b {pol} L=22n W={r.w!r}\n.control\nop\n{readout}.endc\n.end\n")
        (SIM / f"{name}.cir").write_text(net, newline="\n")
        jobs.append(("A", name, case, [r.id, r.ig, getattr(r, "is"), r.ib], sign))

# ---------------- Part B: gate netlists (circuit part unchanged), corner in the card, sources set directly
NODE = {"nand2": ["x"], "nor2": ["y"], "and2": ["x", "n"], "or2": ["y", "n"], "inv": []}
for g in ["inv", "nand2", "nor2", "and2", "or2"]:
    text = (REPO / "Gates_Part-B" / "netlists" / f"{g}_ngspice.cir").read_text().replace("\r", "")
    circuit = text[:text.index(".control")]
    d = pd.read_csv(REPO / "Gates_Part-B" / "data" / f"{g}.csv")
    for i in rng.choice(len(d), N, replace=False):
        r = d.iloc[i]
        name = f"B_{g}_{i}"
        (SIM / f"{name}.pm").write_text(card_with(corner_values(r)), newline="\n")
        c = circuit.replace(".include 22nm_HP.pm", f".include {name}.pm").replace("Vsup vdd 0 dc 1", f"Vsup vdd 0 dc {r.vdd!r}")
        c = c.replace("Va a 0 dc 0", f"Va a 0 dc {r.A * r.vdd!r}")
        if g != "inv":
            c = c.replace("Vb b 0 dc 0", f"Vb b 0 dc {r.B * r.vdd!r}")
        cols = ["leak"] + (["v_int"] if NODE[g] else []) + (["v_n"] if len(NODE[g]) > 1 else []) + ["v_out"]
        prints = "-i(vsup) " + " ".join(f"v({n})" for n in NODE[g]) + " v(out)"
        c += f".temp {r.temp!r}\n.control\nop\nprint {prints}\n.endc\n.end\n"
        (SIM / f"{name}.cir").write_text(c, newline="\n")
        jobs.append(("B", name, g, [r[k] for k in cols], [1.0] * len(cols)))

# ---------------- Part C: CLA netlist (circuit part unchanged), cla3.sub + gates.sub copied next to it
pc = REPO / "PartC_3bit CLA adder" / "netlists"
sub = pc / "cla3.sub"
(SIM / "cla3.sub").write_text(sub.read_text() if sub.exists() else subprocess.run(
    ["git", "-C", str(REPO), "show", "HEAD:PartC_3bit CLA adder/netlists/cla3.sub"],
    capture_output=True, text=True, check=True).stdout, newline="\n")
(SIM / "gates.sub").write_text((pc / "gates.sub").read_text(), newline="\n")
text = (pc / "cla3_ngspice.cir").read_text().replace("\r", "")
circuit = text[:text.index(".control")]
d = pd.read_csv(REPO / "PartC_3bit CLA adder" / "data" / "cla3.csv")
for i in rng.choice(len(d), N, replace=False):
    r = d.iloc[i]
    name = f"C_cla3_{i}"
    (SIM / f"{name}.pm").write_text(card_with(corner_values(r)), newline="\n")
    c = circuit.replace(".include 22nm_HP.pm", f".include {name}.pm").replace("Vsup vdd 0 dc 1", f"Vsup vdd 0 dc {r.vdd!r}")
    for bit in ("A0", "A1", "A2", "B0", "B1", "B2"):
        c = c.replace(f"V{bit.lower()} {bit.lower()} 0 dc 0", f"V{bit.lower()} {bit.lower()} 0 dc {r[bit] * r.vdd!r}")
    c += f".temp {r.temp!r}\n.control\nop\nprint -i(vsup) v(s0) v(s1) v(s2) v(cout)\n.endc\n.end\n"
    (SIM / f"{name}.cir").write_text(c, newline="\n")
    jobs.append(("C", name, "cla3", [r.leak, r.v_s0, r.v_s1, r.v_s2, r.v_cout], [1.0] * 5))

print(f"simulating {len(jobs)} netlists ...", flush=True)
outs = run_folder(SIM)

rows = []
for kind, name, group, expected, sign in jobs:
    text = outs[name]
    named = printed(text) + [(m.group(1), float(m.group(2))) for m in
                             re.finditer(r"(?m)^\s*(cig|cib)\s*=\s*([-+0-9.eE]+)", text)]
    order = ["i(vd)", "i(vg)", "i(vs)", "i(vb)"] if kind == "A" and group != "pmos_off" else None
    if kind == "A" and group == "pmos_off":
        order = ["i(vd)", "cig", "i(vs)", "cib"]
    d_ = dict(named)
    vals = [d_[k] for k in order] if order and all(k in d_ for k in order) else (
        [] if order else [v for _, v in printed(text)])
    if len(vals) != len(expected):
        rows.append({"part": kind, "file": group, "row": name, "status": "SIM FAILED"})
        continue
    sim = np.array(vals) * np.array(sign)
    exp = np.array(expected, float)
    if kind == "A":
        rel = np.abs(sim - exp) / np.maximum(np.abs(exp), 1e-18)
        rows.append({"part": kind, "file": group, "row": name, "rel_id": rel[0], "rel_ig": rel[1], "rel_is": rel[2],
                     "rel_ib": rel[3], "abs_ig_A": abs(sim[1] - exp[1]), "abs_ib_A": abs(sim[3] - exp[3])})
    else:
        rows.append({"part": kind, "file": group, "row": name, "rel_leak": abs(sim[0] - exp[0]) / abs(exp[0]),
                     "max_node_diff_V": float(np.max(np.abs(sim[1:] - exp[1:])))})
res = pd.DataFrame(rows)
res.to_csv(Path(__file__).resolve().parent / "a3_ground_truth.csv", index=False)
pd.set_option("display.width", 220)
summary = res.groupby(["part", "file"]).agg({c: "max" for c in res.columns if c.startswith(("rel_", "abs_", "max_"))})
print("max over the 20 rows (relative differences; node voltages in V):")
print(summary.to_string(float_format=lambda x: f"{x:.1e}"))
print("failed simulations:", int((res.get("status") == "SIM FAILED").sum()) if "status" in res else 0)
