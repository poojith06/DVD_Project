"""Check C1: the Part B Kirchhoff (super-node) method is exact when fed the true currents.

10 random rows per Part B input case (18 cases) are re-simulated with a 0 V source in series with every terminal
of every transistor (corner written into the model card). The SPICE terminal currents are passed to the SAME
function ML_Models/evaluate_gates.py uses (supply_kcl, with the recorded node voltages and the same ON/OFF rule)
and the result is compared with the recorded leakage."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simlib import REPO, card_with, corner_values, printed, run_folder

sys.path.insert(0, str(REPO / "ML_Models"))
from evaluate_gates import GATES, node_voltages, supply_kcl

SIM = HERE / "sim" / "c1"
SIM.mkdir(parents=True, exist_ok=True)
for f in SIM.glob("*"):
    f.unlink()
rng = np.random.default_rng(1)
TERMS = ["d", "g", "s", "b"]
jobs = []
for g in GATES:
    text = (REPO / "Gates_Part-B" / "netlists" / f"{g}_ngspice.cir").read_text().replace("\r", "")
    circuit = text[:text.index(".control")]
    lines, probes = [], []
    for line in circuit.splitlines():
        m = re.match(r"^\s*(M\w+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$", line)
        if not m:
            lines.append(line)
            continue
        name, nodes, rest = m.group(1), m.group(2, 3, 4, 5), m.group(6) + " " + m.group(7)
        new = [f"{name.lower()}_{t}" for t in TERMS]
        lines.append(f"{name} {' '.join(new)} {rest}")
        for t, node, nn in zip(TERMS, nodes, new):
            lines.append(f"Vm_{name.lower()}_{t} {node} {nn} dc 0")      # current n+ -> n- = into the terminal
            probes.append(f"i(vm_{name.lower()}_{t})")
    base = "\n".join(lines) + "\n"
    d = pd.read_csv(REPO / "Gates_Part-B" / "data" / f"{g}.csv")
    keys = ["A"] + (["B"] if "B" in d else [])
    for _, grp in d.groupby(keys):
        for i in rng.choice(grp.index.to_numpy(), 10, replace=False):
            r = d.loc[i]
            name = f"{g}_{i}"
            (SIM / f"{name}.pm").write_text(card_with(corner_values(r)), newline="\n")
            c = base.replace(".include 22nm_HP.pm", f".include {name}.pm").replace("Vsup vdd 0 dc 1", f"Vsup vdd 0 dc {r.vdd!r}")
            c = c.replace("Va a 0 dc 0", f"Va a 0 dc {r.A * r.vdd!r}")
            if "B" in d:
                c = c.replace("Vb b 0 dc 0", f"Vb b 0 dc {r.B * r.vdd!r}")
            # starting guess from the recorded voltages (helps convergence with the extra probe sources only)
            guess = {"out": r.v_out, "x" if g in ("nand2", "and2") else "y": r.get("v_int"), "n": r.get("v_n")}
            ns = " ".join(f"v({k})={v!r}" for k, v in guess.items() if v is not None and not pd.isna(v))
            c += f".nodeset {ns}\n.temp {r.temp!r}\n.control\nop\nprint -i(vsup)\n" + "".join(f"print {p}\n" for p in probes) + ".endc\n.end\n"
            (SIM / f"{name}.cir").write_text(c, newline="\n")
            jobs.append((g, i, name))

print(f"simulating {len(jobs)} netlists ...", flush=True)
outs = run_folder(SIM)
rows = []
for g, i, name in jobs:
    vals = dict(printed(outs[name]))
    if "-i(vsup)" not in vals:
        print(f"  {name}: ngspice did not converge - skipped")
        continue
    d = pd.read_csv(REPO / "Gates_Part-B" / "data" / f"{g}.csv").loc[[i]].reset_index(drop=True)
    v = node_voltages(d)
    vdd = d["vdd"].to_numpy()
    devices = []
    for mname, pol, dn, gn, sn, bn, _ in GATES[g]:
        cur = np.array([[vals[f"i(vm_{mname.lower()}_{t})"] for t in TERMS]])
        on = v[gn] > vdd / 2 if pol == "nmos" else v[gn] < vdd / 2
        devices.append((pol, (dn, gn, sn, bn), on, cur))
    kcl = supply_kcl(v, devices)[0]
    leak_csv, leak_sim = float(d.leak[0]), vals["-i(vsup)"] if "-i(vsup)" in vals else -vals["i(vsup)"]
    inputs = f"{int(d.A[0])}" + (f"{int(d.B[0])}" if "B" in d else "")
    rows.append({"gate": g, "inputs": inputs, "row": i, "leak_csv": leak_csv, "leak_resim": leak_sim, "kcl_spice_currents": kcl,
                 "rel_kcl_vs_csv": abs(kcl - leak_csv) / leak_csv, "rel_kcl_vs_resim": abs(kcl - leak_sim) / leak_sim})
res = pd.DataFrame(rows)
res.to_csv(HERE / "c1_kcl_method.csv", index=False)
t = res.groupby(["gate", "inputs"], sort=False)[["rel_kcl_vs_csv", "rel_kcl_vs_resim"]].max()
pd.set_option("display.width", 200)
print(t.to_string(float_format=lambda x: f"{x:.1e}"))
w = res.loc[res.rel_kcl_vs_csv.idxmax()]
print(f"\nworst: {w.gate} {w.inputs} row {w.row}: KCL(SPICE currents) {w.kcl_spice_currents:.6e} vs CSV {w.leak_csv:.6e} "
      f"(rel {w.rel_kcl_vs_csv:.1e}); vs re-simulated -i(vsup) rel {w.rel_kcl_vs_resim:.1e}")
