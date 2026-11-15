"""Symmetry check for the drain-source preprocessing: is the 22nm HP transistor symmetric?
For 10 random rows per Part A case, simulate the row as recorded and with vd and vs swapped (process values in the
model card). Symmetric device: id(swapped) = is(original), is(swapped) = id(original), ig and ib unchanged."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simlib import OPTIONS, PARAMS, REPO, card_with, printed, run_folder

SIM = HERE / "sim" / "sym"
SIM.mkdir(parents=True, exist_ok=True)
for f in SIM.glob("*"):
    f.unlink()
rng = np.random.default_rng(2)
jobs = []
for case in ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]:
    d = pd.read_csv(REPO / "partA_transistor" / "data" / f"{case}.csv")
    pol = case[:4]
    for i in rng.choice(len(d), 10, replace=False):
        r = d.iloc[i]
        vals = {"nmos": {}, "pmos": {}}
        vals[pol] = {p: float(r[p]) for p in PARAMS}
        (SIM / f"{case}_{i}.pm").write_text(card_with(vals), newline="\n")
        for tag, vd, vs in (("orig", r.vd, r.vs), ("swap", r.vs, r.vd)):
            net = (f"* {case} {i} {tag}\n{OPTIONS}\n.include {case}_{i}.pm\n.temp {r.temp!r}\n"
                   f"Vd d 0 dc {vd!r}\nVg g 0 dc {r.vg!r}\nVs s 0 dc {vs!r}\nVb b 0 dc {r.vb!r}\n"
                   f"M1 d g s b {pol} L=22n W={r.w!r}\n.control\nop\nprint i(vd) i(vg) i(vs) i(vb)\n.endc\n.end\n")
            (SIM / f"{case}_{i}_{tag}.cir").write_text(net, newline="\n")
        jobs.append((case, i))
outs = run_folder(SIM)
rows = []
for case, i in jobs:
    o = -np.array([v for _, v in printed(outs[f"{case}_{i}_orig"])])   # currents into the terminals
    s = -np.array([v for _, v in printed(outs[f"{case}_{i}_swap"])])
    mirrored = s[[2, 1, 0, 3]]                                          # swapped run with id <-> is exchanged
    rel = np.abs(mirrored - o) / np.maximum(np.abs(o), 1e-18)
    rows.append({"case": case, "row": i, "rel_id": rel[0], "rel_ig": rel[1], "rel_is": rel[2], "rel_ib": rel[3],
                 "abs_ig_A": abs(mirrored[1] - o[1]), "abs_ib_A": abs(mirrored[3] - o[3])})
res = pd.DataFrame(rows)
res.to_csv(HERE / "sym_check.csv", index=False)
print("max over 10 rows: relative difference between the original and the swapped-and-mirrored simulation")
print(res.groupby("case")[["rel_id", "rel_is", "rel_ig", "rel_ib", "abs_ig_A", "abs_ib_A"]].max()
      .to_string(float_format=lambda x: f"{x:.1e}"))
