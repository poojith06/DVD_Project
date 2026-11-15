"""Task 3: cause of the drain-source asymmetry of the OFF devices (analysis only; no netlist used for the data is touched).

Hypothesis: the asymmetry comes from the default junction geometry (model card geomod = 1, no AD/AS/PD/PS on the
instance), which gives the drain junction more leakage than the source.

A swap test: simulate a bias point and the same point with vd and vs exchanged. For a symmetric device the currents
mirror: id(swapped) = is(original), is(swapped) = id(original), ig and ib unchanged.
Variants (same rows each):
  (a) as the data were generated: M1 d g s b <type> L=22n W=...   (model card unchanged)
  (b) equal junction geometry on the instance: AD = AS, PD = PS
  (c) diagnostic: model card geomod = 0 (only used to look for the cause if (b) does not remove the asymmetry)
Operating points:
  Part A rows : 15 random rows each of nmos_off.csv and pmos_off.csv (random vd, vs)
  gate points : one terminal at each rail (the points a logic gate puts an OFF transistor in), 20 Part B corners
Output: the asymmetry measure is |mirrored - original| / |original| per current.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simlib import OPTIONS, PARAMS, REPO, card_with, corner_values, printed, run_folder

SIM = HERE / "sim" / "cause"
SIM.mkdir(parents=True, exist_ok=True)
for f in SIM.glob("*"):
    f.unlink()
rng = np.random.default_rng(7)
READ_ALL = ("let cig = -(@m1[igs] + @m1[igd] + @m1[igb] + @m1[igcs] + @m1[igcd])\n"
            "let cib = -(@m1[ibd] + @m1[ibs]) + @m1[igb] + @m1[isub] + @m1[igidl] + @m1[igisl]\n"
            "print i(vd) cig i(vs) cib\n")
VARIANTS = {"a_as_generated": ("", ""), "b_equal_AD_AS_PD_PS": ("EQUAL", ""), "c_geomod0": ("", "geomod0")}
jobs = []   # (point set, case, row id, variant, orientation, file name)


def write(name, case, vals, w, vdd_r, temp, vd, vs, vg, vb, variant):
    inst_extra, card_extra = VARIANTS[variant]
    pol = case[:4]
    card = card_with(vals)
    if card_extra == "geomod0":
        card = re.sub(r"(?i)(geomod\s*=\s*)1", r"\g<1>0", card)
    (SIM / f"{name}.pm").write_text(card, newline="\n")
    extra = f" AD={w * 1e-7!r} AS={w * 1e-7!r} PD={2 * (w + 1e-7)!r} PS={2 * (w + 1e-7)!r}" if inst_extra == "EQUAL" else ""
    net = (f"* {name}\n{OPTIONS}\n.include {name}.pm\n.temp {temp!r}\n"
           f"Vd d 0 dc {vd!r}\nVg g 0 dc {vg!r}\nVs s 0 dc {vs!r}\nVb b 0 dc {vb!r}\n"
           f"M1 d g s b {pol} L=22n W={w!r}{extra}\n.control\nop\n{READ_ALL}.endc\n.end\n")
    (SIM / f"{name}.cir").write_text(net, newline="\n")


# ---- Part A random rows
for case in ("nmos_off", "pmos_off"):
    d = pd.read_csv(REPO / "partA_transistor" / "data" / f"{case}.csv")
    pol = case[:4]
    for i in rng.choice(len(d), 15, replace=False):
        r = d.iloc[i]
        vals = {"nmos": {}, "pmos": {}}
        vals[pol] = {p: float(r[p]) for p in PARAMS}
        for var in VARIANTS:
            for ori, (vd, vs) in (("orig", (r.vd, r.vs)), ("swap", (r.vs, r.vd))):
                name = f"A_{case}_{i}_{var}_{ori}"
                write(name, case, vals, float(r.w), r.vdd, r.temp, vd, vs, r.vg, r.vb, var)
                jobs.append(("Part A rows", case, i, var, ori, name))

# ---- gate operating points: one terminal at each rail, 20 Part B corners
inv = pd.read_csv(REPO / "partB_gates" / "data" / "inv.csv")
corners = inv[inv.A == 0].sort_values("corner").iloc[::10].head(20)
for _, r in corners.iterrows():
    for case in ("nmos_off", "pmos_off"):
        pol = case[:4]
        vals = corner_values(r)
        w = 88e-9
        # NMOS OFF: gate = body = 0, current flows between the rails; PMOS OFF: gate = body = VDD
        rail = 0.0 if pol == "nmos" else r.vdd
        vd0, vs0 = (r.vdd, 0.0)
        for var in VARIANTS:
            for ori, (vd, vs) in (("orig", (vd0, vs0)), ("swap", (vs0, vd0))):
                name = f"G_{case}_{int(r.corner)}_{var}_{ori}"
                write(name, case, vals, w, r.vdd, r.temp, vd, vs, rail, rail, var)
                jobs.append(("gate points", case, int(r.corner), var, ori, name))

print(f"simulating {len(jobs)} netlists ...", flush=True)
outs = run_folder(SIM)


def read(name):
    text = outs[name]
    vals = dict(printed(text))
    vals.update({m.group(1): float(m.group(2)) for m in re.finditer(r"(?m)^\s*(cig|cib)\s*=\s*([-+0-9.eE]+)", text)})
    if not all(k in vals for k in ("i(vd)", "cig", "i(vs)", "cib")):
        return None
    return np.array([-vals["i(vd)"], vals["cig"], -vals["i(vs)"], vals["cib"]])    # id, ig, is, ib into the terminals


rows = []
for pset, case, rid, var, ori, name in jobs:
    if ori != "orig":
        continue
    o, s = read(name), read(name.replace("_orig", "_swap"))
    if o is None or s is None:
        continue
    m = s[[2, 1, 0, 3]]
    rel = np.abs(m - o) / np.maximum(np.abs(o), 1e-18)
    big = np.abs(o).max()
    rows.append({"set": pset, "case": case, "row": rid, "variant": var, "rel_id": rel[0], "rel_is": rel[2], "rel_ig": rel[1],
                 "rel_ib": rel[3], "abs_diff_over_largest_I": np.abs(m - o).max() / big,
                 "rel_dominant": rel[0] if abs(o[0]) >= abs(o[2]) else rel[2]})
res = pd.DataFrame(rows)
res.to_csv(HERE / "sym_cause.csv", index=False)
pd.set_option("display.width", 220)
fmt = lambda x: f"{x:.2e}"
print(f"\nSymmetry test results ({len(res)} point pairs). rel_dominant = mirror error of the larger of |id|, |is|;")
print("abs_diff_over_largest_I = largest absolute mirror difference of any current / largest terminal current")
for pset in ("Part A rows", "gate points"):
    t = res[res.set == pset].groupby(["case", "variant"]).agg(
        n=("row", "size"), dominant_median=("rel_dominant", "median"), dominant_max=("rel_dominant", "max"),
        small_id_is_median=("rel_id", "median"), over_largest_median=("abs_diff_over_largest_I", "median"),
        over_largest_max=("abs_diff_over_largest_I", "max"))
    print(f"\n{pset}:")
    print(t.to_string(float_format=fmt))
