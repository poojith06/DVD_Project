"""Re-measure the PMOS OFF dataset with higher current resolution (does not touch Datasets/).

Why: in Datasets/data/pmos_off.csv the gate current ig is limited by double-precision rounding of the
voltage-source branch current (75% of values are multiples of 2.22e-16 A). Here each of the gate and
body sources feeds its terminal through a sense resistor (RG, RB) and the current is taken from the
voltage across the resistor, so the resolution is far finer. Same 10,000 input rows as the original
(same seed, same samples); only the current measurement differs.

    python make_hires_pmos_off.py     # writes netlists/pmos_off_hires_ngspice.cir
    (then)  ngspice -b pmos_off_hires_ngspice.cir   from inside netlists/   -> ../data/pmos_off.csv
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RG, RB = 1e9, 1e8  # sense resistors. Voltage drop = I*R: ig<=1e-14 -> <=10 uV, ib<=4e-12 -> <=0.4 mV
src = pd.read_csv(HERE.parent / "Datasets" / "data" / "pmos_off.csv")

L = ["* PMOS OFF, high-resolution gate/body current via sense resistors (replays the original 10,000 rows)",
     ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9", ".include 22nm_HP.pm",
     "Vd d 0 dc 0", "Vgs gs 0 dc 0", "Vs s 0 dc 0", "Vbs bs 0 dc 0",
     f"Rg gs g {RG:g}", f"Rb bs b {RB:g}", "M1 d g s b pmos L=22n W=44n", ".control",
     'echo "vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib,ig_src,ib_src" > ../data/pmos_off_hires_raw.csv']
for _, r in src.iterrows():
    L += [f"alter vd dc = {r.vd:.9g}", f"alter vs dc = {r.vs:.9g}", f"alter vgs dc = {r.vg:.9g}", f"alter vbs dc = {r.vb:.9g}",
          f"alter m1 w = {r.w:.9g}",
          f"altermod pmos vth0 = {r.vth0:.9g} toxe = {r.toxe:.9g} ndep = {r.ndep:.9g} u0 = {r.u0:.9g} xj = {r.xj:.9g}",
          f"option temp = {r.temp:.9g}", "op",
          "let cid = -i(vd)", "let cis = -i(vs)",
          f"let cig = (v(gs) - v(g)) / {RG:g}", f"let cib = (v(bs) - v(b)) / {RB:g}",   # current INTO the terminal
          "let cigs = -i(vgs)", "let cibs = -i(vbs)",
          f'echo "{r.vdd:.9g},{r.temp:.9g},{r.w:.9g},{r.vd:.9g},{r.vs:.9g},{r.vg:.9g},{r.vb:.9g},{r.vth0:.9g},{r.toxe:.9g},{r.ndep:.9g},{r.u0:.9g},{r.xj:.9g},$&cid,$&cig,$&cis,$&cib,$&cigs,$&cibs" >> ../data/pmos_off_hires_raw.csv',
          "destroy all"]
L += [".endc", ".end"]
(HERE / "netlists" / "pmos_off_hires_ngspice.cir").write_text("\n".join(L))
print("wrote netlist,", len(src), "rows")
