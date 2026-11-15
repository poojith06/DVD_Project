"""Detail for Task 3: which rows are asymmetric (current size) and how much the explicit geometry changes the absolute current."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simlib import printed

S = HERE / "sim" / "cause"
r = pd.read_csv(HERE / "sym_cause.csv")


def read(n):
    t = (S / (n + ".out")).read_text(errors="replace")
    v = dict(printed(t))
    v.update({m.group(1): float(m.group(2)) for m in re.finditer(r"(?m)^\s*(cig|cib)\s*=\s*([-+0-9.eE]+)", t)})
    return np.array([-v["i(vd)"], v["cig"], -v["i(vs)"], v["cib"]])


a = r[(r.variant == "a_as_generated") & (r.set == "Part A rows")]
rows = []
for _, x in a.iterrows():
    o = read(f"A_{x.case}_{int(x.row)}_a_as_generated_orig")
    rows.append({"case": x.case, "dominant_I_A": max(abs(o[0]), abs(o[2])), "rel": x.rel_dominant})
t = pd.DataFrame(rows)
print("Part A rows, variant (a): mirror error of the dominant current vs its size")
for case, g in t.groupby("case"):
    bad = g[g.rel > 1e-2]
    print(f"  {case}: {len(g)} rows; {len(bad)} with mirror error > 1 % (dominant current "
          f"{sorted(float(f'{v:.1e}') for v in bad.dominant_I_A)} A); the other {len(g) - len(bad)} rows: median dominant current "
          f"{g[g.rel <= 1e-2].dominant_I_A.median():.1e} A, max mirror error {g[g.rel <= 1e-2].rel.max():.1e}")
print("\nGate points: effect of the explicit geometry (b) on the absolute dominant current, |I(a)/I(b) - 1|")
for case in ("nmos_off", "pmos_off"):
    d = []
    for c in sorted(set(r[(r.set == "gate points") & (r.case == case)].row)):
        x = read(f"G_{case}_{c}_a_as_generated_orig")
        y = read(f"G_{case}_{c}_b_equal_AD_AS_PD_PS_orig")
        d.append(abs(max(abs(x[0]), abs(x[2])) / max(abs(y[0]), abs(y[2])) - 1))
    print(f"  {case}: median {np.median(d):.1e}, max {max(d):.1e} (20 corners)")
