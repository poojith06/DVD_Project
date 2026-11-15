"""PMOS ON data: which rows have tiny currents (1e-13 .. 1e-11 A), and are they the rows with vd and vs both near VDD
(so Vds ~ 0, no channel current)?"""
from pathlib import Path

import numpy as np
import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[1] / "partA_transistor" / "data" / "pmos_on.csv")
n = len(d)
vdd = d.vdd
vds = (d.vd - d.vs).abs()
near = lambda v, tol: (vdd - v) <= tol          # distance below VDD
print(f"PMOS ON rows: {n}")
print("\n1. How many rows have BOTH vd and vs within a tolerance of VDD, and the same for |Vds|:")
for tol in (0.001, 0.01, 0.05, 0.1, 0.2):
    both = (near(d.vd, tol) & near(d.vs, tol))
    print(f"   vd and vs both within {tol * 1000:5.0f} mV of VDD: {both.sum():5d} rows ({100 * both.mean():5.2f} %)")
print("   exact vd == vdd or vs == vdd:", int(((d.vd >= vdd) | (d.vs >= vdd)).sum()))
for tol in (1e-6, 1e-4, 1e-3, 1e-2, 0.05):
    print(f"   |Vds| < {tol:7.0e} V: {int((vds < tol).sum()):5d} rows")

print("\n2. How many rows have each current in a given magnitude band:")
bands = [(0, 1e-14), (1e-14, 1e-13), (1e-13, 1e-12), (1e-12, 1e-11), (1e-11, 1e-10), (1e-10, 1e-9), (1e-9, np.inf)]
rows = []
for c in ("id", "is", "ig", "ib"):
    a = d[c].abs()
    rows.append({"current": c, **{f"{lo:.0e}..{hi:.0e}": int(((a >= lo) & (a < hi)).sum()) for lo, hi in bands}})
print(pd.DataFrame(rows).to_string(index=False))

print("\n3. Rows with |id| in 1e-13..1e-11 A (the 'tiny channel current' rows): what are vd, vs there?")
tiny = d[(d.id.abs() >= 1e-13) & (d.id.abs() < 1e-11)]
print(f"   {len(tiny)} rows; |Vds| median {np.median((tiny.vd - tiny.vs).abs()):.1e} V, max {(tiny.vd - tiny.vs).abs().max():.1e} V")
print(f"   of them with vd and vs both within 100 mV of VDD: {int((near(tiny.vd, 0.1) & near(tiny.vs, 0.1)).sum())}"
      f"; with vd and vs both below VDD/2: {int(((tiny.vd < tiny.vdd / 2) & (tiny.vs < tiny.vdd / 2)).sum())}")
print("   |Vds| / VDD of those rows (percentiles 0, 50, 90, 100):",
      np.percentile(((tiny.vd - tiny.vs).abs() / tiny.vdd), [0, 50, 90, 100]).round(6))

print("\n4. Median |current| by |Vds| (only the channel current id is expected to vanish as Vds -> 0):")
edges = [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 2]
cut = pd.cut(vds, edges, include_lowest=True)
t = d.assign(bin=cut).groupby("bin", observed=True).agg(rows=("id", "size"), id_med=("id", lambda x: x.abs().median()),
                                                       ig_med=("ig", lambda x: x.abs().median()),
                                                       ib_med=("ib", lambda x: x.abs().median()))
print(t.to_string(float_format=lambda x: f"{x:.2e}" if x < 100 else f"{x:.0f}"))

print("\n5. Rows where |id| < 1e-11: fraction of those with vd, vs both within 200 mV of VDD, "
      f"overall: {int((d.id.abs() < 1e-11).sum())} rows")
small = d[d.id.abs() < 1e-11]
print(f"   |Vds| < 1e-3 V in {int(((small.vd - small.vs).abs() < 1e-3).sum())} of them; "
      f"vd,vs both within 200 mV of VDD in {int((near(small.vd, 0.2) & near(small.vs, 0.2)).sum())}")
