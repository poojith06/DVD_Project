"""PMOS ON (gate = 0, body = VDD): tiny currents vs the HIGHER of vd/vs (the effective source of a PMOS)."""
from pathlib import Path

import numpy as np
import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[1] / "partA_transistor" / "data" / "pmos_on.csv")
hi, lo = np.maximum(d.vd, d.vs), np.minimum(d.vd, d.vs)
vth = d.vth0.abs()
print(f"rows: {len(d)}; |vth0| range {vth.min():.3f}..{vth.max():.3f} V (median {vth.median():.3f})")
print("\n1. Rows where the transistor is not actually strongly on: effective source (higher of vd, vs) below |vth0|")
weak = hi < vth
print(f"   hi < |vth0|: {int(weak.sum())} rows ({100 * weak.mean():.1f} %)")
print(f"   hi < 0.2 V : {int((hi < 0.2).sum())} rows;  hi < 0.1 V: {int((hi < 0.1).sum())};  hi < 0.05 V: {int((hi < 0.05).sum())}")
print(f"   (for comparison) hi > VDD - 0.05 V: {int((hi > d.vdd - 0.05).sum())} rows;  lo > VDD - 0.1 V (both near VDD): "
      f"{int((lo > d.vdd - 0.1).sum())} rows")

print("\n2. Median |current| by the higher terminal voltage hi = max(vd, vs):")
edges = [0, 0.1, 0.2, 0.3, 0.46, 0.6, 0.8, 1.0, 1.2]
t = d.assign(bin=pd.cut(hi, edges, include_lowest=True)).groupby("bin", observed=True).agg(
    rows=("id", "size"), id_med=("id", lambda x: x.abs().median()), ig_med=("ig", lambda x: x.abs().median()),
    ib_med=("ib", lambda x: x.abs().median()))
print(t.to_string(float_format=lambda x: f"{x:.2e}" if x < 100 else f"{x:.0f}"))

print("\n3. Where do the 1e-13..1e-11 A currents come from? (rows with that current, split by hi)")
for c in ("id", "ig", "ib"):
    a = d[c].abs()
    m = (a >= 1e-13) & (a < 1e-11)
    print(f"   |{c}| in 1e-13..1e-11 A: {int(m.sum()):5d} rows;  of these hi < 0.3 V: {int((m & (hi < 0.3)).sum()):5d}, "
          f"hi < |vth0|: {int((m & weak).sum()):5d}, hi > VDD - 0.1 V: {int((m & (hi > d.vdd - 0.1)).sum())}")

print("\n4. The Vds = 0 case at the high end (vd and vs both near VDD -> full gate drive but no channel current):")
for tol in (0.01, 0.05, 0.1):
    m = (lo > d.vdd - tol)
    print(f"   both within {tol * 1000:3.0f} mV of VDD: {int(m.sum()):3d} rows; median |id| there {d.id.abs()[m].median() if m.any() else float('nan'):.1e} A, "
          f"median |ig| {d.ig.abs()[m].median() if m.any() else float('nan'):.1e} A")
