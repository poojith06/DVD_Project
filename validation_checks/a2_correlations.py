"""Check A2: process parameters must change the currents (the multi-parameter altermod bug gives ~0 correlation).
Pearson correlation of each process parameter with log10|I| in every Part A CSV."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "partA_transistor" / "data"
for f in ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]:
    d = pd.read_csv(DATA / f"{f}.csv")
    log = lambda c: np.log10(np.abs(d[c]).clip(lower=1e-18))
    r = lambda p, c: np.corrcoef(d[p], log(c))[0, 1]
    print(f"{f:20s} rows={len(d)}  toxe~log|ig| {r('toxe', 'ig'):+.3f}  vth0~log|id| {r('vth0', 'id'):+.3f}  "
          f"ndep~log|id| {r('ndep', 'id'):+.3f}  u0~log|id| {r('u0', 'id'):+.3f}  xj~log|id| {r('xj', 'id'):+.3f}  "
          f"xj~log|ib| {r('xj', 'ib'):+.3f}")
