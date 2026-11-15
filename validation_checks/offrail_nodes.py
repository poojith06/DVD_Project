"""How far do node voltages sit from the rails (0 and VDD)? The Part C estimate assumes every gate input is exactly 0 or VDD.
Reads the recorded SPICE node voltages only: (1) the internal and output nodes of the CLA (cla3.csv), (2) the first-stage output of AND2/OR2 (Part B).
Output: results/analysis/loading_analysis/offrail_nodes.csv and offrail_nodes.txt"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "analysis" / "loading_analysis"
OUT.mkdir(exist_ok=True)
lines = []
say = lambda s="": (print(s), lines.append(s))
cla = pd.read_csv(REPO / "partC_cla" / "data" / "cla3.csv")
nodes = [c for c in cla.columns if c.startswith("v_")]
dev = pd.DataFrame({c: np.minimum(cla[c].abs(), (cla.vdd - cla[c]).abs()) for c in nodes})
flat = dev.to_numpy().ravel()
say(f"CLA: {len(nodes)} recorded internal / output nodes x {len(cla)} simulations = {flat.size} node voltages")
say(f"  deviation from the nearer rail (V): median {np.median(flat):.2e}, 95th percentile {np.percentile(flat, 95):.2e}, max {flat.max():.2e}")
say(f"  node voltages more than 1 mV from a rail: {100 * (flat > 1e-3).mean():.2f} %; more than 10 mV: {100 * (flat > 1e-2).mean():.2f} %")
per = pd.DataFrame({"node": nodes, "median_V": dev.median().values, "p95_V": dev.quantile(0.95).values, "max_V": dev.max().values})
per.to_csv(OUT / "offrail_nodes.csv", index=False)
say("  worst nodes (max deviation, V): " + ", ".join(f"{r.node} {r.max_V:.2e}" for r in per.sort_values("max_V", ascending=False).head(5).itertuples()))
hot = cla.temp > 85
say(f"  median deviation by temperature: T <= 85 C {np.median(dev[~hot].to_numpy()):.2e} V, T > 85 C {np.median(dev[hot].to_numpy()):.2e} V")
for g in ("and2", "or2"):
    d = pd.read_csv(REPO / "partB_gates" / "data" / f"{g}.csv")
    x = np.minimum(d.v_n.abs(), (d.vdd - d.v_n).abs())
    say(f"Part B {g.upper()} first-stage output v_n (isolated gate): deviation from the nearer rail median {x.median():.2e} V, max {x.max():.2e} V")
(OUT / "offrail_nodes.txt").write_text("\n".join(lines), encoding="utf-8")
