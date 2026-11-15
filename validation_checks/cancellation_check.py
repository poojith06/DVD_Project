"""Does the ML sum differ systematically from the ceiling sum (error cancellation in Part C)?
Reads results/partC/cla3_predictions.csv and partB_gates/gate_predictions.csv (saved predictions only).
Output: results/analysis/loading_analysis/cancellation_check.txt"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
F = REPO / "results"
c = pd.read_csv(F / "partC" / "cla3_predictions.csv")
d = 100 * (c.leak_ml - c.leak_ceiling) / c.leak_ceiling
lines = [f"Part C, relative difference of the ML sum to the ceiling sum, (ML - ceiling) / ceiling in percent, 12,800 rows:",
         f"  median {d.median():.3f}, mean {d.mean():.3f}, 5th percentile {d.quantile(0.05):.3f}, 95th percentile {d.quantile(0.95):.3f}"]
c["band"] = pd.cut(c.temp, [-40, 0, 40, 85, 125], include_lowest=True)
for k, v in d.groupby(c.band, observed=True).median().items():
    lines.append(f"  temperature {k}: median {v:.3f}")
b = pd.read_csv(F / "partB" / "gate_predictions.csv")
e = 100 * (b.leak_ml - b.leak_spice) / b.leak_spice
lines.append(f"Part B, signed gate error (ML - SPICE) / SPICE in percent, 3,600 rows: median {e.median():.3f}, mean {e.mean():.3f}")
(F / "analysis" / "loading_analysis" / "cancellation_check.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
