"""Data statistics for the report, written to results/analysis/report_numbers/ (analysis only):
  EXP-03  PMOS OFF ig / ib readout: source branch current (rounded to ~2.2e-16 A) vs BSIM4 internal currents
  EXP-04  6-significant-digit CSV output: relative rounding bound, and how well the ON-PMOS Vds inside gates can be
          resolved from the recorded node voltages
  EXP-06  current-magnitude distributions (percent of rows per decade) for id, ig, is, ib in all four cases
  EXP-10  how many training rows lie at the gate operating points (source on its rail, |Vds| ~ VDD)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "ml"))
from pipeline import split  # noqa: E402

Q = 2.0 ** -52          # ~2.22e-16: spacing of the double-precision solve for currents of order 1
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "results" / "analysis" / "report_numbers")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    # ------------------------------------------------------------------ EXP-03
    say("EXP-03  PMOS OFF gate/body current readout")
    old = pd.read_csv(REPO / "partA_transistor" / "pmos_off_readout_check" / "data" / "pmos_off_old_readout.csv")
    new = pd.read_csv(REPO / "partA_transistor" / "data" / "pmos_off.csv")
    ro = pd.read_csv(REPO / "partA_transistor" / "pmos_off_readout_check" / "data" / "pmos_off_source_readout.csv")
    # the readout-check run (same netlist, rndseed 15) wrote a copy of pmos_off.csv; the copy was byte-identical to
    # partA_transistor/data/pmos_off.csv (verified when the repository was restructured) and was removed
    say("  the readout-check run's copy of pmos_off.csv was byte-identical to partA_transistor/data/pmos_off.csv (removed as a duplicate)")

    def stats(x):
        x = np.asarray(x, float)
        on_grid = np.abs(x / Q - np.round(x / Q)) < 1e-6
        return {"rows": len(x), "distinct_values": len(np.unique(x)), "exact_zeros": int((x == 0).sum()),
                "negative": int((x < 0).sum()), "on_2.22e-16_grid_pct": 100 * on_grid.mean(),
                "abs_below_5e-15_pct": 100 * (np.abs(x) < 5e-15).mean(), "median_abs_A": float(np.median(np.abs(x)))}
    tab = pd.DataFrame({"old dataset, ig from -i(vg)": stats(old.ig), "same rows' ig from -i(vg) (source readout)": stats(ro.ig_source),
                        "final dataset, ig from BSIM4": stats(new.ig),
                        "old dataset, ib from -i(vb)": stats(old.ib), "same rows' ib from -i(vb) (source readout)": stats(ro.ib_source),
                        "final dataset, ib from BSIM4": stats(new.ib)}).T
    tab.to_csv(args.out / "pmos_off_readout_precision.csv")
    say(tab.to_string(float_format=lambda x: f"{x:.4g}"))
    d = (ro.ig_source - new.ig).abs()
    say(f"  |ig(source readout) - ig(BSIM4)| over the same rows: median {d.median():.2e} A, max {d.max():.2e} A "
        f"(grid spacing {Q:.2e} A); rows where the source readout is zero but BSIM4 ig is not: {int(((ro.ig_source == 0) & (new.ig != 0)).sum())}")
    big = new.ig.abs() > 1e-13
    say(f"  rows with |ig| > 1e-13 A (resolvable by the source readout): {int(big.sum())}; there max relative difference "
        f"{float((d[big] / new.ig.abs()[big]).max()):.2e}")

    # ------------------------------------------------------------------ EXP-04
    say("\nEXP-04  6-significant-digit output")
    say(f"  rounding to 6 significant digits changes a value by at most 5e-6 relative (half a unit in the 6th digit, worst case at "
        f"mantissa 1.00000: 5e-6).")
    pairs = []
    for g in ["inv", "nand2", "nor2", "and2", "or2"]:
        gd = pd.read_csv(REPO / "partB_gates" / "data" / f"{g}.csv")
        gd["inputs"] = gd.A.astype(int).astype(str) + (gd.B.astype(int).astype(str) if "B" in gd else "")
        pairs.append((g, gd))
    res = []
    for g, gd in pairs:
        for inp, grp in gd.groupby("inputs"):
            out_high = grp.v_out > grp.vdd / 2
            if g == "inv" and inp == "0":
                vds = grp.vdd - grp.v_out
            elif g == "nand2" and inp == "00":
                vds = grp.vdd - grp.v_out                      # both PMOS ON, source = vdd, drain = out
            elif g == "nor2" and inp == "00":
                vds = grp.vdd - grp.v_int                      # series stack: Mpa source vdd, drain y (both PMOS ON)
            else:
                continue
            vds = vds.abs()
            res.append({"gate": g, "inputs": inp, "device": "ON PMOS (source at VDD)", "rows": len(grp), "median_|Vds|_V": vds.median(),
                        "min_|Vds|_V": vds.min(), "rows_|Vds|_below_2e-5V": int((vds < 2e-5).sum()),
                        "rows_|Vds|_exactly_0_in_csv": int((vds == 0).sum())})
    r = pd.DataFrame(res)
    r.to_csv(args.out / "on_pmos_vds_resolution.csv", index=False)
    say("  ON-PMOS |Vds| as it can be derived from the recorded 6-digit node voltages (vdd - node):")
    say(r.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    say("  a difference of two 6-digit numbers near 1 V is resolved to about 1e-5 V; |Vds| below that cannot be recovered.")

    # ------------------------------------------------------------------ EXP-06
    say("\nEXP-06  current-magnitude distribution (percent of the 10,000 rows per case)")
    edges = [0, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, np.inf]
    labels = ["<1e-15", "1e-15..1e-14", "1e-14..1e-13", "1e-13..1e-12", "1e-12..1e-11", "1e-11..1e-10", "1e-10..1e-9",
              "1e-9..1e-8", "1e-8..1e-7", "1e-7..1e-6", "1e-6..1e-5", "1e-5..1e-4", ">=1e-4"]
    dist = {}
    for c in CASES:
        dd = pd.read_csv(REPO / "partA_transistor" / "data" / f"{c}.csv")
        for cur in ("id", "ig", "is", "ib"):
            dist[(c, cur)] = 100 * pd.cut(dd[cur].abs(), edges, right=False, labels=labels).value_counts(normalize=True).reindex(labels)
    D = pd.DataFrame(dist)
    D.columns = [f"{c} |{cur}|" for c, cur in D.columns]
    D.to_csv(args.out / "current_magnitude_distribution_pct.csv")
    say(D[[f"{c} |is|" for c in CASES]].to_string(float_format=lambda x: f"{x:.2f}"))
    pon = pd.read_csv(REPO / "partA_transistor" / "data" / "pmos_on.csv")
    hi = np.maximum(pon.vd, pon.vs)
    lo = np.minimum(pon.vd, pon.vs)
    tiny = (pon.id.abs() >= 1e-13) & (pon.id.abs() < 1e-11)
    say(f"  PMOS ON: {int(tiny.sum())} rows with |id| in 1e-13..1e-11 A; of them {int((tiny & (hi < pon.vth0.abs())).sum())} have the higher of vd, vs below |vth0| "
        f"(subthreshold), {int((tiny & (lo > pon.vdd - 0.1)).sum())} have both terminals within 100 mV of VDD; "
        f"rows with both terminals within 100 mV of VDD overall: {int((lo > pon.vdd - 0.1).sum())} ({100 * (lo > pon.vdd - 0.1).mean():.2f} %)")

    # ------------------------------------------------------------------ EXP-10
    say("\nEXP-10  training rows at the gate operating points (80:10:10 split, seed 42; 8,000 training rows per case)")
    rows = []
    for c in ("nmos_off", "pmos_off"):
        dd = pd.read_csv(REPO / "partA_transistor" / "data" / f"{c}.csv")
        tr, va, te = split(dd, (0.8, 0.1, 0.1), 42)
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            if c.startswith("n"):
                src, other = part[["vd", "vs"]].min(axis=1), part[["vd", "vs"]].max(axis=1)
                gap = src
            else:
                src, other = part[["vd", "vs"]].max(axis=1), part[["vd", "vs"]].min(axis=1)
                gap = part.vdd - src
            vds = (part.vd - part.vs).abs() / part.vdd
            rows.append({"case": c, "split": name, "rows": len(part),
                         "source_within_20mV_of_rail_and_|Vds|>0.9VDD": int(((gap < 0.02) & (vds > 0.9)).sum()),
                         "source_within_50mV_of_rail_and_|Vds|>0.8VDD": int(((gap < 0.05) & (vds > 0.8)).sum()),
                         "|Vds|>0.9VDD": int((vds > 0.9).sum()), "|Vds|>0.9VDD_pct": 100 * (vds > 0.9).mean()})
    R = pd.DataFrame(rows)
    R.to_csv(args.out / "rail_point_row_counts.csv", index=False)
    say(R.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    (args.out / "report_data_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
