"""Deliverable 3 - data check of the four Part A datasets.

1. Kirchhoff check: id + ig + is + ib for every row (currents are into the terminals, so the sum must be ~0).
   Output kcl_check.csv: median and max |sum| (A) and median / p95 / max of |sum| / max terminal |I| per case.
2. Three trend plots (log y-axis, faint scatter + binned median):
   - OFF current vs temperature (NMOS OFF, PMOS OFF)
   - OFF current vs vth0 (NMOS OFF, PMOS OFF; |vth0| for PMOS)
   - gate current |ig| vs toxe (all four cases)
   "OFF current" = max(|id|, |is|): the drain and source currents swap roles when vs > vd, so the larger of the two
   is the channel (leakage) current in either orientation.
   trend_numbers.csv holds, per plot and case: Spearman correlation, binned-median ratio last/first bin, and the
   multivariate effect (OLS of log10 I on all inputs, so other parameters are held fixed).

    python validation_checks/data_check.py [--out partA_transistor/data_check]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
LABEL = {"nmos_on": "NMOS ON", "nmos_off": "NMOS OFF", "pmos_on": "PMOS ON", "pmos_off": "PMOS OFF"}
INPUTS = ["vdd", "temp", "w", "vd", "vs", "vth0", "toxe", "ndep", "u0", "xj"]
CUR = ["id", "ig", "is", "ib"]


def load(case):
    return pd.read_csv(REPO / "partA_transistor" / "data" / f"{case}.csv")


def kcl_table(data):
    rows = []
    for case, d in data.items():
        c = d[CUR].to_numpy()
        s = np.abs(c.sum(1))
        rel = s / np.abs(c).max(1)
        rows.append({"case": case, "rows": len(d), "median_abs_sum_A": np.median(s), "max_abs_sum_A": s.max(),
                     "median_rel": np.median(rel), "p95_rel": np.percentile(rel, 95), "max_rel": rel.max(),
                     "rows_rel_gt_1e-3": int((rel > 1e-3).sum()), "median_max_terminal_I_A": np.median(np.abs(c).max(1))})
    return pd.DataFrame(rows)


def binned(x, y, bins=20, log=False):
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    centre = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.median(y[idx == b]) if (idx == b).any() else np.nan for b in range(bins)])
    return centre, med


def ols_effect(d, ycol_log, xname, scale):
    """Effect of +scale in input xname on log10 y, all other inputs held fixed (linear regression on all inputs)."""
    X = d[INPUTS].copy()
    X["vds"] = (d.vd - d.vs).abs()
    X = (X - X.mean()) / X.std()
    A = np.c_[np.ones(len(X)), X.to_numpy()]
    coef, *_ = np.linalg.lstsq(A, ycol_log, rcond=None)
    j = list(X.columns).index(xname) + 1
    return coef[j] / d[xname].std() * scale


def scatter(ax, x, y, title, xlabel, ylabel):
    ax.semilogy(x, y, ".", markersize=2, alpha=0.12, color="tab:blue", rasterized=True)
    c, m = binned(x, y)
    ax.semilogy(c, m, "-o", color="crimson", markersize=3, linewidth=1.6, label="binned median (20 bins)")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8)
    return c, m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO / "partA_transistor" / "data_check")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = {c: load(c) for c in CASES}

    # ---------------------------------------------------------------- 1. Kirchhoff
    kcl = kcl_table(data)
    kcl.to_csv(args.out / "kcl_check.csv", index=False)
    pd.set_option("display.width", 220)
    print("1. Kirchhoff check, id + ig + is + ib per row")
    print(kcl.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
    # PMOS OFF: where does the larger relative residual come from?
    d = data["pmos_off"]
    c = d[CUR].to_numpy()
    s, big = np.abs(c.sum(1)), np.abs(c).max(1)
    rel = s / big
    print("\n   PMOS OFF residual vs size of the largest terminal current:")
    for lo, hi in [(0, 1e-15), (1e-15, 1e-14), (1e-14, 1e-12), (1e-12, 1e-9), (1e-9, 1)]:
        m = (big >= lo) & (big < hi)
        if m.any():
            print(f"     max|I| in [{lo:.0e}, {hi:.0e}) A: {int(m.sum()):5d} rows, median |sum| {np.median(s[m]):.1e} A, "
                  f"max |sum| {s[m].max():.1e} A, median rel {np.median(rel[m]):.1e}, max rel {rel[m].max():.1e}")
    print(f"     rows with |sum| > 1e-15 A: {int((s > 1e-15).sum())}; with |sum| > 1e-16 A: {int((s > 1e-16).sum())}")

    # ---------------------------------------------------------------- 2. trends
    numbers = []

    def off_current(d):
        return np.maximum(d["id"].abs(), d["is"].abs()).to_numpy()

    # (a) OFF current vs temperature
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, case in zip(axes, ["nmos_off", "pmos_off"]):
        d = data[case]
        y = off_current(d)
        c, m = scatter(ax, d.temp, y, LABEL[case], "temperature (deg C)", "OFF current max(|id|, |is|) (A)")
        rho = spearmanr(d.temp, np.log10(y)).correlation
        eff = ols_effect(d, np.log10(np.maximum(y, 1e-18)), "temp", 100.0)
        numbers.append({"plot": "OFF current vs temperature", "case": case, "spearman": rho,
                        "binned_median_first_bin_A": m[0], "binned_median_last_bin_A": m[-1], "ratio_last_over_first": m[-1] / m[0],
                        "controlled_effect_decades_per_100C": eff})
    fig.suptitle("OFF current vs temperature (all other parameters drawn at random)")
    fig.tight_layout()
    fig.savefig(args.out / "trend_off_vs_temperature.png", dpi=200)
    plt.close(fig)

    # (b) OFF current vs vth0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, case in zip(axes, ["nmos_off", "pmos_off"]):
        d = data[case]
        y = off_current(d)
        x = d.vth0.abs()
        c, m = scatter(ax, x, y, LABEL[case], "|vth0| (V)" if case.startswith("p") else "vth0 (V)",
                       "OFF current max(|id|, |is|) (A)")
        rho = spearmanr(x, np.log10(y)).correlation
        dd = d.assign(vth0=x)
        eff = ols_effect(dd, np.log10(np.maximum(y, 1e-18)), "vth0", 0.05)
        numbers.append({"plot": "OFF current vs |vth0|", "case": case, "spearman": rho,
                        "binned_median_first_bin_A": m[0], "binned_median_last_bin_A": m[-1], "ratio_last_over_first": m[-1] / m[0],
                        "controlled_effect_decades_per_50mV": eff})
    fig.suptitle("OFF current vs threshold voltage (|vth0| for PMOS)")
    fig.tight_layout()
    fig.savefig(args.out / "trend_off_vs_vth0.png", dpi=200)
    plt.close(fig)

    # (c) gate current vs toxe
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, case in zip(axes, CASES):
        d = data[case]
        y = d.ig.abs().clip(lower=1e-18).to_numpy()
        c, m = scatter(ax, d.toxe * 1e9, y, LABEL[case], "toxe (nm)", "|ig| (A)")
        rho = spearmanr(d.toxe, np.log10(y)).correlation
        eff = ols_effect(d, np.log10(y), "toxe", 1e-10)
        numbers.append({"plot": "|ig| vs toxe", "case": case, "spearman": rho, "binned_median_first_bin_A": m[0],
                        "binned_median_last_bin_A": m[-1], "ratio_last_over_first": m[-1] / m[0],
                        "controlled_effect_decades_per_0.1nm": eff})
    fig.suptitle("Gate current |ig| vs oxide thickness toxe")
    fig.tight_layout()
    fig.savefig(args.out / "trend_ig_vs_toxe.png", dpi=200)
    plt.close(fig)

    # (d) subthreshold subset: effective source within 50 mV of its rail, other terminal > 0.5 V away. There the channel
    #     (subthreshold) current dominates, so the textbook trends (rises with T, falls with vth0) should be sharper.
    for case in ["nmos_off", "pmos_off"]:
        d = data[case]
        if case.startswith("n"):
            near_rail, far = d[["vd", "vs"]].min(axis=1), d[["vd", "vs"]].max(axis=1)
            m = (near_rail < 0.05) & (far > 0.5)
        else:
            near_rail, far = d.vdd - d[["vd", "vs"]].max(axis=1), d.vdd - d[["vd", "vs"]].min(axis=1)
            m = (near_rail < 0.05) & (far > 0.5)
        s = d[m].copy()
        y = np.log10(np.maximum(off_current(s), 1e-18))
        s["vth0"] = s.vth0.abs()
        numbers.append({"plot": "subthreshold subset (source < 50 mV from rail, other > 0.5 V)", "case": case,
                        "rows": int(m.sum()), "spearman": spearmanr(s.temp, y).correlation,
                        "controlled_effect_decades_per_100C": ols_effect(s, y, "temp", 100.0),
                        "controlled_effect_decades_per_50mV": ols_effect(s, y, "vth0", 0.05)})

    num = pd.DataFrame(numbers)
    num.to_csv(args.out / "trend_numbers.csv", index=False)
    print("\n2. Trends (Spearman: monotonic correlation of log10 I with the parameter; ratio: last / first bin median;")
    print("   controlled effect: change of log10 I (decades) for the stated step, other inputs held fixed)")
    print(num.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
