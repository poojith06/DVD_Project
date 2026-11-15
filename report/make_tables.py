"""Generate every table of the report (report/tables/*.tex, booktabs) from the saved result files.
Each file starts with a '% src:' comment naming the file(s) it comes from. Numbers are never typed by hand.
    python report/make_tables.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
F = REPO / "results"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
CL = {"nmos_on": "NMOS ON", "nmos_off": "NMOS OFF", "pmos_on": "PMOS ON", "pmos_off": "PMOS OFF"}
MN = {"gpr": "GPR", "hybrid_analytic_mlp": "Hybrid", "resnet_mlp": "ResNet", "pinn": "PINN", "components_mlp": "Components MLP",
      "components_mlp_vf": "Components MLP (VF)", "ensemble_top3": "Ensemble top-3", "mlp_final": "MLP", "moe": "MoE",
      "ft_transformer": "FT-Transformer", "mlp_multitask": "Multi-task MLP", "hybrid_analytic_lgbm": "Hybrid (LightGBM)"}
P = lambda x: f"{x:.2f}"
P1 = lambda x: f"{x:.1f}"
P3 = lambda x: f"{x:.3f}"
E = lambda x: "\\num{" + f"{x:.2e}" + "}"
E1 = lambda x: "\\num{" + f"{x:.1e}" + "}"
csv = lambda rel, **kw: pd.read_csv(REPO / rel, **kw)


def tex(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#"), ("|", "$|$"), ("<", "$<$"), (">", "$>$")):
        s = s.replace(a, b)
    return s


def write(name, src, colspec, header, rows, midrules=(), note=None):
    """rows: list of lists of already-formatted cells; midrules: row indices after which a \\midrule is added."""
    lines = [f"% src: {src}", "\\begin{tabular}{" + colspec + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        lines.append(" & ".join(r) + " \\\\")
        if i in midrules:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ III: data generation
write("t_cases", "partA_transistor/README.md; partA_transistor/netlists/*_ngspice.cir (rndseed)", "lcccc", ["Case", "$V_g$", "$V_b$", "rndseed", "Rows"],
      [["NMOS ON", "VDD", "0", "13", "10,000"], ["NMOS OFF", "0", "0", "12", "10,000"], ["PMOS ON", "0", "VDD", "14", "10,000"],
       ["PMOS OFF", "VDD", "VDD", "15", "10,000"]])
write("t_sampling", "partA_transistor/README.md (nominal values from 22nm_HP.pm); spec Section 4", "lccc", ["Parameter", "Rule", "NMOS range", "PMOS range"], [
    ["VDD", "nominal $\\pm 10\\,\\%$", "0.9--1.1 V", "0.9--1.1 V"],
    ["$V_d$, $V_s$", "uniform, 0 to VDD", "0--VDD", "0--VDD"],
    ["$W$", "uniform, $1$--$8\\times L_{min}$", "22--176 nm", "22--176 nm"],
    ["Temperature", "uniform", "$-40$--$125\\,^{\\circ}$C", "$-40$--$125\\,^{\\circ}$C"],
    ["vth0 (V)", "nominal $\\pm 10\\,\\%$", "0.4528--0.5534", "$-0.5067$ to $-0.4145$"],
    ["toxe (m)", "nominal $\\pm 10\\,\\%$", "\\num{0.945e-9}--\\num{1.155e-9}", "\\num{0.99e-9}--\\num{1.21e-9}"],
    ["ndep (cm$^{-3}$)", "nominal $\\pm 10\\,\\%$", "\\num{4.95e18}--\\num{6.05e18}", "\\num{3.96e18}--\\num{4.84e18}"],
    ["u0 (m$^2$/Vs)", "nominal $\\pm 10\\,\\%$", "0.036--0.044", "0.00855--0.01045"],
    ["xj (m)", "nominal $\\pm 10\\,\\%$", "\\num{6.48e-9}--\\num{7.92e-9}", "\\num{6.48e-9}--\\num{7.92e-9}"]])

# ------------------------------------------------------------------ EXP-03 / EXP-04
pp = csv("results/analysis/report_numbers/pmos_off_readout_precision.csv", index_col=0)
sel = [("old dataset, ig from -i(vg)", "ig, source current (old)"), ("final dataset, ig from BSIM4", "ig, BSIM4 current (final)"),
       ("old dataset, ib from -i(vb)", "ib, source current (old)"), ("final dataset, ib from BSIM4", "ib, BSIM4 current (final)")]
write("t_pmos_precision", "results/analysis/report_numbers/pmos_off_readout_precision.csv", "lrrrrr",
      ["PMOS OFF readout", "Distinct", "Zeros", "Neg.", "On grid (\\%)", "Median $|I|$ (A)"],
      [[lab, f"{int(pp.loc[k, 'distinct_values']):,}", f"{int(pp.loc[k, 'exact_zeros']):,}", f"{int(pp.loc[k, 'negative']):,}",
        P(pp.loc[k, "on_2.22e-16_grid_pct"]), E(pp.loc[k, "median_abs_A"])] for k, lab in sel])
vr = csv("results/analysis/report_numbers/on_pmos_vds_resolution.csv", dtype={"inputs": str})
write("t_vds_res", "results/analysis/report_numbers/on_pmos_vds_resolution.csv", "llrrr",
      ["Gate", "Inputs", "Median $|V_{ds}|$ (V)", "Rows $<2\\times10^{-5}$\\,V", "Rows exactly 0"],
      [[r.gate.upper(), r.inputs, E1(r["median_|Vds|_V"]) if r["median_|Vds|_V"] > 0 else "0", f"{int(r['rows_|Vds|_below_2e-5V'])}/200",
        f"{int(r['rows_|Vds|_exactly_0_in_csv'])}/200"] for _, r in vr.iterrows()])
a3 = csv("validation_checks/a3_ground_truth.csv")
rows = []
for (part, file), g in a3.groupby(["part", "file"], sort=False):
    cur = max([g[c].max() for c in ("rel_id", "rel_ig", "rel_is", "rel_ib", "rel_leak") if c in g and g[c].notna().any()])
    node = g.max_node_diff_V.max() if "max_node_diff_V" in g and g.max_node_diff_V.notna().any() else np.nan
    rows.append([{"A": "Part A", "B": "Part B", "C": "Part C"}[part], file.upper().replace("_", " ") if part == "A" else file.upper(), E1(cur),
                 E1(node) if not np.isnan(node) else "--"])
write("t_a3", "validation_checks/a3_ground_truth.csv", "llrr", ["Data", "File", "Max rel. diff. (20 rows)", "Max node diff. (V)"], rows, midrules=(3, 8))

# ------------------------------------------------------------------ IV: data check
kc = csv("partA_transistor/data_check/kcl_check.csv")
write("t_kcl", "partA_transistor/data_check/kcl_check.csv", "lrrrrr",
      ["Case", "Median $|\\Sigma I|$ (A)", "Max $|\\Sigma I|$ (A)", "Median rel.", "P95 rel.", "Max rel."],
      [[CL[r.case], E(r.median_abs_sum_A), E(r.max_abs_sum_A), E1(r.median_rel), E1(r.p95_rel), E1(r.max_rel)] for r in kc.itertuples()])
tn = csv("partA_transistor/data_check/trend_numbers.csv")
rows = []
for _, r in tn.iterrows():
    if str(r["plot"]).startswith("subthreshold"):
        rows.append(["OFF, subthreshold subset", CL[r["case"]], P3(r["spearman"]), "--",
                     f"{r['controlled_effect_decades_per_100C']:+.2f} dec/100\\,$^{{\\circ}}$C, {r['controlled_effect_decades_per_50mV']:+.2f} dec/50\\,mV"])
    elif "temperature" in r["plot"]:
        rows.append(["OFF vs temperature", CL[r["case"]], P3(r["spearman"]), f"{r['ratio_last_over_first']:.3g}", f"{r['controlled_effect_decades_per_100C']:+.2f} dec/100\\,$^{{\\circ}}$C"])
    elif "vth0" in r["plot"]:
        rows.append(["OFF vs $|$vth0$|$", CL[r["case"]], P3(r["spearman"]), f"{r['ratio_last_over_first']:.3g}", f"{r['controlled_effect_decades_per_50mV']:+.2f} dec/50\\,mV"])
    else:
        rows.append(["$|i_g|$ vs toxe", CL[r["case"]], P3(r["spearman"]), f"{r['ratio_last_over_first']:.3g}", f"{r['controlled_effect_decades_per_0.1nm']:+.2f} dec/0.1\\,nm"])
write("t_trends", "partA_transistor/data_check/trend_numbers.csv", "llrrl", ["Trend", "Case", "Spearman", "Last/first bin", "Controlled effect"], rows, midrules=(1, 3, 7))
dist = csv("results/analysis/report_numbers/current_magnitude_distribution_pct.csv", index_col=0)
rows = [[tex(i).replace("..", "--").replace(">=", "$\\geq$"), *[P(dist.loc[i, f"{c} |is|"]) for c in CASES]] for i in dist.index]
write("t_dist_is", "results/analysis/report_numbers/current_magnitude_distribution_pct.csv", "lrrrr", ["$|i_s|$ range (A)", "NMOS ON", "NMOS OFF", "PMOS ON", "PMOS OFF"], rows)

# ------------------------------------------------------------------ V: models
v = csv("results/partA/model_comparison_validation.csv")
order = ["gpr", "hybrid_analytic_mlp", "resnet_mlp", "pinn"]
rows, mids = [], []
for case in CASES:
    d = v[v.case == case].set_index("model").loc[order]
    best = d.mean_median_error_pct.idxmin()
    for m in order:
        r = d.loc[m]
        bold = (lambda s: f"\\textbf{{{s}}}") if m == best else (lambda s: s)
        rows.append([CL[case] if m == order[0] else "", bold(MN[m]), bold(f"{r.mean_r2_log:.4f}"), bold(P(r.mean_median_error_pct)), bold(P(r.mean_p95_error_pct)),
                     P(r.id_median_error_pct), P(r.is_median_error_pct)])
    mids.append(len(rows) - 1)
write("t_modelcomp", "results/partA/model_comparison_validation.csv", "llrrrrr",
      ["Case", "Model", "$R^2$ (log, mean)", "Median (\\%, mean)", "P95 (\\%, mean)", "Median $i_d$", "Median $i_s$"], rows, midrules=tuple(mids[:-1]))
rows, mids = [], []
for case in CASES:
    d = v[v.case == case].set_index("model").loc[order]
    for m in order:
        r = d.loc[m]
        rows.append([CL[case] if m == order[0] else "", MN[m]] + [f"{r[f'{c}_r2_log']:.4f}" for c in ("id", "ig", "is", "ib")] + [P(r[f"{c}_median_error_pct"]) for c in ("id", "ig", "is", "ib")] +
                    [P(r[f"{c}_p95_error_pct"]) for c in ("id", "ig", "is", "ib")])
    mids.append(len(rows) - 1)
write("t_modelcomp_full", "results/partA/model_comparison_validation.csv", "ll" + "r" * 12,
      ["Case", "Model"] + [f"$R^2_{{{c}}}$" for c in ("d", "g", "s", "b")] + [f"med {c}" for c in ("$i_d$", "$i_g$", "$i_s$", "$i_b$")] + [f"p95 {c}" for c in ("$i_d$", "$i_g$", "$i_s$", "$i_b$")],
      rows, midrules=tuple(mids[:-1]))
ch = csv("results/partA/chosen_models_test.csv")
write("t_chosen", "results/partA/chosen_models_test.csv", "llrrrrrrrr",
      ["Case", "Model"] + [f"med {c}" for c in ("$i_d$", "$i_g$", "$i_s$", "$i_b$")] + [f"p95 {c}" for c in ("$i_d$", "$i_g$", "$i_s$", "$i_b$")],
      [[CL[r.case], MN[r.model]] + [P(getattr(r, f"{c}_median_error_pct")) for c in ("id", "ig", "is", "ib")] + [P(getattr(r, f"{c}_p95_error_pct")) for c in ("id", "ig", "is", "ib")] for r in ch.itertuples()])
rw = csv("results/partA/rail_weight_tuning.csv")
write("t_railweight", "results/partA/rail_weight_tuning.csv", "rrrrr",
      ["Weight $A$", "Val. score (\\%)", "Near-rail $i_d,i_s$ (\\%)", "NMOS OFF", "PMOS OFF"],
      [[f"{r.rail_weight:g}", P3(r.val_mean_median_pct), P3(r.near_rail_id_is_mean_pct), P3(r.nmos_off_nr_id_is), P3(r.pmos_off_nr_id_is)] for r in rw.itertuples()])
rt = csv("results/partA/resnet_tuning.csv")
limit = 1.05 * rt.loc[rt.run == "tune_A0", "val_score"].iloc[0]
rows = [[tex(r.settings), P3(r.val_score), P3(r.near_rail_id_is), "yes" if r.val_score <= limit else "no"] for r in rt.itertuples()]
write("t_tuning", f"results/partA/resnet_tuning.csv (limit 1.05 x default = {limit:.4f})", "lrrc",
      ["Setting", "Val. score", "Near-rail", "Within limit"], rows)
bm = csv("archive/model_benchmark/results/summary_validation.csv")
tb = bm.pivot_table(index="model", columns="case", values="score mean median %")
tb["mean"] = tb.mean(axis=1)
tb = tb.sort_values("mean").head(12)
write("t_benchmark", "archive/model_benchmark/results/summary_validation.csv", "lrrrrr", ["Model type", "NMOS ON", "NMOS OFF", "PMOS ON", "PMOS OFF", "Mean"],
      [[MN.get(m, tex(m)), P3(r.nmos_on), P3(r.nmos_off), P3(r.pmos_on), P3(r.pmos_off), P3(r["mean"])] for m, r in tb.iterrows()])
ss = pd.read_csv(REPO / "archive/model_benchmark" / "seed_runs" / "summary_mean_std.csv", skiprows=3, header=None)
names = ["case", "model", "split", "current"] + [f"{s}_{a}" for s in ("median", "p95", "MAE", "R2", "sign") for a in ("mean", "std", "min", "max")] + ["n_seeds"]
ss.columns = names
tt_ = ss[(ss.split == "test") & (ss.current == "id")]
rows = []
for m in ["components_mlp_vf", "gpr", "hybrid_analytic_mlp", "pinn", "resnet_mlp"]:
    d = tt_[tt_.model == m].set_index("case")
    rows.append([MN[m]] + [f"{d.loc[c, 'median_mean']:.2f}$\\pm${d.loc[c, 'median_std']:.2f}" for c in CASES])
write("t_seed", "archive/model_benchmark/seed_runs/summary_mean_std.csv", "lrrrr", ["Model", "NMOS ON", "NMOS OFF", "PMOS ON", "PMOS OFF"], rows)
d1 = csv("validation_checks/d1_validation_table.csv")
rows, mids = [], []
for case in CASES:
    d = d1[d1.case == case].sort_values("mean_median_error_pct")
    for r in d.itertuples():
        rows.append([CL[case] if r.Index == d.index[0] else "", MN[r.model] + (" (excluded)" if r.excluded else ""), P(r.mean_median_error_pct), P(r.mean_p95_error_pct)])
    mids.append(len(rows) - 1)
write("t_firstset", "validation_checks/d1_validation_table.csv", "llrr", ["Case", "Model", "Median (\\%, mean of 4 currents)", "P95 (\\%, mean)"], rows, midrules=tuple(mids[:-1]))

# ------------------------------------------------------------------ VI: Part B
g = csv("results/partB/gate_results.csv", dtype={"inputs": str})
write("t_gate", "results/partB/gate_results.csv", "llrrrrrr",
      ["Gate", "Inputs", "SPICE med. (A)", "ML med. (A)", "Bias (\\%)", "Med. err. (\\%)", "P95 err. (\\%)", "Mean acc. (\\%)"],
      [[r.gate.upper(), r.inputs, E(r.spice_median_A), E(r.ml_median_A), f"{r.bias_pct:+.2f}", P(r.median_error_pct), P(r.p95_error_pct), P(r.mean_accuracy_pct)] for r in g.itertuples()],
      midrules=(1, 5, 9, 13))
ps = csv("results/analysis/report_numbers/plain_sum_check.csv")
rows = []
for gate in ["inv", "nand2", "nor2", "and2", "or2", "ALL"]:
    a = ps[(ps.gate == gate) & ps.method.str.startswith("plain")].iloc[0]
    b = ps[(ps.gate == gate) & ps.method.str.startswith("super")].iloc[0]
    rows.append([gate.upper() if gate != "ALL" else "All gates", f"{a.mean_accuracy_pct:.3g}" if abs(a.mean_accuracy_pct) < 1e3 else f"{a.mean_accuracy_pct:,.0f}", f"{int(a.rows_with_negative_leakage)}",
                 P(a.median_error_pct), P(b.mean_accuracy_pct), P(b.median_error_pct)])
write("t_plainsum", "results/analysis/report_numbers/plain_sum_check.csv", "lrrrrr",
      ["Gate", "Plain sum: mean acc. (\\%)", "Neg. rows", "Med. err. (\\%)", "Super-node: mean acc. (\\%)", "Med. err. (\\%)"], rows, midrules=(4,))
ad = csv("results/analysis/report_numbers/accuracy_definitions.csv")
write("t_accdefs", "results/analysis/report_numbers/accuracy_definitions.csv", "lrrr",
      ["Definition (accuracy in \\%)", "Part B (ML)", "Part C (ML)", "Part C (ceiling)"],
      [[["$100-\\overline{|e|}$", "$100-\\mathrm{median}|e|$", "$100-|\\bar e|$", "$100-\\overline{|\\mathrm{med}_{ML}/\\mathrm{med}_{SPICE}-1|}$"][i]] + [P(ad.iloc[i, j]) for j in (1, 2, 3)] for i in range(4)])
fa = csv("results/analysis/report_numbers/family_gate_accuracy.csv")
rows, mids = [], []
for split in fa.split.unique():
    d = fa[fa.split == split]
    for idx, r in d.iterrows():
        rows.append([split.split(" (")[0] if idx == d.index[0] else "", MN[r["family (all four cases)"]], P(r["partB_mean_acc_pct"]), P(r["nand2_00_mean_acc_pct"]), P(r["partB_nand2_mean_acc_pct"]), P(r["partC_mean_acc_pct"])])
    mids.append(len(rows) - 1)
write("t_family", "results/analysis/report_numbers/family_gate_accuracy.csv", "llrrrr",
      ["Split", "Family", "Part B mean acc.", "NAND2 00", "NAND2 all", "Part C mean acc."], rows, midrules=tuple(mids[:-1]))

# ------------------------------------------------------------------ VII: Part C
ce = csv("results/partC/cla3_results.csv")
al = ce[ce.vector == "ALL"].iloc[0]
write("t_cla_overall", "results/partC/cla3_results.csv (row ALL)", "lrr", ["Metric", "ML (sum of ML gates)", "Ceiling (sum of SPICE gates)"], [
    ["SPICE median (A)", E(al.ml_spice_median_A), E(al.ceiling_spice_median_A)], ["Estimate median (A)", E(al.ml_est_median_A), E(al.ceiling_est_median_A)],
    ["Bias (\\%)", f"{al.ml_bias_pct:+.2f}", f"{al.ceiling_bias_pct:+.2f}"], ["Median $|e|$ (\\%)", P(al.ml_median_abs_error_pct), P(al.ceiling_median_abs_error_pct)],
    ["P95 $|e|$ (\\%)", P(al.ml_p95_abs_error_pct), P(al.ceiling_p95_abs_error_pct)], ["Max $|e|$ (\\%)", P(al.ml_max_abs_error_pct), P(al.ceiling_max_abs_error_pct)],
    ["Mean accuracy (\\%)", P(100 - al.ml_mean_abs_error_pct), P(100 - al.ceiling_mean_abs_error_pct)],
    ["Median accuracy (\\%)", P(100 - al.ml_median_abs_error_pct), P(100 - al.ceiling_median_abs_error_pct)]], midrules=(5,))
vec = ce[ce.vector != "ALL"]
lines = [f"% src: results/partC/cla3_results.csv", "\\begin{longtable}{lrrrrr}", "\\toprule",
         "Vector & SPICE med. (A) & ML med. (A) & Bias (\\%) & Med. err. (\\%) & P95 err. (\\%) \\\\", "\\midrule", "\\endhead"]
for r in vec.itertuples():
    lines.append(f"{r.vector.replace('+', '{+}')} & {E(r.ml_spice_median_A)} & {E(r.ml_est_median_A)} & {r.ml_bias_pct:+.2f} & {P(r.ml_median_abs_error_pct)} & {P(r.ml_p95_abs_error_pct)} \\\\")
lines += ["\\bottomrule", "\\end{longtable}"]
(OUT / "t_cla_vectors.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
lc = csv("results/analysis/loading_analysis/cla_ceiling_with_pins.csv")
write("t_ceiling_pins", "results/analysis/loading_analysis/cla_ceiling_with_pins.csv", "lrrrr",
      ["Perfect gate models", "Mean acc. (\\%)", "Median acc. (\\%)", "Bias (\\%)", "P95 $|e|$ (\\%)"],
      [["Sum of gate leakages (ceiling)", *[P(lc.iloc[0, j]) if j != 3 else f"{lc.iloc[0, j]:+.2f}" for j in (1, 2, 3, 4)]],
       ["+ pin currents of driven inputs at 1", *[P(lc.iloc[1, j]) if j != 3 else f"{lc.iloc[1, j]:+.2f}" for j in (1, 2, 3, 4)]]])
pc = csv("results/analysis/loading_analysis/pin_current_shares.csv", dtype={"inputs": str})
write("t_pin_shares", "results/analysis/loading_analysis/pin_current_shares.csv", "llrrrr",
      ["Gate", "Inputs", "Supplied med. (\\%)", "Supplied max (\\%)", "Absorbed med. (\\%)", "Absorbed max (\\%)"],
      [[r.gate.upper(), r.inputs] + [f"{abs(getattr(r, c)):.3g}" for c in ("median_supplied_over_leak_pct", "max_supplied_over_leak_pct", "median_absorbed_over_leak_pct", "max_absorbed_over_leak_pct")] for r in pc.itertuples()],
      midrules=(1, 5, 9, 13))
cb = csv("results/phase2_comparison/comparison.csv")
rows = []
for r in cb.itertuples():
    f2 = lambda x: "--" if pd.isna(x) else (f"{x:+.2f}" if "Bias" in r.metric else P(x))
    rows.append([tex(r.metric).replace("$|$", "$|$"), f2(r.set_a_GPR_ON_resnet_OFF), f2(r.set_b_GPR_ON_components_OFF), f2(r.ceiling_perfect_gates)])
write("t_comparison", "results/phase2_comparison/comparison.csv", "lrrr", ["Metric (\\%)", "Set (a)", "Set (b)", "Ceiling"], rows, midrules=(1,))

# ------------------------------------------------------------------ VIII: error analysis
bb = csv("results/analysis/report_numbers/error_breakdown_partB.csv")
bc = csv("results/analysis/report_numbers/error_breakdown_partC.csv")
rows = []
for (var, rng), rb in bb.groupby(["variable", "range"], sort=False):
    rc = bc[(bc.variable == var) & (bc.range == rng)].iloc[0]
    label = rng.replace("(", "").replace("]", "").replace("-40.001", "-40").replace(", ", "--")
    rows.append([tex(var) if rng == bb[bb.variable == var].range.iloc[0] else "", tex(label), P(rb.mean_abs_error_pct_ml.iloc[0] if hasattr(rb.mean_abs_error_pct_ml, "iloc") else rb.mean_abs_error_pct_ml),
                 P(rc.mean_abs_error_pct_ml), P(rc.mean_abs_error_pct_ceiling), f"{rc.bias_pct_ml:+.2f}", f"{rc.bias_pct_ceiling:+.2f}"])
mids = [i for i in range(len(rows) - 1) if rows[i + 1][0] != ""]
write("t_breakdown", "results/analysis/report_numbers/error_breakdown_partB.csv, error_breakdown_partC.csv", "llrrrrr",
      ["Variable", "Range", "B: mean $|e|$", "C: mean $|e|$ ML", "C: ceiling", "C: bias ML", "C: bias ceil."], rows, midrules=tuple(mids))
gr = csv("results/analysis/report_numbers/gate_point_replay_summary.csv").set_index("case")
write("t_gatepoint", "results/analysis/report_numbers/gate_point_replay_summary.csv", "lrr", ["Inverter operating point, 200 corners", "NMOS OFF", "PMOS OFF"], [
    ["Dominant current, median (A)", E(gr.loc["nmos_off", "dominant_current_median_A"]), E(gr.loc["pmos_off", "dominant_current_median_A"])],
    ["Median $|e|$ of $i_d$ (\\%)", P(gr.loc["nmos_off", "median_|err|_id_%"]), P(gr.loc["pmos_off", "median_|err|_id_%"])],
    ["Median $|e|$ of $i_s$ (\\%)", P(gr.loc["nmos_off", "median_|err|_is_%"]), P(gr.loc["pmos_off", "median_|err|_is_%"])],
    ["Dominant current: median $|e|$ (\\%)", P(gr.loc["nmos_off", "dominant_current_median_|err|_%"]), P(gr.loc["pmos_off", "dominant_current_median_|err|_%"])],
    ["Dominant current: P95 $|e|$ (\\%)", P(gr.loc["nmos_off", "dominant_p95_|err|_%"]), P(gr.loc["pmos_off", "dominant_p95_|err|_%"])],
    ["Dominant current: bias (\\%)", f"{gr.loc['nmos_off', 'dominant_bias_%']:+.2f}", f"{gr.loc['pmos_off', 'dominant_bias_%']:+.2f}"],
    ["Part A test median $|e|$, $i_d$ (\\%)", P(gr.loc["nmos_off", "PartA_test_median_%_id"]), P(gr.loc["pmos_off", "PartA_test_median_%_id"])],
    ["Part A test median $|e|$, $i_s$ (\\%)", P(gr.loc["nmos_off", "PartA_test_median_%_is"]), P(gr.loc["pmos_off", "PartA_test_median_%_is"])]], midrules=(0, 5))
rp = csv("results/analysis/report_numbers/rail_point_row_counts.csv")
write("t_railcounts", "results/analysis/report_numbers/rail_point_row_counts.csv", "llrrr", ["Case", "Split", "Rows", "Source within 20\\,mV of rail, $|V_{ds}|>0.9$\\,VDD", "$|V_{ds}|>0.9$\\,VDD"],
      [[CL[r["case"]] if r["split"] == "train" else "", r["split"], f"{int(r['rows']):,}", f"{int(r['source_within_20mV_of_rail_and_|Vds|>0.9VDD'])}",
        f"{int(r['|Vds|>0.9VDD'])} ({r['|Vds|>0.9VDD_pct']:.2f}\\%)"] for _, r in rp.iterrows()], midrules=(2,))
sc = csv("validation_checks/sym_cause.csv")
rows = []
for (s, case), gg in sc.groupby(["set", "case"], sort=False):
    row = [s.replace("Part A rows", "Part A rows").replace("gate points", "Gate points"), CL[case]]
    for var in ("a_as_generated", "b_equal_AD_AS_PD_PS", "c_geomod0"):
        x = gg[gg.variant == var].rel_dominant
        row.append(E1(x.max()) if x.max() > 0 else "0")
    rows.append(row)
write("t_sym", "validation_checks/sym_cause.csv", "llrrr", ["Bias points", "Case", "(a) as generated", "(b) equal AD/AS/PD/PS", "(c) geomod = 0"], rows,
      midrules=(1,))
print("tables written:", len(list(OUT.glob("*.tex"))))
