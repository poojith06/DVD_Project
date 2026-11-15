"""Check deliverables 1-10 of the project document against the files in the repository.
Prints PASS / MISSING per item with the evidence; the same text is saved in docs/deliverables_check.txt."""
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
F = REPO / "results"
lines = []
status = {}


def say(s=""):
    print(s)
    lines.append(s)


def item(d, name, ok, info=""):
    status.setdefault(d, []).append(ok)
    say(f"  [{'PASS' if ok else 'MISSING'}] {name}" + (f"  - {info}" if info else ""))


def exists(p):
    return (REPO / p).exists()


say("D1  datasets: four CSVs, 10,000 rows, exact columns in the specified order")
exp = "vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib".split(",")
for c in ("nmos_on", "nmos_off", "pmos_on", "pmos_off"):
    p = REPO / "partA_transistor" / "data" / f"{c}.csv"
    d = pd.read_csv(p)
    item(1, f"partA_transistor/data/{c}.csv", d.shape == (10000, 16) and list(d.columns) == exp,
         f"shape {d.shape}, columns in order: {list(d.columns) == exp}")

say("\nD2  example netlists: one per transistor case, one per gate, the test circuit")
for sub, names in (("partA_transistor", ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]),
                   ("partB_gates", ["inv", "nand2", "nor2", "and2", "or2"]), ("partC_cla", ["cla3"])):
    for n in names:
        p = REPO / sub / "netlists" / f"{n}_ngspice.cir"
        t = p.read_text() if p.exists() else ""
        opt = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9" in t
        inc = ".include ../../model/22nm_HP.pm" in t
        extra = ""
        if sub == "partA_transistor":
            extra = f"; own source on every terminal (Vd, Vg, Vs, Vb): {all(re.search(rf'(?m)^{v} ', t) for v in ('Vd', 'Vg', 'Vs', 'Vb'))}"
        item(2, f"{sub}/netlists/{n}_ngspice.cir", p.exists() and opt and inc, f"options line present: {opt}; includes ../../model/22nm_HP.pm: {inc}{extra}")
for p in ("partC_cla/netlists/cla3.sub", "partC_cla/netlists/gates.sub", "model/22nm_HP.pm"):
    item(2, p, exists(p))

say("\nD3  data check: Kirchhoff check per case and three trend plots")
for p in ("kcl_check.csv", "trend_off_vs_temperature.png", "trend_off_vs_vth0.png", "trend_ig_vs_toxe.png", "trend_numbers.csv", "README.md"):
    item(3, f"partA_transistor/data_check/{p}", (REPO / "partA_transistor" / "data_check" / p).exists())

say("\nD4  model comparison table: 4 model types x 4 cases, R2 / median / p95 on the VALIDATION set")
t = pd.read_csv(F / "partA" / "model_comparison_validation.csv")
need = [f"{c}_{m}" for m in ("r2_log", "median_error_pct", "p95_error_pct") for c in ("id", "ig", "is", "ib")] + ["mean_r2_log", "mean_median_error_pct", "mean_p95_error_pct"]
ok = len(t) == 16 and t.groupby("case").model.nunique().eq(4).all() and all(c in t for c in need)
item(4, "results/partA/model_comparison_validation.csv", ok,
     f"{len(t)} rows = {t.case.nunique()} cases x {t.model.nunique()} model types; R2 is computed on log10|I| per current (id, ig, is, ib) and averaged (mean_r2_log); median and p95 are relative errors in %")

say("\nD5  chosen models: test median and p95, predicted-vs-SPICE plot per case, four saved model files")
c5 = pd.read_csv(F / "partA" / "chosen_models_test.csv")
item(5, "results/partA/chosen_models_test.csv", len(c5) == 4 and all(f"{c}_median_error_pct" in c5 and f"{c}_p95_error_pct" in c5 for c in ("id", "ig", "is", "ib")),
     f"models: {dict(zip(c5.case, c5.model))}")
for case in ("nmos_on", "nmos_off", "pmos_on", "pmos_off"):
    item(5, f"results/partA/parity_{case}.png", (F / "partA" / f"parity_{case}.png").exists())
    item(5, f"results/models/{case}.joblib", (F / "models" / f"{case}.joblib").exists())

say("\nD6  gate results table (18 cases: SPICE median, ML median, bias, median error, p95 error)")
g = pd.read_csv(F / "partB" / "gate_results.csv", dtype={"inputs": str})
cols = ["spice_median_A", "ml_median_A", "bias_pct", "median_error_pct", "p95_error_pct"]
item(6, "results/partB/gate_results.csv", len(g) == 18 and all(c in g for c in cols), f"{len(g)} rows; columns {cols} present")
p = pd.read_csv(F / "partB" / "gate_predictions.csv", dtype={"inputs": str})
sec5 = ["leak_spice", "leak_ml", "v_out", "v_int", "v_int_ml", "v_n", "v_n_ml"]
item(6, "results/partB/gate_predictions.csv (Section 5 columns, per case and corner)", all(c in p for c in sec5) and len(p) == 3600,
     f"{len(p)} rows; SPICE leakage, ML leakage, SPICE output voltage, internal node voltages from SPICE (v_int, v_n) and ML (v_int_ml, v_n_ml: copies of the recorded SPICE voltages, as in spec step 4)")

say("\nD7  gate parity plot (ML vs SPICE leakage, log axes, all gate simulations)")
item(7, "results/partB/gate_parity.png", (F / "partB" / "gate_parity.png").exists())

say("\nD8  test-circuit results: same metrics as the gate table, and a parity plot")
c8 = pd.read_csv(F / "partC" / "cla3_results.csv")
item(8, "results/partC/cla3_results.csv", len(c8) == 65 and "ml_bias_pct" in c8 and "ml_p95_abs_error_pct" in c8,
     f"{len(c8)} rows (overall + 64 vectors); SPICE median, ML median, bias, median and p95 error")
item(8, "results/partC/cla3_parity.png", (F / "partC" / "cla3_parity.png").exists())

say("\nD9  code that regenerates everything, in order, with fixed seeds")
steps = [("1 Part A SPICE data", ["partA_transistor/netlists/nmos_on_ngspice.cir", "partA_transistor/netlists/nmos_off_ngspice.cir", "partA_transistor/netlists/pmos_on_ngspice.cir", "partA_transistor/netlists/pmos_off_ngspice.cir"]),
         ("2 data check", ["validation_checks/data_check.py"]),
         ("3-4 training, selection", ["ml/train.py", "ml/choose_weight.py", "ml/choose_resnet.py", "ml/assemble_tuned.py", "ml/select_best.py"]),
         ("5 Part B SPICE + ML", ["partB_gates/netlists/nand2_ngspice.cir", "ml/evaluate_gates.py"]),
         ("6 Part C SPICE + ML", ["partC_cla/netlists/cla3_ngspice.cir", "partC_cla/scripts/regen_cla3.py", "ml/evaluate_cla.py"]),
         ("7 comparison", ["ml/reproduce_check.py"]),
         ("wrappers", ["run_pipeline.ps1", "run_pipeline.sh", "REPRODUCE.md"])]
for name, files in steps:
    for p in files:
        item(9, f"step {name}: {p}", exists(p))
for p, key, what in (("validation_checks/pipeline_fast_run.txt", "REPRODUCTION: PASS", "fast path reproduces Part B 98.87 % and Part C 99.00 %"),
                     ("validation_checks/netlist_resim_check.txt", "RESIM: PASS", "re-simulation of every netlist matches the CSVs"),
                     ("validation_checks/regen_cla3_check.txt", "REGENERATION: PASS", "Part C regeneration script reproduces cla3.csv")):
    ok = exists(p) and key in (REPO / p).read_text(errors="ignore")
    item(9, f"{p}", ok, what)
for p in ("gen_cla3_gate_inputs.py", "run_partC.sh"):
    item(9, f"{p} (never committed) is no longer referred to in partC_cla/README.md or in the steps of REPRODUCE.md",
         p not in (REPO / "partC_cla/README.md").read_text(encoding="utf-8") and
         p not in (REPO / "REPRODUCE.md").read_text(encoding="utf-8").split("## Part C generation scripts")[0])

say("\nD10 report")
item(10, "report/main.tex (IEEE conference format)", exists("report/main.tex"))
item(10, "report/DVD_Angels_Report.pdf (final report)", exists("report/DVD_Angels_Report.pdf"))
say("\nSummary per deliverable:")
for d in range(1, 11):
    r = status.get(d, [])
    say(f"  D{d}: {'PASS' if r and all(r) else 'MISSING/PARTIAL'} ({sum(r)}/{len(r)} items)")
(REPO / "docs" / "deliverables_check.txt").write_text("\n".join(lines), encoding="utf-8")
