# Final models and results (PTM 22 nm HP, VDD = 1 V) — `final_v2`

Everything produced by the final pipeline and by the validation-check task (Phase 1 checks, Phase 2 Part C), in one
folder. Scripts are **not** copied here; they stay in their repo folders (listed at the bottom) so their relative
paths keep working. `Final_models/models`, `results`, `logs` and `train_final.py` next to this folder are the
earlier (2026-11-10) model set and are not part of this submission.

## 1. The final transistor models — `models/`

| File | Case | Model type | Test median error (id / ig / is / ib) |
|---|---|---|---|
| `nmos_on.joblib` | NMOS ON | hybrid analytic + MLP | 0.18 / 0.19 / 0.16 / 0.21 % |
| `nmos_off.joblib` | NMOS OFF | ResNet MLP, 5 residual blocks | 0.31 / 0.04 / 0.31 / 0.28 % |
| `pmos_on.joblib` | PMOS ON | Gaussian process (GPR) | 0.22 / 0.03 / 0.10 / 0.17 % |
| `pmos_off.joblib` | PMOS OFF | ResNet MLP, 5 residual blocks | 0.45 / 0.09 / 0.59 / 0.46 % |

How they were chosen (validation data only; Part B and Part C data were never used for training, tuning or choice):

1. Four spec-compliant model types (GPR, hybrid analytic + MLP, ResNet MLP, PINN) trained on the four dataset
   currents only, split **80 / 10 / 10** (seed 42), 8,000 training rows per case.
2. Near-rail loss weighting tried (weights 0, 10, 30); validation chose **0** (no weighting).
3. ResNet settings tuned for the two OFF cases (7 single changes from the defaults, `partA_models/resnet_tuning.csv`).
   Rule fixed before the runs: lowest validation error on near-rail rows (both terminals within 0.2 V of a rail)
   among settings whose normal validation score is within 5 % of the default. Winner: **5 blocks**.
4. Per case, the model type with the lowest mean validation median error (mean p95 breaks ties) is the final model
   (`partA_models/model_comparison_validation.csv`, 4 types x 4 cases, R², median and p95 on validation).

Using a model (needs `ML_Models_v2` on the import path, because the files are pickled objects of its classes):

```python
import sys; sys.path.insert(0, "ML_Models_v2")
from pipeline import TransistorSurrogate
model = TransistorSurrogate.load("Final_models/final_v2/models/nmos_off.joblib")
currents = model.predict(df)   # df: columns vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj  ->  id, ig, is, ib in amps
```

## 2. Results of the final models

| | Part B (3,600 gate simulations) | Part C (12,800 CLA simulations) | Part C ceiling (perfect gate models) |
|---|---|---|---|
| Mean accuracy, 100 − mean \|e\| | **98.87 %** | **99.00 %** | 99.16 % |
| Median accuracy, 100 − median \|e\| | 99.18 % | 99.17 % | 99.34 % |
| 100 − \|mean signed e\| | – | 99.96 % | 99.16 % |
| Mean over vectors of 100 − \|ML median / SPICE median − 1\| | – | 99.44 % | 99.23 % |
| p95 \|error\| | 3.24 % | 2.57 % | 2.11 % |
| Bias (median signed error) | – | −0.23 % | −0.66 % |
| Worst case | 8.10 % | +4.02 % (vector 1+7, corner 39, 122 °C) | 3.83 % |

Part B per gate (mean accuracy): INV 98.47 %, NAND2 99.02 %, NOR2 98.96 %, AND2 98.84 %, OR2 98.85 %.
Part C is the plain sum of the 33 Part B ML gate leakages (27 NAND2, 3 INV, 2 AND2, 1 OR2); the "ceiling" is the
same sum built from the Part B SPICE gate leakages and shows what the summing itself costs.

Before the ResNet tuning (same pipeline, default ResNet): Part B 98.89 % / Part C 98.89 % mean accuracy
(`reference_before_tuning/`). Tuning changed Part B by −0.02 % and Part C by +0.11 %.

## 3. Folder contents

| Folder | Spec deliverable | Contents |
|---|---|---|
| `models/` | 5 | the four final model files |
| `partA_models/` | 4, 5 | `model_comparison_validation.csv` (4 types x 4 cases), `chosen_models_test.csv`, `parity_<case>.png`, `validation_comparison_<case>.csv`, `resnet_tuning.csv`, `rail_weight_tuning.csv` |
| `partB_gates/` | 6, 7 | `gate_results.csv` (18 cases: SPICE median, ML median, bias, median / p95 error), `gate_parity.png`, `gate_predictions.csv` (every simulation) |
| `partC/` | 8 | `cla3_results.csv` (overall and per vector), `cla3_parity.png`, `cla3_predictions.csv`, `cla3_gates.csv`, `summary.csv` |
| `phase1_checks/` | – | outputs of the Phase 1 checks; start with `PHASE1_SUMMARY.md` |
| `phase2_partC_comparison/` | – | Part C for the two model sets of the validation task: (a) GPR ON + ResNet OFF, (b) GPR ON + components MLP OFF, with `comparison.csv` and their Part B predictions |
| `reference_before_tuning/` | – | the results before the ResNet tuning |

## 4. Scripts that produce all of this (they stay in their repo folders)

| Script | Purpose |
|---|---|
| `ML_Models_v2/models.py`, `pipeline.py`, `train.py`, `choose_weight.py`, `choose_resnet.py`, `assemble_tuned.py`, `select_best.py` | models, training, selection on validation |
| `ML_Models_v2/run_parallel.ps1`, `tune_resnet.ps1`, `finish_tuned.ps1` | the runs that were executed (Windows) |
| `ML_Models_v2/evaluate_gates.py` | Part B |
| `ML_Models/evaluate_cla.py` | Part C (`--predictions` Part B predictions CSV, `--out` folder) |
| `Validation_checks/*.py` | Phase 1 checks (`a1_a4_scan`, `a2_correlations`, `a3_ground_truth`, `b_consistency`, `c1_kcl_method`, `sym_check`) |

Re-run Part C on the final models:

```powershell
python ML_Models\evaluate_cla.py --predictions ML_Models_v2\results_tuned\gate_results\gate_predictions.csv --out Final_models\final_v2\partC
```

## 5. Notes for the report

- **Gate leakage** is the current drawn from VDD (as the SPICE netlists record it, `−i(vsup)`), assembled from the
  predicted terminal currents with Kirchhoff's current law on the supply node (VDD plus every node joined to it
  through an ON PMOS). Summing the currents at the VDD terminals alone fails because an ON PMOS inside a gate has
  almost no voltage across it. `phase1_checks/c1_kcl_method.txt` shows the method is exact (8e-6) with real SPICE
  currents.
- **Open questions for the TA:** the exact definition of "mean accuracy"; whether gate-input pin currents belong in
  the gate leakage; PMOS OFF `ig`/`ib` are read from BSIM4's internal currents (the source branch currents are
  rounded to ~2.2e-16 A) while every terminal still has its own voltage source.
- **No drain–source symmetry preprocessing:** `phase1_checks/sym_check.txt` shows the OFF transistors of this model
  card are not symmetric in ngspice, so terminals are used as simulated.
- **Simulator:** ngspice-44.2. The random stream behind `rndseed` differs between ngspice builds, so the corners are
  fixed by the recorded CSVs.
- **Files missing in the working tree** but tracked in git: `Gates_Part-B/netlists/22nm_HP.pm`,
  `PartC_3bit CLA adder/netlists/22nm_HP.pm`, `PartC_3bit CLA adder/netlists/cla3.sub`
  (`git checkout --` restores them; `evaluate_cla.py` reads `cla3.sub` from git when it is missing).
