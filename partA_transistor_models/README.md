# Part A – Transistor models (assignment steps 1–2)

Single-transistor datasets for the four cases (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF), their data check, and the
training and choice of one ML model per case. Model file `../model/22nm_HP.pm` (PTM 22 nm HP), VDD = 1 V, L = 22 nm,
ngspice-44.2.

## step1_data_generation/ — step 1, deliverables D1, D2

Each netlist draws 10,000 random samples inside ngspice (VDD 1 V ± 10 % first, then T −40…125 °C, W 22–176 nm,
Vd and Vs 0…VDD, vth0 / toxe / ndep / u0 / xj nominal ± 10 %), runs a `.op` for each and appends one CSV row.
Vg and Vb are fixed by the case. Run from `netlists/`: `ngspice_con -b <case>_ngspice.cir`, which writes `../data/<case>.csv`.

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `netlists/nmos_on_ngspice.cir` | NMOS ON netlist (Vg = VDD, Vb = 0), `rndseed` 13 | 1 | D2 |
| `netlists/nmos_off_ngspice.cir` | NMOS OFF netlist (Vg = 0, Vb = 0), `rndseed` 12 | 1 | D2 |
| `netlists/pmos_on_ngspice.cir` | PMOS ON netlist (Vg = 0, Vb = VDD), `rndseed` 14 | 1 | D2 |
| `netlists/pmos_off_ngspice.cir` | PMOS OFF netlist (Vg = VDD, Vb = VDD), `rndseed` 15; ig / ib from BSIM4 instance currents | 1 | D2 |
| `data/nmos_on.csv` | NMOS ON dataset, 10,000 rows, columns `vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib` | 1 | D1 |
| `data/nmos_off.csv` | NMOS OFF dataset, same columns | 1 | D1 |
| `data/pmos_on.csv` | PMOS ON dataset, same columns | 1 | D1 |
| `data/pmos_off.csv` | PMOS OFF dataset, same columns | 1 | D1 |

Currents are into each terminal (A), so `id + ig + is + ib ≈ 0`.

## data_check/ — step 1, deliverable D3

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `data_check.py` | reads the 4 datasets and writes the five files below (`python data_check.py [--out DIR]`, default: this folder) | 1 | D3, D9 |
| `kcl_check.csv` | Kirchhoff check per case: median / max of \|id+ig+is+ib\| and of the sum relative to the largest terminal current | 1 | D3 |
| `trend_off_vs_temperature.png` | OFF current max(\|id\|, \|is\|) vs temperature, NMOS OFF and PMOS OFF | 1 | D3 |
| `trend_off_vs_vth0.png` | OFF current vs vth0 (NMOS OFF) and \|vth0\| (PMOS OFF) | 1 | D3 |
| `trend_ig_vs_toxe.png` | gate current \|ig\| vs toxe, all four cases | 1 | D3 |
| `trend_numbers.csv` | numbers behind the trend plots (Spearman correlation, binned medians, effect with other inputs held fixed) | 1 | D3 |

## step2_training/ — step 2, deliverables D4, D5, D9

### code/ (used in this order)

| # | File | What it does |
|---|---|---|
| 1 | `config.py` | Shared settings: data and output paths, split 80:10:10 with seed 42, model seed 0, the 12 input and 4 output columns, the four cases, the four model types, near-rail weighting settings. |
| 2 | `pipeline.py` | Shared helpers: loads and checks a case CSV, splits it, builds scaled features, computes log10\|I\| targets and error metrics, `SignModel` (predicts the sign of each current), and `TransistorSurrogate`, the saved-model wrapper (`load`, `predict` → id, ig, is, ib in A). |
| 3 | `models.py` | The four model types, each predicting log10\|I\| of id, ig, is, ib: `gpr` (exact Gaussian process), `hybrid_analytic_mlp` (analytic device equations + MLP correction), `resnet_mlp` (MLP with residual blocks), `pinn` (physics-informed MLP). |
| 4 | `train.py` | Trains the chosen model types on the chosen cases. Input: `step1_data_generation/data/*.csv`. Output: `runs/<tag>/<model>/saved_models/<case>.joblib`, train / validation / test metrics, test parity plots. |
| 5 | `choose_weight.py` | Picks the near-rail training weight from the runs `runs/tune_A0`, `tune_A10`, `tune_A30` on validation data only. Output: `runs/chosen_rail_weight.txt`, `runs/rail_weight_tuning.csv`. |
| 6 | `choose_resnet.py` | Picks one ResNet setting for the two OFF cases from `runs/tune_A0` and the 7 `runs/rn_*` variants, on validation data only. Output: `runs/chosen_resnet.txt`, `runs/resnet_tuning.csv`. |
| 7 | `assemble_tuned.py` | Builds `runs/final_tuned` = `runs/final` with the OFF ResNet models replaced by the chosen tuned run. |
| 8 | `select_best.py` | Picks, per case, the model type with the lowest mean validation median error. Output (git-ignored `results/`): model comparison table, chosen-model test table, `<case>/model.joblib`, parity plots. |
| 9 | `run_parallel.ps1` | Windows run of steps 4, 5, 8 in parallel (GPU + CPU), then the Part B / Part C check on the selected models. Output in `runs/`, `results/`, `logs/`. |
| 10 | `tune_resnet.ps1` | Windows run of the 7 ResNet tuning variants for the OFF cases, then `choose_resnet.py`. |
| 11 | `finish_tuned.ps1` | Waits for the tuning runs, then runs `choose_resnet.py`, `assemble_tuned.py`, `select_best.py --tag final_tuned` and the Part B / Part C check, into `results_tuned/`. |
| 12 | `reproduce_check.py` | Compares a pipeline run (`--out pipeline_out`) with the submitted Part B and Part C predictions; prints the accuracies, the largest difference and PASS / FAIL. |

`runs/`, `results/`, `results_tuned/` and `logs/` are created by these scripts and are git-ignored; nothing in them
overwrites the files below. The whole chain is run by `../../run_all.ps1` from the repository root.

### model_comparison/ — deliverable D4

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `model_comparison_validation.csv` | 4 model types × 4 cases on the validation set: R² (on log10\|I\|), median and 95th-percentile error per current and averaged | 2 | D4 |
| `validation_comparison_nmos_on.csv` | the 4 model types for NMOS ON, ranked by validation score | 2 | D4 |
| `validation_comparison_nmos_off.csv` | the same for NMOS OFF | 2 | D4 |
| `validation_comparison_pmos_on.csv` | the same for PMOS ON | 2 | D4 |
| `validation_comparison_pmos_off.csv` | the same for PMOS OFF | 2 | D4 |
| `resnet_tuning.csv` | the default and 7 tuned ResNet settings for the OFF cases on validation (5 blocks chosen) | 2 | D4, D10 |
| `rail_weight_tuning.csv` | near-rail training weight 0 / 10 / 30 on validation (0 chosen) | 2 | D4, D10 |

### chosen_models/ — deliverable D5

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `chosen_models_test.csv` | chosen model per case with its test-set median and 95th-percentile error per current | 2 | D5 |
| `parity_nmos_on.png` | predicted vs SPICE currents (test set), NMOS ON | 2 | D5 |
| `parity_nmos_off.png` | the same, NMOS OFF | 2 | D5 |
| `parity_pmos_on.png` | the same, PMOS ON | 2 | D5 |
| `parity_pmos_off.png` | the same, PMOS OFF | 2 | D5 |

### saved_models/ — deliverable D5

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `nmos_on.joblib` | final NMOS ON model (hybrid analytic + MLP) | 2 | D5 |
| `nmos_off.joblib` | final NMOS OFF model (ResNet MLP, 5 blocks) | 2 | D5 |
| `pmos_on.joblib` | final PMOS ON model (GPR) | 2 | D5 |
| `pmos_off.joblib` | final PMOS OFF model (ResNet MLP, 5 blocks) | 2 | D5 |

The model files are pickled objects of the classes in `code/`. Load them with `code/` on the import path:

```python
import sys; sys.path.insert(0, "partA_transistor_models/step2_training/code")
from pipeline import TransistorSurrogate
model = TransistorSurrogate.load("partA_transistor_models/step2_training/saved_models/nmos_off.joblib")
currents = model.predict(df)   # df: vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj -> id, ig, is, ib (A)
```
