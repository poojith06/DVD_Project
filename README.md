# ML-driven PVT-aware leakage estimation, PTM 22 nm HP

Digital VLSI Design course project (assignment: [`docs/DVD_ML_PROJECT.pdf`](docs/DVD_ML_PROJECT.pdf)).

We replace SPICE for leakage-current estimation with machine-learning models of single transistors. Four models
(NMOS ON, NMOS OFF, PMOS ON, PMOS OFF) predict the four terminal currents (id, ig, is, ib) of a PTM 22 nm HP transistor
(VDD = 1 V, L = 22 nm) from bias, temperature, width and five process parameters. They are trained on 10,000 ngspice-44.2
simulations per case (Part A). The leakage of the gates INV, NAND2, NOR2, AND2 and OR2 is built from the transistor
predictions with Kirchhoff's current law and checked against 3,600 gate simulations (Part B). The leakage of a 3-bit
carry-lookahead adder is the sum of its 33 gate leakages and is checked against 12,800 full-adder simulations (Part C).
No Part B or Part C data is used for training or model choice.

## Results

| | Part B: 18 gate cases, 3,600 simulations | Part C: 3-bit CLA, 12,800 simulations |
|---|---|---|
| Mean accuracy (100 − mean \|error\|) | **98.87 %** | **99.00 %** |
| Median accuracy (100 − median \|error\|) | 99.18 % | 99.17 % |
| Ceiling (same sum built from the SPICE gate leakages) | – | 99.16 % |

Final models (80:10:10 split, chosen on the validation set only):

| Case | Model |
|---|---|
| NMOS ON | hybrid analytic model + MLP |
| NMOS OFF | ResNet MLP, 5 residual blocks |
| PMOS ON | Gaussian process regression (GPR) |
| PMOS OFF | ResNet MLP, 5 residual blocks |

The full report is [`DVD_Angels_Report.pdf`](DVD_Angels_Report.pdf).

## Folder map

| Folder / file | Assignment step | Deliverable | Contents |
|---|---|---|---|
| `model/22nm_HP.pm` | all | input of D2, D9 | PTM 22 nm HP model card, included by every netlist |
| `partA_transistor_models/step1_data_generation/` | Part A, step 1 | D1, D2 | 4 single-transistor netlists and the 4 datasets (10,000 rows each) |
| `partA_transistor_models/data_check/` | Part A, step 1 | D3 | Kirchhoff check, 3 trend plots and the script that makes them |
| `partA_transistor_models/step2_training/code/` | Part A, step 2 | D9 | model code, training, selection and run scripts |
| `partA_transistor_models/step2_training/model_comparison/` | Part A, step 2 | D4 | 4 model types × 4 cases on the validation set, tuning tables |
| `partA_transistor_models/step2_training/chosen_models/` | Part A, step 2 | D5 | test-set errors and predicted-vs-SPICE plots of the chosen models |
| `partA_transistor_models/step2_training/saved_models/` | Part A, step 2 | D5 | the 4 final model files |
| `partB_gates/step3_spice_reference/` | Part B, step 3 | D2 | 5 gate netlists and their SPICE data (200 corners) |
| `partB_gates/step4_ml_prediction/` | Part B, step 4 | D9 | gate leakage from the transistor models |
| `partB_gates/step5_results/` | Part B, step 5 | D6, D7 | gate results table, all predictions, parity plot |
| `partC_cla/step6_spice_reference/` | Part C, step 6 | D2 | CLA netlists, SPICE data (64 vectors × 200 corners), regeneration script |
| `partC_cla/step7_ml_prediction/` | Part C, step 7 | D9 | CLA leakage = sum of the gate leakages |
| `partC_cla/step8_results/` | Part C, step 8 | D8 | CLA results table, all predictions, parity plot |
| `run_all.ps1` | steps 1–8 | D9 | runs every step in order |
| `requirements.txt` | – | D9 | Python packages and versions |
| `DVD_Angels_Report.pdf` | – | D10 | the report |
| `docs/DVD_ML_PROJECT.pdf` | – | – | the assignment |

Each part folder has its own README that describes every file.

## How to run

```powershell
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File run_all.ps1 -Fast        # saved models + saved SPICE data, about a minute
powershell -ExecutionPolicy Bypass -File run_all.ps1 -SkipSpice   # re-train (hours, GPU), keep the saved SPICE data
powershell -ExecutionPolicy Bypass -File run_all.ps1 -SkipTrain   # re-run SPICE (ngspice), keep the saved models
powershell -ExecutionPolicy Bypass -File run_all.ps1              # everything (asks for confirmation; -Yes skips it)
```

Options: `-Out DIR` (default `pipeline_out`), `-Ngspice PATH` (path to `ngspice_con.exe`), `-Python PATH`.
All outputs go to `pipeline_out/`; the submitted files in the step folders are never overwritten. The fast path ends
with Part B 98.87 %, Part C 99.00 % and `REPRODUCTION: PASS`.

### Steps and seeds

| Step | What | Script | Seeds and settings |
|---|---|---|---|
| 1 | Part A datasets | `partA_transistor_models/step1_data_generation/netlists/*_ngspice.cir` | `rndseed` 13 (NMOS ON), 12 (NMOS OFF), 14 (PMOS ON), 15 (PMOS OFF); VDD 1 V ± 10 %, L 22 nm, W 22–176 nm |
| 1 | Data check (D3) | `partA_transistor_models/data_check/data_check.py` | deterministic |
| 2 | Training and model comparison | `step2_training/code/train.py`, `choose_weight.py` | split 80:10:10, split seed 42, model seed 0; rail weight 0 / 10 / 30 tried, 0 chosen on validation |
| 2 | ResNet OFF tuning, final model set | `train.py --model-params …`, `choose_resnet.py`, `assemble_tuned.py`, `select_best.py` | same split and seeds; 7 variants, 5 blocks chosen on validation |
| 3 | Part B SPICE | `partB_gates/step3_spice_reference/netlists/*_ngspice.cir` | `rndseed` 16 (200 corners, shared with Part C) |
| 4–5 | Part B ML and comparison | `partB_gates/step4_ml_prediction/evaluate_gates.py` | – |
| 6 | Part C SPICE | `partC_cla/step6_spice_reference/netlists/cla3_ngspice.cir` | `rndseed` 16 |
| 7–8 | Part C ML and comparison | `partC_cla/step7_ml_prediction/evaluate_cla.py` | – |
| check | compare with the submitted results | `step2_training/code/reproduce_check.py` | – |

The random stream behind `rndseed` depends on the ngspice build (the data were made with ngspice-44.2 on Windows).
The drawn values are stored in every CSV, and all ML steps work from those saved files.

### Linux / macOS (Python commands)

Fast path, from the repository root:

```bash
pip install -r requirements.txt
python partA_transistor_models/data_check/data_check.py --out pipeline_out/partA_data_check
python partB_gates/step4_ml_prediction/evaluate_gates.py --out pipeline_out/gate_results
python partC_cla/step7_ml_prediction/evaluate_cla.py --predictions pipeline_out/gate_results/gate_predictions.csv --out pipeline_out/cla_results
python partA_transistor_models/step2_training/code/reproduce_check.py --out pipeline_out
```

SPICE (each netlist is run from its own folder and writes to `../data/`; this overwrites the saved CSVs):

```bash
(cd partA_transistor_models/step1_data_generation/netlists && for c in nmos_on nmos_off pmos_on pmos_off; do ngspice -b ${c}_ngspice.cir; done)
(cd partB_gates/step3_spice_reference/netlists && for g in inv nand2 nor2 and2 or2; do ngspice -b ${g}_ngspice.cir; done)
(cd partC_cla/step6_spice_reference/netlists && ngspice -b cla3_ngspice.cir)
```

Training (from `partA_transistor_models/step2_training/code/`, outputs in the git-ignored `runs/` and `results/`):

```bash
for A in 0 10 30; do python train.py --models resnet_mlp --cases nmos_off pmos_off --rail-weight $A --tag tune_A$A --force; done
python choose_weight.py
for M in resnet_mlp hybrid_analytic_mlp pinn gpr; do python train.py --models $M --rail-weight $(cat runs/chosen_rail_weight.txt) --tag final --force; done
for kv in ep1000:epochs=1000 lr5e-4:lr=5e-4 wd0:weight_decay=0 bs64:batch_size=64 hub002:huber_beta=0.02 w512:width=512 b5:blocks=5; do
  python train.py --models resnet_mlp --cases nmos_off pmos_off --model-params "${kv#*:}" --tag "rn_${kv%%:*}" --force; done
python choose_resnet.py && python assemble_tuned.py && python select_best.py --tag final_tuned
```

## Assumptions

1. **Accuracy** = 100 − mean absolute percentage error over all simulations, error = (ML − SPICE) / SPICE.
2. **Gate leakage** = the current drawn from the VDD supply only; currents of the input sources are not included.
3. **Training data**: the models use only the 16 dataset columns of the assignment (no internal BSIM4 component currents).
4. **PMOS OFF ig / ib** are read from the BSIM4 instance currents, because the voltage-source branch currents are
   rounded to steps of about 2.2e-16 A; every terminal still has its own voltage source.

## Team

Team **Dvd Angels**, IIIT Hyderabad

- Mididoddi Sai Poojith
- Devang Bordoloi
- Vajrapu Venkata Sai Aneesh
- Apanna Nihaal
