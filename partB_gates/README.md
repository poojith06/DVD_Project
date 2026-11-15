# Part B – Gate-level validation (assignment steps 3–5)

INV, NAND2, NOR2, AND2 and OR2 (static CMOS; AND2 / OR2 = NAND2 / NOR2 + INV, simulated as complete gates), every
input combination at 200 random PVT corners: 18 cases × 200 = 3,600 simulations. A corner is one draw of VDD,
temperature and process values; all NMOS share one draw and all PMOS another (`rndseed` 16, the same 200 corners as
Part C). The SPICE data are used for evaluation only, never for training.

**The final Part B results are in `step5_results/`.** Running `evaluate_gates.py` writes to `pipeline_out/` by default
and never overwrites them.

## step3_spice_reference/ — step 3, deliverable D2

Run from `netlists/`: `ngspice_con -b <gate>_ngspice.cir`, which writes `../data/<gate>.csv`.

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `netlists/inv_ngspice.cir` | INV netlist (PMOS 88 nm, NMOS 44 nm), 200 corners × A = 0, 1 | 3 | D2 |
| `netlists/nand2_ngspice.cir` | NAND2 netlist (2 × PMOS 88 nm parallel, 2 × NMOS 88 nm series), internal node x | 3 | D2 |
| `netlists/nor2_ngspice.cir` | NOR2 netlist (2 × PMOS 176 nm series, 2 × NMOS 44 nm parallel), internal node y | 3 | D2 |
| `netlists/and2_ngspice.cir` | AND2 netlist (NAND2 + INV), nodes x and NAND2 output | 3 | D2 |
| `netlists/or2_ngspice.cir` | OR2 netlist (NOR2 + INV), nodes y and NOR2 output | 3 | D2 |
| `data/inv.csv` | INV SPICE results, 400 rows: corner, inputs, corner values, `leak` (current from VDD), `v_out` | 3 | D6, D7 (reference) |
| `data/nand2.csv` | NAND2 SPICE results, 800 rows, plus `v_int` (x) | 3 | D6, D7 (reference) |
| `data/nor2.csv` | NOR2 SPICE results, 800 rows, plus `v_int` (y) | 3 | D6, D7 (reference) |
| `data/and2.csv` | AND2 SPICE results, 800 rows, plus `v_int` (x) and `v_n` (NAND2 output) | 3 | D6, D7 (reference) |
| `data/or2.csv` | OR2 SPICE results, 800 rows, plus `v_int` (y) and `v_n` (NOR2 output) | 3 | D6, D7 (reference) |

## step4_ml_prediction/ — step 4, deliverable D9

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `evaluate_gates.py` | Gives every transistor of each gate its terminal voltages from the recorded SPICE node voltages, predicts its currents with the four final transistor models, and computes the gate leakage drawn from VDD with Kirchhoff's current law. Writes `gate_predictions.csv`, `gate_results.csv`, `gate_parity.png`. Run: `python partB_gates/step4_ml_prediction/evaluate_gates.py [--models-dir DIR] [--out DIR]` (defaults: `step2_training/saved_models`, `pipeline_out/gate_results`). | 4–5 | D6, D7, D9 |

## step5_results/ — step 5, deliverables D6, D7

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `gate_results.csv` | 18 gate / input cases: SPICE median, ML median, bias, median, 95th-percentile, mean and max error, mean accuracy | 5 | D6 |
| `gate_predictions.csv` | all 3,600 simulations: corner values, SPICE and ML leakage, error, SPICE output voltage, internal node voltages from SPICE and ML (the ML values are the recorded SPICE voltages used as inputs in step 4) | 5 | D6 |
| `gate_parity.png` | ML vs SPICE gate leakage, all 3,600 simulations, log axes | 5 | D7 |

Mean accuracy over all 3,600 simulations: **98.87 %** (per gate: INV 98.47 %, NAND2 99.02 %, NOR2 98.96 %,
AND2 98.84 %, OR2 98.85 %).
