# Part C – Test circuit: 3-bit carry-lookahead adder (assignment steps 6–8)

The 3-bit CLA (inputs A = a2a1a0, B = b2b1b0, no carry-in, all 64 vectors) is built only from Part B gates: 27 NAND2,
3 INV, 2 AND2 and 1 OR2 (33 gates, 138 transistors). It is simulated as one complete netlist at the same 200 corners
as Part B (`rndseed` 16): 64 × 200 = 12,800 simulations. The ML leakage of each vector and corner is the sum of the 33
Part B ML gate leakages.

**The final Part C results are in `step8_results/`.** Running `evaluate_cla.py` writes to `pipeline_out/` by default
and never overwrites them.

## step6_spice_reference/ — step 6, deliverable D2

Run from `netlists/`: `ngspice_con -b cla3_ngspice.cir`, which writes `../data/cla3.csv` and `../data/corners.csv`
(about 15–20 minutes).

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `netlists/cla3_ngspice.cir` | testbench: supply, 6 input sources, the CLA, 200 corners × 64 vectors | 6 | D2 |
| `netlists/cla3.sub` | `.subckt CLA3`: the 33 gate calls (generate, propagate as 4-NAND XORs, carries, sums) | 6 | D2 |
| `netlists/gates.sub` | INV, NAND2, NOR2, AND2, OR2 subcircuits, same transistors and widths as the Part B netlists | 6 | D2 |
| `scripts/regen_cla3.py` | re-runs the unchanged netlist with the recorded corners into a new file (`pipeline_out/partC/cla3_regen.csv`) and, with `--compare`, checks it against `data/cla3.csv` | 6 | D9 |
| `data/cla3.csv` | SPICE results, 12,800 rows: corner, A, B, input bits, corner values, `leak` (current from VDD), all inter-gate and output node voltages | 6 | D8 (reference) |
| `data/corners.csv` | the 200 corners (VDD, temperature, NMOS and PMOS process values), shared with Part B | 6 | D8 (reference) |

## step7_ml_prediction/ — step 7, deliverable D9

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `evaluate_cla.py` | Reads the gate list from the netlist, computes the logic value at every gate input for each vector (checked against the SPICE outputs), sums the 33 Part B ML gate leakages per vector and corner (and the same sum of the SPICE gate leakages as the ceiling), and compares with `cla3.csv`. Run: `python partC_cla/step7_ml_prediction/evaluate_cla.py [--predictions CSV] [--out DIR]` (defaults: `partB_gates/step5_results/gate_predictions.csv`, `pipeline_out/cla_results`). | 7–8 | D8, D9 |

## step8_results/ — step 8, deliverable D8

| File | What it is | Step | Deliverable |
|---|---|---|---|
| `cla3_results.csv` | overall and per-vector metrics (SPICE median, ML median, bias, median, 95th-percentile, mean and max error) for the ML sum and for the ceiling | 8 | D8 |
| `cla3_predictions.csv` | all 12,800 simulations: SPICE leakage, ML leakage, ceiling, errors, corner values | 8 | D8 |
| `cla3_parity.png` | ML vs SPICE CLA leakage, all 12,800 simulations, log axes | 8 | D8 |
| `cla3_gates.csv` | the 33 gates of the CLA in evaluation order: instance, type, input nets, output net | 7 | D8 |
| `summary.csv` | Part B and Part C accuracy under four definitions, Part C error statistics, worst case | 8 | D8 |

Mean accuracy over the 12,800 simulations: **99.00 %** (ceiling with perfect gate models: 99.16 %; median accuracy
99.17 %, p95 error 2.57 %).
