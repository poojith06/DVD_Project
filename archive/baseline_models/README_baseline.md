# ML-Driven PVT-Variation-Aware Leakage Estimation (PTM 22 nm HP)

Course project for **Digital VLSI Design**, IIIT Hyderabad.

Machine-learning models replace SPICE for estimating the leakage current of CMOS circuits under process, voltage
and temperature (PVT) variation. Four transistor-level models (NMOS/PMOS, ON/OFF) predict the terminal currents
`id`, `ig`, `is`, `ib` of a single transistor. These models alone are then used to predict the leakage of five
logic gates and of a 3-bit carry-lookahead adder, and every prediction is checked against a full SPICE simulation.

| Item | Value |
|---|---|
| Technology | PTM 22 nm High Performance, BSIM4 (`22nm_HP.pm`) |
| Nominal supply | 1 V (corners: ±10 %) |
| Channel length | 22 nm; widths 1–8 × Lmin (22–176 nm) |
| Temperature | −40 to 125 °C |
| Process variation | `vth0`, `toxe`, `ndep`, `u0`, `xj` at nominal ±10 %, separate for NMOS and PMOS |
| Test circuit | 3-bit carry-lookahead adder (33 gates, 138 transistors) |
| Simulator | ngspice-44.2, DC operating point (`.op`) |
| Netlist options | `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` |

---

## Workflow

```
Part A  single-transistor SPICE data ──► 4 ML models (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF)
                                              │  predict id, ig, is, ib
Part B  gate SPICE (5 gates, 18 input cases, 200 PVT corners) ──► ML gate leakage vs SPICE
                                              │  same 200 corners
Part C  3-bit CLA SPICE (64 input vectors, 200 corners) ──► sum of ML gate leakages vs SPICE
```

| Part | What is done | Folder |
|---|---|---|
| **A1** Data | 10,000 random samples per case, generated entirely inside ngspice with a fixed seed | [`Datasets/`](Datasets/) |
| **A2** Models | Four model types trained per case on a 70 / 15 / 15 split (seed 42); best model per case chosen on the validation set, reported on the test set | [`ML_Models/`](ML_Models/) |
| **B** Gates | INV, NAND2, NOR2, AND2, OR2 at every input combination × 200 corners; ML leakage from the recorded SPICE node voltages and the four transistor models | [`Gates_Part-B/`](Gates_Part-B/) (SPICE), [`ML_Models/evaluate_gates.py`](ML_Models/evaluate_gates.py) (ML) |
| **C** Test circuit | 3-bit CLA built from the Part B gates; leakage = sum of the gate leakages | [`PartC_3bit CLA adder/`](PartC_3bit%20CLA%20adder/) |

---

## Repository layout

```
repo/
├── Datasets/                    Part A: single-transistor data
│   ├── netlists/                22nm_HP.pm + one ngspice netlist per case (sampling inside the .cir)
│   └── data/                    nmos_on.csv, nmos_off.csv, pmos_on.csv, pmos_off.csv (10,000 rows each)
├── ML_Models/                   Part A models and Part B evaluation (main pipeline)
│   ├── common/                  shared code: features, data split, metrics, training, sign model
│   ├── components_mlp/          one MLP per BSIM4 current component
│   ├── gpr/                     exact Gaussian process (ARD RBF / Matern-5/2)
│   ├── hybrid_analytic_mlp/     analytical compact model + MLP correction
│   ├── resnet_mlp/              MLP with residual connections
│   ├── select_best.py           best model per case (validation) -> results/<case>/
│   ├── predict.py               load a saved model and predict currents
│   ├── evaluate_gates.py        Part B: ML gate leakage vs SPICE -> gate_results/
│   ├── results/                 chosen model, parity plot and metrics per case
│   └── gate_results/            gate results table, per-row predictions, gate parity plot
├── Gates_Part-B/                Part B SPICE reference: 5 gate netlists + data (3,600 rows)
├── PartC_3bit CLA adder/        Part C SPICE reference: CLA netlist + data (12,800 rows)
├── ML models/                   first baseline models (MLP, XGBoost, random forest, ridge)
└── requirements.txt
```

Each main folder has its own README with the full details.

---

## Method

**Inputs.** `vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj`. `vg` and `vb` are fixed by the case
(for example NMOS OFF: `vg = 0`, `vb = 0`).

**Targets.** log10 of each terminal current magnitude (floor 1e-18 A, matching `abstol`). Signs are restored
afterwards: from the physics of the drain–source voltage in the ON cases, and with a fixed-sign effective drain
plus a gradient-boosting classifier for the effective source in the OFF cases.

**Features.** The 10 varying inputs plus physics-based features: effective source (lower terminal for NMOS,
higher for PMOS), `Vgs` and `Vbs` relative to it, `|Vds|`, `1/T`, the subthreshold term `(|Vgs| − |vth0|)/VT`,
oxide fields `|Vg − Vd|/toxe` and `|Vg − Vs|/toxe`, `log10 W` and `log10(1 − exp(−|Vds|/VT))`.

**Model choice.** Every model type is trained on the same split. For each case, the model with the lowest mean
validation median error over the four currents is chosen (mean 95th-percentile error breaks ties). The test set is
used only for the final report.

**Gate leakage (Part B).** Each transistor's terminal voltages are taken from the recorded SPICE node voltages
(inputs, internal stack node, first-stage output, output), the ON/OFF model is chosen from its gate voltage, and
its four currents are predicted. Gate leakage is the current drawn from VDD, assembled with Kirchhoff's current law:
the supply node and every node joined to it through ON PMOS are treated as one node, so the channel currents of
ON transistors (which have almost no voltage across them) cancel and only OFF-transistor currents and gate
currents remain.

**Metrics.** Relative error `(ML − SPICE) / SPICE`; median, 95th percentile and mean absolute error.
Mean accuracy = 100 % − mean absolute error.

---

## Results

### Part A: chosen transistor models (test set, median relative error)

| Case | Model | `id` | `ig` | `is` | `ib` | Mean p95 |
|---|---|---|---|---|---|---|
| NMOS ON | GPR | 0.160 % | 0.239 % | 0.163 % | 0.164 % | 2.36 % |
| NMOS OFF | components MLP | 0.211 % | 0.107 % | 0.192 % | 0.004 % | 1.39 % |
| PMOS ON | GPR | 0.218 % | 0.033 % | 0.110 % | 0.188 % | 1.19 % |
| PMOS OFF | components MLP | 0.154 % | 0.123 % | 0.355 % | 0.003 % | 1.37 % |

Full comparison of all four model types: `ML_Models/results/<case>/validation_comparison.csv`.

### Part B: gate leakage (3,600 gate simulations)

| Gate | INV | NAND2 | NOR2 | AND2 | OR2 | **All** |
|---|---|---|---|---|---|---|
| Mean accuracy | 96.0 % | 97.9 % | 97.5 % | 95.7 % | 96.9 % | **96.9 %** |
| Median accuracy | 96.9 % | 98.9 % | 98.2 % | 96.9 % | 97.5 % | **97.9 %** |

Per input case (SPICE median, ML median, bias, median and 95th-percentile error):
`ML_Models/gate_results/gate_results.csv`; parity plot: `ML_Models/gate_results/gate_parity.png`.

The largest errors occur where an OFF transistor has the full supply voltage across it with its source at the
rail (INV, NAND2 `11`, AND2): this corner of the input space is sparsely covered by uniformly sampled training data.

### Part C: 3-bit CLA

In progress. The SPICE reference (`PartC_3bit CLA adder/data/cla3.csv`) is complete.

---

## Getting started

**Requirements:** Python 3.10+, ngspice 44 (only to regenerate the SPICE data). A CUDA GPU is optional and speeds
up the neural networks and GPR.

```bash
pip install -r requirements.txt
```

**1. Generate the single-transistor data** (optional, the CSVs are included)

```bash
cd Datasets/netlists
for c in nmos_on nmos_off pmos_on pmos_off; do ngspice -b ${c}_ngspice.cir > /dev/null; done
mv *.csv ../data/
```

On Windows use `ngspice_con.exe` (see [`Datasets/README.md`](Datasets/README.md)).

**2. Train the models and choose one per case** (from inside `ML_Models/`)

```bash
python components_mlp/train.py
python hybrid_analytic_mlp/train.py
python resnet_mlp/train.py
python gpr/train.py          # slowest: about 25-60 min per case
python select_best.py
```

`bash run_all.sh` runs these in order and writes logs to `ML_Models/logs/`.

**3. Evaluate the gates (Part B)**

```bash
python evaluate_gates.py
```

**Use a model in your own code**

```python
import sys; sys.path.insert(0, "ML_Models")
from predict import load_surrogate

currents = load_surrogate("nmos_off").predict(df)   # df: the 12 input columns -> id, ig, is, ib in A
```

---

## Reproducibility

| Item | Seed / setting |
|---|---|
| Single-transistor data | fixed `rndseed` in each netlist (ngspice-44.2) |
| Gate and CLA corners | `rndseed=16`, identical 200 corners in Part B and Part C |
| Data split | 70 / 15 / 15, seed 42 |
| Model initialisation | seed 0 |

ngspice's random stream depends on the build: the same seed can give different corners on another ngspice
version. The drawn values are always written to the CSVs, so the reference data stays fixed.

---

## Status

| Deliverable | Status |
|---|---|
| Datasets, netlists | Done |
| Model comparison, chosen models | Done |
| Gate results table and parity plot | Done |
| Data check (Kirchhoff table, trend plots) | To do |
| Test-circuit results | In progress |
| Report | To do |
