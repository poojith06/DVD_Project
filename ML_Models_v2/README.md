# ML_Models_v2: spec-compliant models, Part B and Part C

Second version of the transistor models, built to follow the project document strictly and to push the gate-level
accuracy towards 99 %. `ML_Models/` is left unchanged.

## What is different from `ML_Models/`

| | `ML_Models/` | `ML_Models_v2/` |
|---|---|---|
| Model types | components MLP, GPR, hybrid, ResNet | **GPR, hybrid, ResNet, PINN**, all in [`models.py`](models.py) |
| Training targets | components MLP also learns BSIM4 internal currents | only the four dataset currents `id, ig, is, ib` |
| Split | 70 / 15 / 15 | **80 / 10 / 10** (allowed by the spec), seed 42 |
| Training loss | uniform | rows near the rails weighted more (strength chosen on validation) |
| Part C | – | sum of the gate leakages ([`evaluate_cla.py`](evaluate_cla.py)) |

The data are the same 10,000 rows per case (`../Datasets/data`). Drain–source symmetry preprocessing is on hold.

## Why near-rail weighting

Inside a logic gate every transistor terminal sits at 0 or VDD (for example an OFF transistor with its source on
one rail and its drain on the other). With Vd and Vs drawn uniformly, very few training rows are close to those
points, and that is where the remaining gate error comes from. The training loss of the neural networks is
weighted by `1 + A * exp(-r / 50 mV)` (normalised to mean 1), where `r` = distance of vd from the nearer rail +
distance of vs from the nearer rail. The data itself is unchanged. `A` is chosen on validation
([`choose_weight.py`](choose_weight.py)): the lowest error on the near-rail validation rows (`r < 0.2 V`) for the
OFF drain/source currents, as long as the normal validation score is at most 10 % worse than with `A = 0`.
GPR has no per-row loss and is trained unweighted.

## Files

| File | Purpose |
|---|---|
| `config.py` | paths, split, seeds, cases, weighting settings |
| `pipeline.py` | data loading and split, features, metrics, sign model, saved-model wrapper |
| `models.py` | the four model types (GPR, hybrid analytic + MLP, ResNet MLP, PINN) |
| `train.py` | train model types on the cases → `runs/<tag>/<model>/` |
| `choose_weight.py` | pick the weighting strength from the tuning runs (validation only) |
| `select_best.py` | best model per case on validation → `results/` (model comparison table, chosen models) |
| `evaluate_gates.py` | Part B: gate leakage vs SPICE → `results/gate_results/` |
| `evaluate_cla.py` | Part C: CLA leakage = sum of gate leakages vs SPICE → `results/cla_results/` |
| `run_all.ps1` / `run_all.sh` | everything above in order (Windows / Linux), logs in `logs/` |

## Running

```bash
cd ML_Models_v2
powershell -ExecutionPolicy Bypass -File run_all.ps1    # Windows
bash run_all.sh                                          # Linux / macOS
```

Steps: tuning runs (`A` = 0, 10, 30) → choose `A` → train all 4 types × 4 cases → select → Part B → Part C.
About 6 hours on an RTX 3050 laptop GPU (GPR is the slowest, ~25 min per case).

## Outputs (`results/`)

| File | Spec deliverable |
|---|---|
| `model_comparison_validation.csv` | 4: 4 model types × 4 cases, R², median and p95 error on validation |
| `chosen_models_test.csv`, `<case>/model.joblib`, `<case>/parity.png` | 5: chosen models, test errors, parity plots, model files |
| `gate_results/gate_results.csv`, `gate_parity.png` | 6, 7: gate results table and parity plot |
| `cla_results/cla_results.csv`, `cla_parity.png` | 8: test-circuit results and parity plot |

Gate leakage is the current drawn from VDD, built from the predicted transistor currents with Kirchhoff's current
law (see the `evaluate_gates.py` docstring). Mean accuracy = 100 % − mean |(ML − SPICE) / SPICE|.

## Using a model

```python
import sys; sys.path.insert(0, "ML_Models_v2")
from pipeline import TransistorSurrogate
model = TransistorSurrogate.load("ML_Models_v2/results/nmos_off/model.joblib")
currents = model.predict(df)      # df with the 12 input columns -> id, ig, is, ib in amps
```
