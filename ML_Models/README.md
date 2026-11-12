# ML_Models: the top 4 transistor models

The four best regression models from the 29-model benchmark in `../Model_comparison/`, ranked by mean validation
median error and confirmed with 3 training seeds (`../Model_comparison/seed_runs/`). Same layout as `../ML models/`:
shared code in `common/`, one folder per model with `model.py` and `train.py`, and `select_best.py` / `predict.py`.

| Folder | Model | Notes |
|---|---|---|
| `components_mlp/` | One MLP per BSIM4 current component (channel, junction, gate tunnelling, GIDL ...); terminal currents rebuilt with the BSIM4 sum formulas | Best in both OFF cases. Signs and Kirchhoff's law come out by construction. Needs the component labels in `component_data/` (internal simulator currents - check with the TA that this is allowed) |
| `gpr/` | Exact Gaussian process (ARD RBF or Matern-5/2 kernel, picked per current on validation) | Best `id`/`is` in the ON cases. Slow: about 40-60 min per case on an RTX 3050 |
| `hybrid_analytic_mlp/` | Analytical compact model (`analytic.py`) + MLP that learns its remaining error | Strong in the ON cases |
| `resnet_mlp/` | MLP with residual (skip) connections, one network per group (`id`+`is`, `ig`, `ib`) | Good all-round; least stable across seeds |

## Common pipeline (identical for all four)

| Step | Choice |
|---|---|
| Data | `../Datasets/data/<case>.csv` (10,000 rows per case), split 70/15/15 with seed 42 |
| Inputs | `common/features.py` (same as `../ML models/`): 10 varying CSV columns + 10 physics features, standardised on the training split |
| Target | log10\|I\| of `id`, `ig`, `is`, `ib` (floor 1e-18 A) |
| Networks | `common/nets.py`: Huber loss (0.05 decades), AdamW lr 1e-3, cosine LR, batch 128, 500 epochs, best validation epoch |
| Signs | `common/signs.py`: ON cases - physics rule sign(id) = sign(vd - vs); OFF cases - the effective drain has a fixed sign, the effective source's sign comes from a gradient-boosting classifier (not needed by `components_mlp`) |
| Tuning | validation split only; the test split is never used for a choice |
| Metrics | relative error \|pred - SPICE\| / \|SPICE\| per current: MAE, median, p95 (%), R2 on log10\|I\|, sign accuracy |

## How to run

From inside `ML_Models/`:

```bash
python components_mlp/train.py          # all four cases; --cases nmos_off pmos_off for some
python hybrid_analytic_mlp/train.py
python resnet_mlp/train.py
python gpr/train.py                     # slow
python select_best.py                   # best model per case -> results/<case>/
python predict.py --case nmos_off --input samples.csv --output currents.csv
```

`bash run_all.sh` does all of the above in order (logs in `logs/`). Run the models one at a time: two GPU
jobs at once slow each other down a lot.

Each `train.py` writes `results/` (train/validation/test metrics CSVs, a test parity plot per case) and
`saved_models/<case>.joblib` in its own folder. Useful options: `--model-seed` (network/GPR seed, default 0),
`--epochs` (quick test runs), `--data-dir`.

`select_best.py` picks, per case, the model with the lowest mean validation median error over the four currents
(`--choose pmos_on=gpr` to override) and copies it to `results/<case>/model.joblib`.

In Python:

```python
import sys; sys.path.insert(0, "ML_Models")
from predict import load_surrogate
currents = load_surrogate("nmos_off").predict(df)   # df has the 12 input columns -> id, ig, is, ib in amps
```

## Part B: gate leakage (`evaluate_gates.py` -> `gate_results/`)

`python evaluate_gates.py` predicts the leakage of every row of `../Gates_Part-B/data/<gate>.csv` (5 gates,
18 input cases, 200 corners) with the chosen models and writes `gate_predictions.csv` (every row),
`gate_results.csv` (18-row table: SPICE median, ML median, bias, mean / median / p95 / max error) and
`gate_parity.png`.

- Each transistor's terminal voltages come from the recorded SPICE node voltages (inputs, `v_int`, `v_n`,
  `v_out`); ON/OFF is decided from its gate voltage (NMOS ON if Vg > VDD/2, PMOS ON if Vg < VDD/2).
- Leakage = current drawn from VDD (as in the SPICE netlists), built with Kirchhoff's current law on the
  "supply node" S = vdd plus every node joined to it through ON PMOS: the sum of the predicted currents into
  all terminals on S. ON PMOS inside S contribute only -ig, because their channel current (with ~0 V across
  them, below the CSV's 6-digit resolution) cancels. The literal sum over the terminals tied to vdd is also
  saved (`leak_ml_supply_sum`) but fails badly for exactly that reason.
- Error = (ML - SPICE) / SPICE, bias = median signed error, mean accuracy = 100 % - mean |error|.

## Component labels (`components_mlp/component_data/`)

`<case>_components.csv` holds, for every dataset row, the BSIM4 component currents read in ngspice.
`python components_mlp/make_components.py` regenerates them (ngspice in WSL, about 3 minutes per case): it writes a
netlist that replays the exact dataset rows (`<case>_replay.cir`) and runs it. Rerun it whenever a dataset CSV changes.
