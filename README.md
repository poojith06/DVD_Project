# ML Models: Transistor Terminal Currents

Trains machine learning models that predict a transistor's four terminal currents (`id`, `ig`, `is`, `ib`) from its 12 inputs. This is Part A of the Digital VLSI Design leakage project. There is one dataset and one saved model for each of the four cases: NMOS ON, NMOS OFF, PMOS ON and PMOS OFF.

For how the code works and why, see [ML models/explanation.md](ML%20models/explanation.md).

## Folder layout

```
DVD_Project/
├── README.md
├── requirements.txt
├── data/                       the four CSV files
├── results/                    created by select_best.py: the final results
│   ├── nmos_on/                model.joblib, parity.png, validation_comparison.csv, test_metrics.csv
│   ├── nmos_off/               the same files
│   ├── pmos_on/                the same files
│   └── pmos_off/               the same files
└── ML models/
    ├── explanation.md          how the code works and why
    ├── select_best.py          pick the best trained model for each case
    ├── predict.py              use a saved model
    ├── mlp/                    neural network
    │   ├── model.py            the model and its hyperparameters
    │   ├── train.py            train this model
    │   ├── results/            created by train.py
    │   └── saved_models/       created by train.py
    ├── xgboost_regressor/      XGBoost, the same files
    ├── random_forest/          Random Forest, the same files
    ├── ridge_polynomial/       ridge polynomial, the same files
    └── common/                 code shared by every model
```

Install from the main directory. Every other command below is run from inside the `ML models` folder:

```
cd "ML models"
```

## 1. Install

You need Python 3.10 or newer. From the main `DVD_Project` directory:

```
pip install -r requirements.txt
```

If pip fails with `CERTIFICATE_VERIFY_FAILED`, add `--use-feature=truststore`.

## 2. Add your data

Put these four files in the `data/` folder in the main `DVD_Project` directory, next to `ML models`:

```
data/nmos_on.csv    data/nmos_off.csv    data/pmos_on.csv    data/pmos_off.csv
```

Each file needs these 16 columns, in any order:

```
vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib
```

Units are volts for voltages, °C for `temp`, metres for `w`, amps for currents, and BSIM model units for the process parameters. The currents keep their SPICE signs.

The code checks `vg` and `vb` against the case. For example, `nmos_on` needs `vg = vdd` and `vb = 0` on every row. A CSV saved under the wrong name is rejected with an error. Rows with NaN or infinite values are dropped with a message.

To use a different folder, pass `--data-dir path/to/folder`.

## 3. Train

Each model is trained on its own, never as part of a combination. Train every model on the four cases, look at the results, then pick the best model for each case.

### Step 1: train each model separately

```
python mlp/train.py
python xgboost_regressor/train.py
python random_forest/train.py
python ridge_polynomial/train.py
```

Each script trains its model on all four cases (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF) and writes to its own folder:

| Output | Contents |
|---|---|
| `<model>/saved_models/<case>.joblib` | The trained model for each case |
| `<model>/results/validation_metrics.csv` | R², median error and 95th-percentile error on the validation set |
| `<model>/results/test_metrics.csv` | The same on the test set, plus sign accuracy and KCL residual |
| `<model>/results/parity_<case>.png` | Predicted vs SPICE currents on the test set |

To train one case only, add `--cases pmos_off`.

### Step 2: pick the best model for each case

```
python select_best.py
```

This reads the results of every model you have trained, then for each case picks the model with the lowest validation 95th-percentile error. Median error breaks ties. The final results go in the main directory's `results/` folder, with one folder for each case. Each case folder holds the chosen model and its results:

| File | Contents |
|---|---|
| `results/<case>/model.joblib` | The chosen, saved model (deliverable 5) |
| `results/<case>/validation_comparison.csv` | All the models you trained for this case, compared on the validation set (this case's part of deliverable 4) |
| `results/<case>/test_metrics.csv` | The chosen model's test results, including which model it is (deliverable 5) |
| `results/<case>/parity.png` | The chosen model's predicted vs SPICE plot (deliverable 5) |

`<case>` is `nmos_on`, `nmos_off`, `pmos_on` or `pmos_off`. The command also prints the comparison for all four cases together.

The comparison uses validation results only, so the test set stays untouched by the choice. The test numbers appear in each model's own `test_metrics.csv`, but don't choose by them.

If you prefer a different model for a case, override it:

```
python select_best.py --choose pmos_off=mlp nmos_on=xgb
```

The model keys are `mlp`, `xgb`, `rf` and `ridge`. A case you don't override still uses the best score. You can run `select_best.py` after training only some of the models, and it compares whichever ones it finds. Running it again overwrites the files in `results/`.

### Options

These work with every model's `train.py`:

| Option | Meaning | Default |
|---|---|---|
| `--cases nmos_off pmos_on` | Train only these cases | all four |
| `--split 0.8 0.1 0.1` | Train, validation and test fractions | `0.7 0.15 0.15` |
| `--mlp-epochs 50` | Override the MLP epoch count (`mlp/train.py` only) | 500 |
| `--seed 7` | Random seed for the split and the model | 42 |
| `--data-dir`, `--results-dir`, `--models-dir` | Change the folders | `../data` and the model's own folder |

Use the same `--split` and `--seed` for every model, so they are compared on the same data. Run any `train.py --help` for the full list.

A quick test run before the full one:

```
python mlp/train.py --cases nmos_off --mlp-epochs 20
```

On 12 CPU threads with 10,000 rows per case, the MLP takes about 1.5 to 2 minutes per case and the other models under 30 seconds per case.

### Tuning a model

Each model's hyperparameters are a dictionary at the top of its `model.py`:

| File | Dictionary |
|---|---|
| `mlp/model.py` | `MLP_PARAMS` |
| `xgboost_regressor/model.py` | `XGB_PARAMS` |
| `random_forest/model.py` | `RF_PARAMS` |
| `ridge_polynomial/model.py` | `RIDGE_PARAMS` |

Edit the values and train again. Shared settings, such as the columns, the split and the seed, are in `common/config.py`.

## 4. Predict

### From the command line

Make a CSV with the 12 input columns (no current columns needed), then run this from the `ML models` folder:

```
python predict.py --case nmos_off --input samples.csv --output currents.csv
```

The output CSV has the columns `id, ig, is, ib`, in amps. This uses the chosen model in `results/<case>/`. To use a model straight from its own folder, add `--models-dir mlp/saved_models`.

### From Python (for Part B)

Add the `ML models` folder to the import path first. A saved model refers to the code in `common/` and the model folders, so Python needs to find them when it loads the file.

```python
import sys
sys.path.insert(0, "ML models")   # relative to the main DVD_Project folder

import pandas as pd
from predict import load_surrogate

model = load_surrogate("nmos_off")   # loads results/nmos_off/model.joblib, the chosen model

terminals = pd.DataFrame([{
    "vdd": 1.0, "temp": 25.0, "w": 44e-9,
    "vd": 0.1, "vs": 0.0, "vg": 0.0, "vb": 0.0,
    "vth0": 0.3692, "toxe": 6.5e-10, "ndep": 1.2e19, "u0": 0.0181, "xj": 7.2e-9,
}])
currents = model.predict(terminals)   # DataFrame with columns id, ig, is, ib
```

`predict` takes many rows at once, so pass a whole gate or corner as one DataFrame. The four currents of each row sum to zero. Pass `kirchhoff=False` to see the uncorrected values.

To load from a model folder, use `load_surrogate("nmos_off", "ML models/mlp/saved_models")`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Missing dataset ...` | The CSV isn't in the main folder's `data/`. Check the file name and `--data-dir`. |
| `FileNotFoundError` for `results/<case>/model.joblib` | `select_best.py` hasn't been run yet. |
| `nmos_on: vg should equal vdd on every row` | The CSV was saved under the wrong case name, or its gate and body voltages don't match the spec. |
| `ModuleNotFoundError: xgboost` | Run `pip install -r requirements.txt`. The other models work without it. |
| `ModuleNotFoundError: common` or `mlp` when loading a model | The `ML models` folder isn't on `sys.path`. See the Python snippet above. |
| `CERTIFICATE_VERIFY_FAILED` from pip | Add `--use-feature=truststore`. |
| `No trained models found` from `select_best.py` | Train at least one model first, for example `python mlp/train.py`. |
| `model 'xgb' has not been trained for this case` | The model you passed to `--choose` hasn't been trained on that case yet. |
| MLP training is slow | Lower `--mlp-epochs` for a trial run. |
| A 136 MB model file | Random Forest saves large files when it wins a case. Don't commit it to git. |

## Status

The code has been run end to end on synthetic data only, not SPICE data. Expect to review the first real results before relying on them. See "Known limitations" in [ML models/explanation.md](ML%20models/explanation.md).
