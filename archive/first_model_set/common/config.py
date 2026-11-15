"""Constants shared by every model. Per-model settings live in each model's folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the ML_Models folder
DATA_DIR = ROOT.parent / "Datasets" / "data"  # the four dataset CSVs, one per case
# Final results: one folder per case inside this one, each holding that case's chosen model.
RESULTS_DIR = ROOT / "results"
MODEL_FILENAME = "model.joblib"  # the chosen model's file name inside a case's results folder

SEED = 42  # data split (same split as the benchmark in Model_comparison/)
MODEL_SEED = 0  # network initialisation, batch order and GPR row subset

# The 12 inputs and 4 outputs, using the CSV column names from the project spec.
# Units: V, degC and metres for vdd/temp/w, BSIM model units for the process parameters.
INPUT_COLUMNS = [
    "vdd", "temp", "w", "vd", "vs", "vg", "vb",
    "vth0", "toxe", "ndep", "u0", "xj",
]
TARGET_COLUMNS = ["id", "ig", "is", "ib"]

# vg and vb are fixed by the case, so they carry no information as raw columns.
# They reach the models through the engineered features instead.
ACTIVE_COLUMNS = [c for c in INPUT_COLUMNS if c not in ("vg", "vb")]

# case name -> (polarity, gate voltage rail, body voltage rail)
CASES = {
    "nmos_on": ("nmos", "vdd", "gnd"),
    "nmos_off": ("nmos", "gnd", "gnd"),
    "pmos_on": ("pmos", "gnd", "vdd"),
    "pmos_off": ("pmos", "vdd", "vdd"),
}

# Matches the abstol=1e-18 option every netlist uses. Currents below this are noise.
CURRENT_FLOOR = 1e-18

SPLIT = (0.70, 0.15, 0.15)  # train, validation, test

# Output groups for the neural networks: one network per group.
GROUPS = {"id+is": [0, 2], "ig": [1], "ib": [3]}
