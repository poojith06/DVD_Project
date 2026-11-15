"""Settings shared by every script in ml."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # partA_transistor_models/step2_training/code/
REPO = ROOT.parents[2]                            # repository root
DATA_DIR = REPO / "partA_transistor_models" / "step1_data_generation" / "data"   # the four Part A CSVs (10,000 rows each)
SAVED_MODELS = ROOT.parent / "saved_models"       # the four final models (<case>.joblib)
PIPELINE_OUT = REPO / "pipeline_out"              # default output folder of the evaluation scripts
import os as _os
# Part B / Part C SPICE reference (env GATE_DATA_DIR / CLA_DIR let run_all.ps1 point at regenerated SPICE data)
GATE_DATA = Path(_os.environ.get("GATE_DATA_DIR", REPO / "partB_gates" / "step3_spice_reference" / "data"))
CLA_DIR = Path(_os.environ.get("CLA_DIR", REPO / "partC_cla" / "step6_spice_reference"))   # holds netlists/ and data/
RUNS_DIR = ROOT / "runs"                         # one sub-folder per training run (--tag), git-ignored
import os
# selection output of select_best.py (git-ignored scratch; env ML_V2_RESULTS picks another folder, e.g. for the tuned run)
RESULTS_DIR = ROOT / os.environ.get("ML_V2_RESULTS", "results")

SEED = 42                     # data split
MODEL_SEED = 0                # network initialisation, batch order, GPR row subset
SPLIT = (0.80, 0.10, 0.10)    # train / validation / test - the spec's alternative 80:10:10 split

# The 12 inputs and 4 outputs (column names from the spec).
INPUT_COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
TARGET_COLUMNS = ["id", "ig", "is", "ib"]
ACTIVE_COLUMNS = [c for c in INPUT_COLUMNS if c not in ("vg", "vb")]   # vg, vb are fixed by the case

# case -> (polarity, gate rail, body rail)
CASES = {
    "nmos_on": ("nmos", "vdd", "gnd"),
    "nmos_off": ("nmos", "gnd", "gnd"),
    "pmos_on": ("pmos", "gnd", "vdd"),
    "pmos_off": ("pmos", "vdd", "vdd"),
}
CURRENT_FLOOR = 1e-18         # = abstol in every netlist
GROUPS = {"id+is": [0, 2], "ig": [1], "ib": [3]}   # one network per output group

MODEL_TYPES = ["gpr", "hybrid_analytic_mlp", "resnet_mlp", "pinn"]

# Near-rail training weight (only the training loss of the neural networks; the data itself is unchanged).
# rail distance r = dist(vd) + dist(vs), dist(v) = distance of v from the nearer rail (0 or VDD).
# Logic gates use transistors at r ~ 0 (every terminal at a rail), which uniform sampling covers sparsely.
# weight = 1 + RAIL_WEIGHT * exp(-r / RAIL_TAU), normalised to mean 1. RAIL_WEIGHT is chosen on validation.
RAIL_TAU = 0.05               # V
NEAR_RAIL_V = 0.2             # validation check: rows with r < 0.2 V
