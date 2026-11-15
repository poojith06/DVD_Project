"""Compare the outputs of a pipeline run with the submitted ones (results).

    python ml/reproduce_check.py --out pipeline_out

Part B: pipeline_out/gate_results/gate_predictions.csv vs final_v2/partB_gates/gate_predictions.csv
Part C: pipeline_out/cla_results/cla3_predictions.csv  vs final_v2/partC/cla3_predictions.csv
Prints the headline accuracies and the largest relative difference of the ML leakages; PASS if the headline numbers
agree to 2 decimals and the ML leakages agree to 1e-9 relative (float64 GPR on a GPU vs a CPU can differ at ~1e-15).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "results"


def acc(df, spice, ml):
    e = 100 * (df[ml] - df[spice]) / df[spice]
    return 100 - e.abs().mean(), 100 - e.abs().median()


def compare(label, new_path, ref_path, key, expect_mean):
    if not new_path.exists():
        print(f"{label}: MISSING {new_path}")
        return False
    new, ref = pd.read_csv(new_path), pd.read_csv(ref_path)
    new = new.set_index(key).sort_index()
    ref = ref.set_index(key).sort_index()
    same_rows = new.index.equals(ref.index)
    mean_new, med_new = acc(new, "leak_spice", "leak_ml")
    mean_ref, med_ref = acc(ref, "leak_spice", "leak_ml")
    diff = float(np.max(np.abs(new.leak_ml / ref.leak_ml - 1))) if same_rows else float("nan")
    spice_same = bool(np.allclose(new.leak_spice, ref.leak_spice, rtol=1e-12)) if same_rows else False
    ok = same_rows and spice_same and f"{mean_new:.2f}" == f"{mean_ref:.2f}" and diff < 1e-9
    print(f"{label}: rows {len(new)} (same keys as submitted: {same_rows}); SPICE leakage identical to submitted: {spice_same}")
    print(f"   mean accuracy {mean_new:.2f} % (submitted {mean_ref:.2f} %, expected {expect_mean}); "
          f"median accuracy {med_new:.2f} % (submitted {med_ref:.2f} %)")
    print(f"   largest relative difference of the ML leakage vs submitted: {diff:.2e}  ->  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="pipeline output folder")
    args = p.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    ok_b = compare("Part B", out / "gate_results" / "gate_predictions.csv",
                   REF / "partB" / "gate_predictions.csv", ["gate", "inputs", "corner"], "98.87 %")
    ok_c = compare("Part C", out / "cla_results" / "cla3_predictions.csv",
                   REF / "partC" / "cla3_predictions.csv", ["A", "B", "corner"], "99.00 %")
    print("REPRODUCTION:", "PASS" if ok_b and ok_c else "FAIL")
    sys.exit(0 if ok_b and ok_c else 1)


if __name__ == "__main__":
    main()
