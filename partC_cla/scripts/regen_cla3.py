"""Regenerate the Part C SPICE data (3-bit CLA, 64 input vectors x PVT corners) into a NEW file.

The original netlist partC_cla/netlists/cla3_ngspice.cir is used unchanged except for one thing: the lines that draw the
random corner values with sunif(0) are replaced by the 200 corners recorded in partC_cla/data/corners.csv. Reason: the random
stream of `set rndseed=16` depends on the ngspice build, so another ngspice build draws different corners; replaying the
recorded corners makes the result independent of the build. Use --native to run the netlist with its own rndseed instead
(then the corners, and therefore all numbers, differ from the saved data on a different ngspice build).

Writes --out (default pipeline_out/partC/cla3_regen.csv, never partC_cla/data/cla3.csv) and, with --compare, prints the
largest relative difference to partC_cla/data/cla3.csv for the rows produced (leakage and all 20 recorded node voltages).

  python partC_cla/scripts/regen_cla3.py --corners 3 --compare      # 3 corners x 64 vectors = 192 rows, a few seconds
  python partC_cla/scripts/regen_cla3.py --compare                  # all 200 corners = 12,800 rows

ngspice is taken from PATH, or from WSL (distribution --wsl-distro, default Ubuntu) on Windows.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CORNER_COLS = {"vdd": "vdd", "t": "temp", "n1": "n_vth0", "n2": "n_toxe", "n3": "n_ndep", "n4": "n_u0", "n5": "n_xj",
               "p1": "p_vth0", "p2": "p_toxe", "p3": "p_ndep", "p4": "p_u0", "p5": "p_xj"}


def vector_lines(corners, n):
    """ngspice 'compose' lines (negative numbers inside a list are not parsed reliably: all-negative vectors are stored
    as magnitudes, mixed-sign vectors with an offset of 1000)."""
    vecs, repl = [], {}
    for var, col in CORNER_COLS.items():
        xs = [float(x) for x in corners[col][:n]]
        if all(x < 0 for x in xs):
            vecs.append(f"compose vec_{var} values " + " ".join(repr(-x) for x in xs))
            repl[var] = f"  let {var} = -vec_{var}[c]"
        elif any(x < 0 for x in xs):
            vecs.append(f"compose vec_{var} values " + " ".join(repr(x + 1000.0) for x in xs))
            repl[var] = f"  let {var} = vec_{var}[c] - 1000"
        else:
            vecs.append(f"compose vec_{var} values " + " ".join(repr(x) for x in xs))
            repl[var] = f"  let {var} = vec_{var}[c]"
    return vecs, repl


def build_netlist(text, corners, n, native):
    text = re.sub(r"let ncorner = \d+", f"let ncorner = {n}", text)
    if native:
        return text
    vecs, repl = vector_lines(corners, n)
    out = []
    for ln in text.splitlines():
        m = re.match(r"\s*let (\w+) = .*sunif\(0\)", ln)
        out.append(repl[m.group(1)] if m and m.group(1) in repl else ln)
    text = "\n".join(out) + "\n"
    return text.replace(".control\n", ".control\n" + "\n".join(vecs) + "\n", 1)


def run_ngspice(workdir, name, distro):
    if shutil.which("ngspice"):
        cmd = ["bash", "-c", f"cd '{workdir}' && ngspice -b {name}"] if sys.platform != "win32" else None
        if cmd is None:
            subprocess.run(["ngspice", "-b", name], cwd=workdir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    p = str(workdir.resolve()).replace("\\", "/")
    wslpath = "/mnt/" + p[0].lower() + p[2:] if re.match(r"^[A-Za-z]:", p) else p
    subprocess.run(["wsl", "-d", distro, "--", "bash", "-c", f"cd '{wslpath}/partC_cla/netlists' && OMP_WAIT_POLICY=passive ngspice -b {name}"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corners", type=int, default=200, help="number of corners (first N of data/corners.csv), default 200")
    ap.add_argument("--out", type=Path, default=REPO / "pipeline_out" / "partC" / "cla3_regen.csv")
    ap.add_argument("--native", action="store_true", help="use the netlist's own rndseed instead of the recorded corners")
    ap.add_argument("--compare", action="store_true", help="compare with partC_cla/data/cla3.csv")
    ap.add_argument("--wsl-distro", default="Ubuntu")
    a = ap.parse_args()
    if a.out.resolve() == (REPO / "partC_cla" / "data" / "cla3.csv").resolve():
        sys.exit("refusing to overwrite the saved dataset partC_cla/data/cla3.csv")

    work = REPO / "pipeline_out" / "partC" / "regen_work"
    if work.exists():
        shutil.rmtree(work)
    (work / "model").mkdir(parents=True)
    (work / "partC_cla" / "data").mkdir(parents=True)
    (work / "partC_cla" / "netlists").mkdir(parents=True)
    shutil.copy(REPO / "model" / "22nm_HP.pm", work / "model")
    for f in (REPO / "partC_cla" / "netlists").glob("*.sub"):
        shutil.copy(f, work / "partC_cla" / "netlists")
    corners = pd.read_csv(REPO / "partC_cla" / "data" / "corners.csv")
    net = build_netlist((REPO / "partC_cla" / "netlists" / "cla3_ngspice.cir").read_text(), corners, a.corners, a.native)
    (work / "partC_cla" / "netlists" / "cla3_ngspice.cir").write_text(net, newline="\n")
    run_ngspice(work, "cla3_ngspice.cir", a.wsl_distro)

    new = pd.read_csv(work / "partC_cla" / "data" / "cla3.csv")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(new)} rows ({a.corners} corners x 64 vectors)")
    if a.compare:
        ref = pd.read_csv(REPO / "partC_cla" / "data" / "cla3.csv").iloc[:len(new)]
        keys = ["corner", "A", "B"]
        assert (new[keys].to_numpy() == ref[keys].to_numpy()).all(), "row order differs"
        cols = [c for c in ref.columns if c not in keys + ["A2", "A1", "A0", "B2", "B1", "B0"]]
        rel = {c: float((np.abs(new[c] - ref[c]) / np.maximum(np.abs(ref[c]), 1e-18)).max()) for c in cols}
        worst = max(rel, key=rel.get)
        print(f"compared with data/cla3.csv: {len(cols)} columns, max relative difference {rel[worst]:.2e} (column {worst}); "
              f"leak: {rel['leak']:.2e}")
        print("REGENERATION: " + ("PASS" if rel[worst] <= 1e-5 else "FAIL"))


if __name__ == "__main__":
    main()
