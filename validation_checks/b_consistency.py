"""Checks B1-B5: consistency between Part B and Part C (corners, gate sizing, input vectors, files, CLA sanity)."""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GB, PC = REPO / "partB_gates", REPO / "partC_cla"
COLS = ["vdd", "temp"] + [f"{d}_{p}" for d in "np" for p in ["vth0", "toxe", "ndep", "u0", "xj"]]


def git_show(path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{path}"], capture_output=True, text=True).stdout


# ---------------- B1 corners
corners = pd.read_csv(PC / "data" / "corners.csv").set_index("corner")[COLS]
ok = True
for g in ["inv", "nand2", "nor2", "and2", "or2"]:
    d = pd.read_csv(GB / "data" / f"{g}.csv")
    for c, grp in d.groupby("corner"):
        if not (grp[COLS].nunique() == 1).all():
            ok = False
            print(f"B1 {g}: corner {c} not constant over its input rows")
    per = d.groupby("corner")[COLS].first()
    diff = (per - corners.loc[per.index]).abs().max().max()
    print(f"B1 {g}: {per.shape[0]} corners, max |difference| from PartC corners.csv = {diff:.1e}")
    ok &= diff == 0
cla = pd.read_csv(PC / "data" / "cla3.csv")
diff_c = (cla.groupby("corner")[COLS].first() - corners).abs().max().max()
print(f"B1 cla3.csv: max |difference| from corners.csv = {diff_c:.1e}")
ok &= diff_c == 0
print("B1 corners identical everywhere:", ok)
for f in [*sorted((GB / "netlists").glob("*.cir")), PC / "netlists" / "cla3_ngspice.cir"]:
    t = f.read_text()
    seed = re.findall(r"rndseed\s*=\s*(\d+)", t)
    print(f"   {f.name}: rndseed {seed}, uses corners.inc: {'corners.inc' in t}")
active = [REPO / "partA_transistor", GB, PC, REPO / "ml"]
hits = [p for a in active for p in a.rglob("*") if p.is_file() and p.suffix in (".cir", ".sub", ".net", ".py", ".sh")
        and ("corners.inc" in p.read_text(errors="ignore") or "Corners_200" in p.read_text(errors="ignore"))]
print("   active files referring to corners.inc / Corners_200:", [str(h.relative_to(REPO)) for h in hits] or "none")
print("   Corners_200 locations:", [str(p.relative_to(REPO)) for p in (REPO / "Corners_200", REPO / "archive/old_work" / "Corners_200") if p.exists()])

# ---------------- B2 gate sizing: gates.sub vs Part B netlists
def devices(text):
    return [tuple(l.split()[i] for i in (1, 2, 3, 4, 5)) + (re.search(r"L=(\S+)", l).group(1), re.search(r"W=(\S+)", l).group(1))
            for l in text.splitlines() if re.match(r"^\s*M\w*\s", l)]

gates_sub = (PC / "netlists" / "gates.sub").read_text()
blocks = dict(re.findall(r"(?is)\.subckt\s+(\w+)[^\n]*\n(.*?)\.ends", gates_sub))
for g in ["inv", "nand2", "nor2", "and2", "or2"]:
    b = devices((GB / "netlists" / f"{g}_ngspice.cir").read_text())
    c = devices(blocks[g.upper()])
    node = lambda x: "0" if x == "gnd" else x
    norm = lambda devs: sorted(tuple(node(x) for x in d[:4]) + d[4:] for d in devs)   # d, g, s, b, model, L, W
    print(f"B2 {g.upper()}: {len(b)} transistors; same connections, L, W as Part B: {norm(b) == norm(c)}"
          f"   (W list {[d[6] for d in b]})")
for f in [GB / "netlists" / "and2_ngspice.cir", PC / "netlists" / "cla3_ngspice.cir"]:
    inc = re.findall(r"(?im)^\.include\s+(\S+)", f.read_text())
    print(f"   {f.name}: includes {inc}")

# ---------------- B3 input vectors
sub = (PC / "netlists" / "cla3.sub").read_text() if (PC / "netlists" / "cla3.sub").exists() else git_show(
    "partC_cla/netlists/cla3.sub")
ports = re.search(r"(?im)^\.subckt\s+CLA3\s+(.*)$", sub).group(1).split()
inputs = [p for p in ports if re.fullmatch(r"[ab]\d|cin|c0", p)]
print(f"B3 CLA3 ports: {ports}\n   primary inputs: {inputs} -> {2 ** len(inputs)} vectors needed; carry-in present: "
      f"{any(p in ('cin', 'c0') for p in ports)}; cla3.csv has {cla.groupby(['A', 'B']).ngroups} distinct vectors")

# ---------------- B4 missing files
for rel in ["partB_gates/netlists/22nm_HP.pm", "partC_cla/netlists/22nm_HP.pm",
            "partC_cla/netlists/cla3.sub", "partC_cla/netlists/gates.sub"]:
    tracked = bool(subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel], capture_output=True).returncode == 0)
    in_head = bool(subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"HEAD:{rel}"], capture_output=True).returncode == 0)
    print(f"B4 {rel}: exists locally {(REPO / rel).exists()}, in the git index {tracked}, in the current HEAD commit {in_head}")

# ---------------- B5 CLA SPICE sanity
n_vec = cla.groupby(["A", "B"]).ngroups
bits = lambda col: (cla[col] > cla.vdd / 2).astype(int)
total = bits("v_s0") + 2 * bits("v_s1") + 4 * bits("v_s2") + 8 * bits("v_cout")
print(f"B5 rows {len(cla)} = {n_vec} vectors x {cla.corner.nunique()} corners: {len(cla) == n_vec * cla.corner.nunique()}; "
      f"leak > 0 everywhere: {(cla.leak > 0).all()} (min {cla.leak.min():.2e}, max {cla.leak.max():.2e} A); "
      f"A = 4*A2+2*A1+A0: {(cla.A == 4 * cla.A2 + 2 * cla.A1 + cla.A0).all()}; "
      f"outputs = A + B on every row: {(total == cla.A + cla.B).all()} ({(total != cla.A + cla.B).sum()} wrong)")
