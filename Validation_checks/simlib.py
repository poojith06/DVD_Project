"""Helpers for the ngspice ground-truth checks: model cards with the process values written into the text,
netlist writing and running ngspice (WSL) on a folder of netlists."""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_CARD = (REPO / "Datasets" / "netlists" / "22nm_HP.pm").read_text()
PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]
OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"


def model_blocks(card):
    """{'nmos': (start, end), 'pmos': (start, end)} character spans of the two .model blocks."""
    starts = [m for m in re.finditer(r"(?im)^\s*\.model\s+(\w+)\s+(\w+)", card)]
    spans = {}
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(card)
        spans[m.group(1).lower()] = (m.start(), end)
    return spans


def card_with(values):
    """Model card text with values = {'nmos': {param: v}, 'pmos': {param: v}} written into the .model text."""
    card = BASE_CARD
    for dev in ("pmos", "nmos"):                      # pmos first or nmos? spans recomputed each time
        a, b = model_blocks(card)[dev]
        block = card[a:b]
        for p, v in values[dev].items():
            block, n = re.subn(rf"(?<![\w.]){p}(\s*=\s*)[-+0-9.eE]+", lambda m: f"{p}{m.group(1)}{v!r}", block,
                               count=1, flags=re.I)
            if n != 1:
                raise ValueError(f"{dev}: parameter {p} not found in the model card")
        card = card[:a] + block + card[b:]
    return card


def corner_values(row, n_prefix="n_", p_prefix="p_"):
    return {"nmos": {p: float(row[n_prefix + p]) for p in PARAMS},
            "pmos": {p: float(row[p_prefix + p]) for p in PARAMS}}


def run_folder(folder):
    """Run every *.cir in folder with ngspice in WSL; returns {name: stdout text}."""
    folder = Path(folder)
    script = "for f in *.cir; do OMP_WAIT_POLICY=passive ngspice -b \"$f\" > \"${f%.cir}.out\" 2>&1; done\n"
    (folder / "run.sh").write_text(script, newline="\n")
    w = "/mnt/c/" + str(folder).replace("C:\\", "").replace("c:\\", "").replace("\\", "/")
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-c", f"cd '{w}' && bash run.sh"], check=True)
    return {f.stem: f.read_text(errors="replace") for f in folder.glob("*.out")}


def printed(text):
    """Values of 'print' lines in ngspice output, in order: [(name, value)]."""
    return [(m.group(1), float(m.group(2))) for m in
            re.finditer(r"(?m)^\s*(-?i\([\w.#]+\)|v\([\w.#]+\))\s*=\s*([-+0-9.eE]+)", text)]
