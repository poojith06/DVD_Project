"""How many significant digits do the numbers in the data CSVs carry? (6 are expected from ngspice's default
number printing, `echo "$&x"` -> 6 digits.)"""
import csv
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = ["partA_transistor/data/nmos_on.csv", "partA_transistor/data/nmos_off.csv", "partA_transistor/data/pmos_on.csv",
         "partA_transistor/data/pmos_off.csv", "partB_gates/data/nand2.csv", "partC_cla/data/cla3.csv"]


def sig_digits(s):
    mantissa = re.sub(r"[eE].*", "", s.strip()).replace("-", "").replace("+", "").replace(".", "")
    mantissa = mantissa.lstrip("0")
    return max(len(mantissa), 1)


for f in FILES:
    rows = list(csv.reader(open(REPO / f)))
    head, body = rows[0], rows[1:]
    print(f"== {f}  ({len(body)} rows)")
    over = []
    for j, h in enumerate(head):
        digs = [sig_digits(r[j]) for r in body if r[j].strip()]
        n_over = sum(d > 6 for d in digs)
        if n_over:
            ex = next(r[j] for r in body if sig_digits(r[j]) > 6)
            over.append(f"{h}: {n_over} rows with >6 digits (max {max(digs)}), e.g. {ex}")
    print("   all columns <= 6 significant digits" if not over else "   " + "\n   ".join(over))
