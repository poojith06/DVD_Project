"""Checks A1 and A4.
A1: every `altermod` line in the Part A / B / C netlists must set exactly ONE parameter (ngspice applies only the
    first parameter on a line). cla3.sub is read from git if it is missing locally.
A4: modification dates of all data CSVs."""
import re
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
files = [p for d in ("Datasets", "Gates_Part-B", "PartC_3bit CLA adder") for p in (REPO / d).rglob("*")
         if p.is_file() and p.suffix in (".cir", ".sub", ".net")]
total = multi = 0
print("A1: altermod lines per file (parameters set per line must be exactly 1)")
for f in sorted(files):
    n = 0
    for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
        if re.match(r"^\s*altermod\s", line):
            n += 1
            k = len(re.findall(r"(\w+)\s*=", re.sub(r"^\s*altermod\s+\S+\s*", "", line)))
            if k != 1:
                multi += 1
                print(f"  MULTI-PARAMETER LINE {f.relative_to(REPO)}:{i}: {line.strip()}")
    total += n
    print(f"  {str(f.relative_to(REPO)):62s} altermod lines: {n}")
sub = REPO / "PartC_3bit CLA adder" / "netlists" / "cla3.sub"
if not sub.exists():
    txt = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:PartC_3bit CLA adder/netlists/cla3.sub"],
                         capture_output=True, text=True).stdout
    print(f"  cla3.sub (read from git HEAD, missing locally)      altermod lines: {len(re.findall('altermod', txt))}")
print(f"A1 result: {total} altermod lines in {len(files)} files, {multi} with more than one parameter -> "
      f"{'PASS' if multi == 0 else 'FAIL'}")
print("\nA4: data CSV modification dates")
for d in ("Datasets/data", "Gates_Part-B/data", "PartC_3bit CLA adder/data"):
    for f in sorted((REPO / d).glob("*.csv")):
        print(f"  {str(f.relative_to(REPO)):58s} {datetime.fromtimestamp(f.stat().st_mtime):%Y-%m-%d %H:%M}  "
              f"{f.stat().st_size / 1024:8.0f} KB")
