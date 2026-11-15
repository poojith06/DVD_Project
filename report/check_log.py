"""Summarise build.log: page count, errors, undefined references/citations and overfull boxes with the table/section file they occur in."""
import re
import sys
from pathlib import Path

log = (Path(__file__).resolve().parent / "build.log").read_text(errors="replace")
log = log[log.rfind("This is pdfTeX"):]          # final pass only (references are resolved by then)
pages = re.findall(r"Output written on main\.pdf \((\d+) pages", log)
print("pages:", pages[-1] if pages else "no PDF")
print("errors:", len(re.findall(r"(?m)^!", log)))
print("undefined references:", sorted(set(re.findall(r"Reference `([^']+)' on page", log))))
print("undefined citations:", sorted(set(re.findall(r"Citation `([^']+)' on page", log))))
print("multiply defined labels:", sorted(set(re.findall(r"Label `([^']+)' multiply defined", log))))
last = {}
cur = "main"
over = []
for line in log.splitlines():
    m = re.findall(r"\((?:\./)?((?:tables|sections)/[\w\.]+\.tex)", line)
    if m:
        cur = m[-1]
    m2 = re.match(r"Overfull \\hbox \(([\d.]+)pt too wide\)(.*)", line)
    if m2 and "output is active" not in line:
        over.append((float(m2.group(1)), cur, m2.group(2).strip()))
print("overfull hboxes:", len(over))
by = {}
for w, f, c in over:
    by.setdefault(f, []).append(w)
for f, ws in sorted(by.items(), key=lambda kv: -max(kv[1])):
    print(f"  {f:34s} {len(ws)} boxes, max {max(ws):.1f} pt")
