# validation_checks

Scripts that check the data and the pipeline, each with its text output next to it (`<name>.txt`). `PHASE1_SUMMARY.md`
summarises the first round of checks (parameter application, ground truth, corner consistency, Kirchhoff method).

| Script | Purpose |
|---|---|
| `data_check.py` | Part A data check (Kirchhoff residuals, trend plots) -> `partA_transistor/data_check/` |
| `resim_netlists.py` | re-simulates a few rows of every netlist (after the restructuring) and compares with the CSVs -> `netlist_resim_check.txt` |
| `deliverables_check.py` | status of deliverables 1-10 -> `docs/deliverables_check.txt` |
| `make_report_notes.py` | writes `docs/REPORT_NOTES.md` from the result files |
| `a*.py`, `b_consistency.py`, `c1_kcl_method.py`, `sym_*.py` | parameter, ground-truth, corner and symmetry checks |
| other scripts | analyses behind single report numbers (see the header of each file) |

`validation_checks/sim/` (scratch SPICE files written by the checks) is not tracked. ngspice is run through WSL (`simlib.py`).
