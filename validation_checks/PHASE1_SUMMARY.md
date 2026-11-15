# Phase 1 checks — summary

All checks read the project data and re-simulate in ngspice (WSL); no netlist or dataset was edited, regenerated or
used for training. Outputs of each check are in this folder.

| Check | Result | Key numbers | Output file |
|---|---|---|---|
| **A1** one parameter per `altermod` line | PASS | 85 `altermod` lines in 13 netlist files (Part A, B, C), none with more than one parameter; `gates.sub` and `cla3.sub` have none | `a1_a4_scan.txt` |
| **A2** process parameters affect the data | PASS | corr(toxe, log\|ig\|): NMOS ON −0.375, NMOS OFF −0.579, PMOS ON −0.201, PMOS OFF −0.685 (the buggy data gave ~0) | `a2_correlations.txt` |
| **A3** ground truth: process values written into the `.model` card (no `altermod`), 20 rows each | PASS | Part A currents ≤ 1.3e-5 relative (PMOS OFF `ig`/`ib` read from BSIM4 internal currents like the dataset: 3e-6 / 4e-6); Part B leakage ≤ 3.9e-6; Part C leakage ≤ 2.7e-6; node voltages ≤ 5e-6 V | `a3_ground_truth.txt`, `.csv` |
| **A4** data file dates | info | Part A CSVs 2026-11-10; Part B and Part C CSVs 2026-11-11 19:20 | `a1_a4_scan.txt` |
| **B1** corners identical everywhere | PASS | all 5 gate CSVs, `cla3.csv` and `corners.csv`: max difference 0; source is the in-netlist `rndseed=16`; `Corners_200` is used nowhere (only in `archive/old_work/`) | `b_consistency.txt` |
| **B2** CLA gates = Part B gates | PASS | every subcircuit in `gates.sub` has the same connections, L = 22n and widths as its Part B netlist; same model file | `b_consistency.txt` |
| **B3** input vectors | PASS | 6 primary inputs (a2..a0, b2..b0), no carry-in, 64 vectors; `cla3.csv` has all 64 | `b_consistency.txt` |
| **B4** missing files | note (resolved 2026-11-15) | `22nm_HP.pm` (Part B and Part C folders) and `cla3.sub` were missing locally. They were removed from the git history by the merge commit `c6a00ac`, so a plain `git checkout --` finds nothing; they were restored into the working tree (untracked, nothing staged) from commit `3e1458f` with `git restore --source=3e1458f --worktree -- <paths>`. Both model files are byte-identical to `model/22nm_HP.pm` (SHA-256 `E661D3FB...EB63`). They must be added to git for the netlists to run from a fresh clone | `b_consistency.txt` (re-run after the restore) |
| **B5** CLA SPICE sanity | PASS | 12,800 rows = 64 x 200; leakage always positive (1.6e-7 to 2.4e-5 A); sum bits and carry out equal A + B on every row | `b_consistency.txt` |
| **C1** Kirchhoff method with real SPICE currents (0 V probe on every terminal, 10 rows per case, 18 cases) | PASS | reproduces the recorded leakage to ≤ 8e-6 (worst: NOR2 `11`); 179 of 180 simulations converged | `c1_kcl_method.txt`, `.csv` |
| **C2** NAND2 / NOR2 pin convention | info | NAND2: input A drives the NMOS next to the output, B the NMOS next to ground. NOR2: A drives the PMOS next to VDD, B the PMOS next to the output. `gates.sub` is identical, so CLA pin order maps directly to Part B's A/B | – |
| **D1** validation table, components MLP excluded | info | validation alone picks GPR (NMOS ON, PMOS ON) and ResNet (NMOS OFF, PMOS OFF) on the 70/15/15 models; the final 80/10/10 table is `../partA_models/model_comparison_validation.csv` | `d1_validation_table.csv` |
| **Extra: drain–source symmetry** | FAIL (so not used) | ON cases symmetric (≤ 0.31 %); OFF cases not (dominant-current mirror error up to 3.2 % for NMOS OFF and up to 171 % for PMOS OFF in 15 random rows each; the first 10-row sample gave up to 30 % for NMOS OFF). Cause verified: the default junction geometry (`geomod = 1`, no AD/AS/PD/PS); with equal AD = AS, PD = PS or `geomod = 0` the mirror error is exactly 0. At gate operating points the device is symmetric to 2e-5 / 4.6e-5. So no symmetry preprocessing | `sym_check.txt`, `sym_cause.csv` |

## D2: deliverables 1-10 of the project document

The status of every deliverable with the exact file paths is in `../DELIVERABLES.md` (generated check: `../deliverables_check.txt`).
Summary on 2026-11-15: D1-D8 PASS; D9 partial (scripts exist and the fast path reproduces Part B and Part C exactly, but the Part C helper scripts listed in the Part C README are missing, see `REPRODUCE.md`); D10 first draft in `report/`.
