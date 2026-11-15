# Leakage-current surrogate models for a 22 nm HP gate-level circuit

Team **Dvd Angels** · IIIT Hyderabad · Digital VLSI Design (Semester 5) course project

We replace SPICE for leakage-current estimation with machine-learning surrogates of single transistors. Four models
(NMOS ON, NMOS OFF, PMOS ON, PMOS OFF) predict the four terminal currents (id, ig, is, ib) of a PTM 22 nm HP transistor
(VDD = 1 V, L = 22 nm) from the bias, temperature, width and five process parameters. The leakage of a gate (INV, NAND2,
NOR2, AND2, OR2) is assembled from the transistor predictions with Kirchhoff's current law on the supply node, and the
leakage of a 3-bit carry-lookahead adder is the sum of its 33 gate leakages. Models are trained on 10,000 ngspice runs
per transistor case and evaluated, without any retraining, on 3,600 gate simulations (Part B) and 12,800 adder
simulations (Part C).

## Headline results

| | Part B: 18 gate cases, 3,600 simulations | Part C: 3-bit CLA, 12,800 simulations |
|---|---|---|
| Mean accuracy (100 − mean absolute percentage error) | **98.87 %** | **99.00 %** |
| Median accuracy | 99.18 % | 99.17 % |
| Ceiling (same sum built from the SPICE gate leakages) | – | 99.16 % |

Final models: NMOS ON hybrid analytic + MLP, NMOS OFF ResNet MLP (5 blocks), PMOS ON Gaussian process, PMOS OFF ResNet
MLP (5 blocks); split 80:10:10, chosen on validation data only. Details: `results/README.md`, report: `report/DVD_Angels_Report.pdf`.

## Folder map

| Folder / file | Contents |
|---|---|
| `partA_transistor/` | single-transistor netlists and the four 10,000-row datasets, data check, PMOS OFF read-out check |
| `partB_gates/` | gate netlists and SPICE data (5 gates, 200 corners) |
| `partC_cla/` | 3-bit CLA netlists, SPICE data (64 vectors × 200 corners), regeneration script |
| `model/` | PTM 22 nm HP model card `22nm_HP.pm` (included by every netlist) |
| `ml/` | model code, training, selection, Part B and Part C evaluation |
| `results/` | final models (`results/models/`), result tables, parity plots, analysis outputs |
| `validation_checks/` | data and consistency checks, report-number scripts, netlist re-simulation check |
| `report/` | LaTeX sources and the final report (`DVD_Angels_Report.pdf`) |
| `docs/` | project description, deliverables status, report notes |
| `archive/` | earlier model sets, benchmarks and working files (evidence for the report, not needed to reproduce) |
| `run_pipeline.ps1`, `run_pipeline.sh`, `REPRODUCE.md` | reproduction scripts and the step list with seeds |

## Quick start (fast path, about a minute)

Re-evaluates the saved models on the saved SPICE data and checks that the headline numbers are reproduced exactly:

```powershell
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Fast      # Windows
```
```bash
bash run_pipeline.sh --fast                                          # Linux / macOS
```

Expected last lines: Part B 98.87 %, Part C 99.00 %, `REPRODUCTION: PASS`. All outputs go to `pipeline_out/`.
Regenerating SPICE data (ngspice 44) or retraining takes hours; see `REPRODUCE.md`.

## Assumptions we made

1. **Accuracy** = 100 − mean absolute percentage error over all simulations (other common definitions are compared in the report).
2. **Gate leakage** = current drawn from the VDD supply; currents of the input sources are not included.
3. **Training data**: the models use only the 16 dataset columns of the specification (no internal BSIM4 component currents).
4. **PMOS OFF gate/body currents** are read from the BSIM4 instance currents, because the source-branch currents are rounded to about 2.2e-16 A.

## Team

| Name | Roll number | Email |
|---|---|---|
| Mididoddi Sai Poojith | 2025122010 | mididoddi.poojith@research.iiit.ac.in |
| Devang Bordoloi | 2025122003 | devang.bordoloi@research.iiit.ac.in |
| Vajrapu Venkata Sai Aneesh | 2025122002 | aneesh.vajrapu@research.iiit.ac.in |
| Apanna Nihaal | 2025122007 | nihaal.appana@research.iiit.ac.in |
