# Experiments in the final report

Final report: `DVD_Angels_Report.pdf` (IEEE conference, two columns, 11 pages: about 9 pages of main text and 2 pages of
appendix). Build with `report/build.sh` (header of that file). Every table is generated from the saved result files
(`report/make_tables.py`), every number in the text carries a `% src:` comment, and `docs/REPORT_NOTES.md` lists the
numbers with their sources.

Four optional parts have a switch in `main.tex` (`\expFirstModels`, `\expSymmetry`, `\expDist`, `\expFamilies`);
everything else is fixed text.

**FINAL** = part of how the submitted results were produced. **EXPLORATORY** = analysis, earlier work or a negative result.

| ID | Experiment | Where in the report | Status | Form |
|---|---|---|---|---|
| EXP-01 | Dataset generation inside ngspice, ranges, seeds | III-A, Table I | FINAL | main text |
| EXP-02 | The `altermod` bug and its fix | III-B | FINAL | one sentence, no numbers |
| EXP-03 | PMOS OFF ig/ib readout (BSIM4 instance currents) | III-C, IX (assumption 4) | FINAL | main text, no table |
| EXP-04 | Six significant digits in the CSV files | III-D | FINAL | two sentences |
| EXP-05 | Data check: Kirchhoff residuals and the three trend plots | IV | FINAL | main text, Table II, Figs. 2-3 |
| EXP-06 | Current-magnitude distributions | Appendix A | EXPLORATORY | appendix table (`\expDist`) |
| EXP-07 | First model set (70:15:15), 29-model benchmark, seed check | V-C | EXPLORATORY | one paragraph, no table (`\expFirstModels`) |
| EXP-08 | Components MLP excluded, PINN added | V-C, IX (assumption 3) | EXPLORATORY | one paragraph |
| EXP-09 | Plain current sum fails; super-node method; check with exact currents | VI-A, VI-C | FINAL | main text, TikZ figure, table |
| EXP-10 | Diagnosis of the remaining gate error (rail-to-rail OFF devices) | VIII-B | EXPLORATORY | main text, no table |
| EXP-11 | Gate accuracy per model family | Appendix B | EXPLORATORY | appendix table, labelled "for comparison only, not used for selection" (`\expFamilies`) |
| EXP-12 | Part C ceiling analysis (loading, off-rail inputs, temperature) | VII-C | EXPLORATORY | main text |
| EXP-13 | Training on 100 % of the data | - | removed (never run to completion, no results) | - |
| EXP-14 | 80:10:10 pipeline, four compliant types, near-rail metric and weighting (weight 0 chosen) | V-D | FINAL | main text, Tables III-IV |
| EXP-15 | Drain-source symmetry (negative result, cause verified) | V-F | EXPLORATORY | one paragraph (`\expSymmetry`) |
| EXP-16 | ResNet OFF tuning (7 variants, rule fixed beforehand, 5 blocks) | V-E | FINAL | compact table |
| EXP-17 | Four accuracy definitions | IX (assumption 1) | FINAL | small table |
| EXP-18 | Corners and ngspice build dependence | III-D | FINAL | two sentences |
| EXP-19 | Leakage definition, size of input-pin currents | IX (assumption 2) | EXPLORATORY | one paragraph |
| EXP-20 | Gate transistor sizing | III-D | FINAL | one sentence |
| EXP-21 | Seed averaging | X (future work) | not done | one clause |

Removed from the draft: the AI-use placeholder section, the per-vector CLA table (data in `results/partC/cla3_results.csv`),
the specification-compliance table, and the tables for the readout precision, digits, benchmark, seed check, symmetry,
rail-point counts, gate-point replay, pin-current shares and the first-set comparison (their numbers are in the text
where needed and all files remain in `results/` and `archive/`).
