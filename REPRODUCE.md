# Reproducing the deliverables

`run_pipeline.ps1` (Windows) and `run_pipeline.sh` (Linux / macOS) run every step that regenerates the deliverables, in
order, from the model file `22nm_HP.pm` (PTM 22 nm HP, VDD = 1 V, L = 22 nm, ngspice-44.2).

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Fast         # saved models + saved SPICE data, about 30 s
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -SkipSpice    # re-train (hours), keep the saved SPICE data
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -SkipTrain    # re-run SPICE, keep the saved models
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1               # everything (asks for confirmation; -Yes skips it)
```
```bash
# Linux / macOS
bash run_pipeline.sh --fast            # also --skip-spice, --skip-train, --yes, --out DIR, --ngspice PATH, --python PATH
```

* **All outputs go to `pipeline_out/`** (`-Out` / `--out`). The submitted files in `results/` and the data
  folders are never overwritten; regenerated SPICE data are written to `pipeline_out/spice/`.
* **Fast path** (`-Fast`): steps 2, 5 and 6 run on the saved models and saved SPICE data, then step 7 compares with the
  submitted results. Dry run on 2026-11-15: Part B 98.87 % and Part C 99.00 % mean accuracy, largest relative difference
  of the ML leakage to the submitted predictions 0.00e+00, `REPRODUCTION: PASS` (`validation_checks/pipeline_fast_run.txt`).
* **Slow steps were not run** in the check above. Their commands are those that produced the submitted results (same
  scripts and arguments, see `ml/run_parallel.ps1`, `tune_resnet.ps1`, `finish_tuned.ps1`), but the
  `run_pipeline` wrappers around them have only been syntax-checked (`bash -n`, PowerShell parser), not executed end to end.
  Every training call uses `--force` so that it really retrains even if `runs/<tag>` exists.

## Steps and seeds

| # | Step | Script / command | Seeds and settings | Output |
|---|---|---|---|---|
| 1 | Part A datasets, 4 single-transistor netlists, 10,000 rows each | `partA_transistor/netlists/{nmos_on,nmos_off,pmos_on,pmos_off}_ngspice.cir`, run with `ngspice_con -b <file>` from inside the folder | `rndseed` 13 (NMOS ON), 12 (NMOS OFF), 14 (PMOS ON), 15 (PMOS OFF); VDD nominal 1 V (sampled +/-10 %), L = 22 nm, W 22 to 176 nm; one parameter per `altermod` line | `partA_transistor/data/*.csv` (slow: SPICE) |
| 2 | Part A data check (deliverable 3) | `validation_checks/data_check.py` | none (deterministic) | Kirchhoff table, three trend plots |
| 3 | Training and model comparison | `ml/train.py` (4 model types x 4 cases), `choose_weight.py` | split 80:10:10, split seed 42, model seed 0; rail weight tried 0 / 10 / 30 and chosen on validation (0) | `runs/final/...` (slow: hours, GPU) |
| 4 | Resnet OFF tuning, selection on validation, final model set | `train.py --model-params ...` for the 7 variants, `choose_resnet.py`, `assemble_tuned.py`, `select_best.py --tag final_tuned` | same split and seeds; rule fixed before the runs; chosen setting 5 blocks | final models (4 files), model comparison table |
| 5 | Part B: gate SPICE and ML evaluation | `partB_gates/netlists/{inv,nand2,nor2,and2,or2}_ngspice.cir`; `ml/evaluate_gates.py --models-dir DIR --out DIR` | `rndseed` 16 (200 corners, shared with Part C) | `gate_predictions.csv`, `gate_results.csv`, `gate_parity.png` |
| 6 | Part C: CLA SPICE and ML evaluation | `partC_cla/netlists/cla3_ngspice.cir` (+ `cla3.sub`, `gates.sub`, `22nm_HP.pm`); `ml/evaluate_cla.py --predictions ... --out ...` | `rndseed` 16 | `cla3_predictions.csv`, `cla3_results.csv`, `cla3_parity.png` |
| 7 | Compare with the submitted results | `ml/reproduce_check.py --out pipeline_out` | - | PASS / FAIL |

## Notes

* **Corners depend on the ngspice build.** `rndseed = 16` gives the 200 corners of the submitted data on Windows
  ngspice-44.2; a different build (for example the ngspice-44+ in WSL) draws a different random stream, so regenerated
  SPICE data and therefore the regenerated numbers differ slightly. The corners that were used are saved in the CSVs
  (`partC_cla/data/corners.csv` and the corner columns of every Part B / Part C CSV), and the fast path and
  the ML evaluation always work from those saved files.
* **Netlist paths.** Every netlist is run from its own `netlists/` folder: it `.include`s the model card as
  `../../model/22nm_HP.pm` and writes its CSV to `../data/`. (`run_pipeline` copies the folder structure into
  `pipeline_out/spice/` so the saved data are not overwritten.)
* **Part C regeneration into a new file.** `python partC_cla/scripts/regen_cla3.py --compare` re-runs the unchanged
  CLA netlist with the recorded corners and writes `pipeline_out/partC/cla3_regen.csv`; it is independent of the ngspice
  build (see the note on corners above).
* **Re-simulation check of all netlists.** `python validation_checks/resim_netlists.py` re-simulates 5 rows (Part A) or
  3 corners (Parts B and C, every input combination) of each moved netlist with the recorded random draws and compares
  with the CSVs (`validation_checks/netlist_resim_check.txt`: largest relative difference 0).
* **Models** are pickled objects of the classes in `ml/` (`models.py`, `pipeline.py`): load them with
  `ml` on `sys.path`.

## Part C generation scripts

The Part C README used to list `gen_cla3_gate_inputs.py`, `run_partC.sh`, `data/cla3_gate_inputs.csv`,
`data/cla3_gate_counts.csv` and `logs/`. A search of the working tree, all branches, tags, stashes and every commit
(file names and contents) found none of them: they were never committed. They are not needed by the pipeline
(`ml/evaluate_cla.py` computes the gate list, the logic at every gate input and the ceiling sum itself), and the
README no longer lists them. The step that generates the Part C data is the netlist itself (step 6) and
`partC_cla/scripts/regen_cla3.py` regenerates it into a new file.
