#!/usr/bin/env bash
# Deliverable 9: run every step that regenerates the deliverables, in order (Linux / macOS).
#   bash run_pipeline.sh --fast          # saved models and saved SPICE data (minutes)
#   bash run_pipeline.sh --skip-spice    # re-train, keep saved SPICE data
#   bash run_pipeline.sh --skip-train    # re-run SPICE, keep saved models
#   bash run_pipeline.sh                 # everything (SPICE + training, ~10+ hours)
# --fast = --skip-spice --skip-train. All outputs go to $OUT (default pipeline_out/); the submitted files in
# results and the data folders are never overwritten. See REPRODUCE.md for the step list and seeds.
# Options: --yes (no confirmation), --out DIR, --ngspice PATH, --python PATH
set -euo pipefail
cd "$(dirname "$0")"
SKIP_SPICE=0; SKIP_TRAIN=0; YES=0; OUT="pipeline_out"; NGSPICE="${NGSPICE:-}"; PY="${PYTHON:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --fast) SKIP_SPICE=1; SKIP_TRAIN=1 ;;
    --skip-spice) SKIP_SPICE=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --yes) YES=1 ;;
    --out) OUT="$2"; shift ;;
    --ngspice) NGSPICE="$2"; shift ;;
    --python) PY="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
if [ -z "$PY" ]; then if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi; fi
if [ "$SKIP_SPICE" = 0 ] || [ "$SKIP_TRAIN" = 0 ]; then
  if [ "$YES" = 0 ]; then
    read -r -p "Slow steps selected (SPICE regeneration and/or training, many hours). Type YES to continue: " a
    [ "$a" = "YES" ] || { echo aborted; exit 1; }
  fi
fi
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
FINAL_MODELS="$(pwd)/results/models"
step() { echo; echo "=== STEP $1 : $2 ($(date +%H:%M))"; }
find_ngspice() {
  if [ -n "$NGSPICE" ]; then echo "$NGSPICE"; return; fi
  for n in ngspice ngspice_con; do if command -v "$n" >/dev/null 2>&1; then echo "$n"; return; fi; done
  echo "ngspice not found: pass --ngspice PATH" >&2; exit 1
}

step 1 "Part A datasets: 4 single-transistor netlists, VDD = 1 V, rndseed 13 / 12 / 14 / 15"
if [ "$SKIP_SPICE" = 1 ]; then echo "skipped: using the saved partA_transistor/data/*.csv (10,000 rows each)"; else
  NG="$(find_ngspice)"; mkdir -p "$OUT/spice/model"; cp -f model/22nm_HP.pm "$OUT/spice/model"/
  D="$OUT/spice/partA_transistor"; mkdir -p "$D/netlists" "$D/data"; cp -f partA_transistor/netlists/*.cir "$D/netlists"/
  ( cd "$D/netlists" && for c in nmos_on nmos_off pmos_on pmos_off; do "$NG" -b "${c}_ngspice.cir" > /dev/null; done )
  echo "regenerated CSVs in $D/data (compare with partA_transistor/data; the random stream depends on the ngspice build)"
fi

step 2 "Part A data check: Kirchhoff residuals and trend plots"
"$PY" -u validation_checks/data_check.py --out "$OUT/partA_data_check"

MODELS="$FINAL_MODELS"
if [ "$SKIP_TRAIN" = 1 ]; then
  step 3 "Training and model comparison (80:10:10, split seed 42, 4 model types x 4 cases)"
  echo "skipped: using the saved models in $FINAL_MODELS and the saved tables in results/partA"
  step 4 "Resnet OFF tuning and selection on validation"
  echo "skipped (saved result: results/partA/resnet_tuning.csv, chosen setting blocks=5)"
else
  step 3 "Training and model comparison (80:10:10, split seed 42, model seed 0, 4 model types x 4 cases)"
  ( cd ml
    for A in 0 10 30; do "$PY" -u train.py --models resnet_mlp --cases nmos_off pmos_off --rail-weight "$A" --tag "tune_A$A" --force; done
    "$PY" -u choose_weight.py
    W="$(tr -d '[:space:]' < runs/chosen_rail_weight.txt)"
    for M in resnet_mlp hybrid_analytic_mlp pinn gpr; do "$PY" -u train.py --models "$M" --rail-weight "$W" --tag final --force; done
    step 4 "Resnet OFF tuning (7 single-setting variants) and selection on validation, then the final model set"
    for kv in ep1000:epochs=1000 lr5e-4:lr=5e-4 wd0:weight_decay=0 bs64:batch_size=64 hub002:huber_beta=0.02 w512:width=512 b5:blocks=5; do
      "$PY" -u train.py --models resnet_mlp --cases nmos_off pmos_off --model-params "${kv#*:}" --tag "rn_${kv%%:*}" --force
    done
    "$PY" -u choose_resnet.py
    "$PY" -u assemble_tuned.py
    ML_V2_RESULTS="$OUT/results_tuned" "$PY" -u select_best.py --tag final_tuned )
  MODELS="$OUT/models"; mkdir -p "$MODELS"
  for c in nmos_on nmos_off pmos_on pmos_off; do cp -f "$OUT/results_tuned/$c/model.joblib" "$MODELS/$c.joblib"; done
fi

step 5 "Part B: gate SPICE (5 netlists, rndseed = 16) and ML gate evaluation"
if [ "$SKIP_SPICE" = 1 ]; then echo "skipped: using the saved partB_gates/data/*.csv (3,600 rows)"; else
  NG="$(find_ngspice)"; mkdir -p "$OUT/spice/model"; cp -f model/22nm_HP.pm "$OUT/spice/model"/; D="$OUT/spice/partB_gates"; mkdir -p "$D/netlists" "$D/data"; cp -f partB_gates/netlists/* "$D/netlists"/
  ( cd "$D/netlists" && for g in inv nand2 nor2 and2 or2; do "$NG" -b "${g}_ngspice.cir" > /dev/null; done )
  export GATE_DATA_DIR="$D/data"; echo "evaluating the regenerated gate data in $D/data"
fi
"$PY" -u ml/evaluate_gates.py --models-dir "$MODELS" --out "$OUT/gate_results"
unset GATE_DATA_DIR || true

step 6 "Part C: CLA SPICE (rndseed = 16, 64 vectors x 200 corners) and ML evaluation (plain gate sum)"
if [ "$SKIP_SPICE" = 1 ]; then echo "skipped: using the saved partC_cla/data/cla3.csv (12,800 rows)"; else
  NG="$(find_ngspice)"; mkdir -p "$OUT/spice/model"; cp -f model/22nm_HP.pm "$OUT/spice/model"/; D="$OUT/spice/partC_cla"; mkdir -p "$D/netlists" "$D/data"; cp -f "partC_cla/netlists/"* "$D/netlists"/
  ( cd "$D/netlists" && "$NG" -b cla3_ngspice.cir > /dev/null )
  export CLA_DIR="$D"; echo "evaluating the regenerated CLA data in $D/data"
fi
"$PY" -u ml/evaluate_cla.py --predictions "$OUT/gate_results/gate_predictions.csv" --out "$OUT/cla_results"
unset CLA_DIR || true

step 7 "Compare with the submitted results"
if [ "$SKIP_SPICE" = 1 ] && [ "$SKIP_TRAIN" = 1 ]; then "$PY" -u ml/reproduce_check.py --out "$OUT"
else echo "regenerated data or models: numbers can differ from the submitted ones (ngspice build, hardware); see $OUT"; fi
echo; echo "done ($(date +%H:%M)), outputs in $OUT"
