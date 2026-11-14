#!/usr/bin/env bash
# Full ML_Models_v2 pipeline, in order. Run from inside ML_Models_v2:  bash run_all.sh
# Logs: logs/<step>.log. Models run one after the other (two GPU jobs at once slow each other down).
set -e
cd "$(dirname "$0")"
mkdir -p logs

# 1. near-rail training weight: tuning runs (resnet_mlp, OFF cases), chosen on validation only
for A in 0 10 30; do
  python -u train.py --models resnet_mlp --cases nmos_off pmos_off --rail-weight $A --tag tune_A$A \
    > logs/tune_A$A.log 2>&1
done
python -u choose_weight.py > logs/choose_weight.log 2>&1
A=$(cat runs/chosen_rail_weight.txt)

# 2. all four model types on all four cases (80:10:10 split) with the chosen weight
for M in resnet_mlp hybrid_analytic_mlp pinn gpr; do
  python -u train.py --models $M --rail-weight $A --tag final > logs/final_$M.log 2>&1
done

# 3. best model per case on validation -> results/ ; 4. Part B ; 5. Part C
python -u select_best.py --tag final > logs/select_best.log 2>&1
python -u evaluate_gates.py > logs/evaluate_gates.log 2>&1
python -u evaluate_cla.py > logs/evaluate_cla.log 2>&1
echo "done $(date)"
