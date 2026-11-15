#!/bin/bash
# Train all four model types on the high-resolution PMOS OFF data. Outputs stay inside Datasets_hires/ml_results.
cd "$(dirname "$0")/../ML models"
for m in mlp xgboost_regressor random_forest ridge_polynomial; do
  python $m/train.py --data-dir ../Datasets_hires/data --cases pmos_off \
     --results-dir ../Datasets_hires/ml_results/$m/results --models-dir ../Datasets_hires/ml_results/$m/saved_models
done
