#!/bin/bash
# Train the four models one after another (sharing the GPU slows everything down), then pick the best per case.
cd "$(dirname "$0")"
for m in components_mlp hybrid_analytic_mlp resnet_mlp gpr; do
  python -u $m/train.py > logs/$m.log 2>&1 || echo "$m FAILED" >> logs/failed.txt
done
python -u select_best.py > logs/select_best.log 2>&1
