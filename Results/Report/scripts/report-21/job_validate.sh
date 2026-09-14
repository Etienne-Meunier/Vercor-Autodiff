#!/bin/bash
# Report 21 pre-check: chained rollout vs monolithic rollout at the start point.
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_N_STEPS=10
cd ~/code/Vercor-Autodiff
python Results/Report/scripts/report-21/validate_segmented.py
echo "=== REPORT21 VALIDATE DONE ==="
