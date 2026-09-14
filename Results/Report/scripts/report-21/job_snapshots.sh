#!/bin/bash
# Report 21: before/after error maps for all three variants.
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_N_STEPS=10
export REPORT21_N_ITERS=100
cd ~/code/Vercor-Autodiff
for obs in mld temp_avg mld_avg; do
  REPORT21_OBS=$obs \
  REPORT21_OUT_DIR=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-21-$obs \
  python Results/Report/scripts/report-21/snapshots.py
done
echo "=== REPORT21 SNAPSHOTS DONE ==="
