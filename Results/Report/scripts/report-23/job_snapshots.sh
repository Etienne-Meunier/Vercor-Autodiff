#!/bin/bash
# Report 23 snapshot figure: top-level density difference and per-column loss at
# truth, at start 3's initial guess and at its fitted point (start 3 is the run
# that recovered the parameters with the tsdiff_avg loss).
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_OBS=tsdiff_avg
export REPORT21_N_STEPS=10
export REPORT21_N_ITERS=200
export REPORT21_INIT_C_K=0.15
export REPORT21_INIT_C_EPS=0.40
export REPORT21_OUT_DIR=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-23-tsdiff-paper200/start_3
cd ~/code/Vercor-Autodiff
python Results/Report/scripts/report-23/snapshots_tsdiff.py
echo "=== REPORT23 SNAPSHOTS DONE ==="
