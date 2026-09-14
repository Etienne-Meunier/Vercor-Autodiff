#!/bin/bash
# Report 21 paper figure 2: MLD error before/after, at the 200-iteration start-0 fit
# (c_k = 0.099997, c_eps = 0.699985 -- exact recovery), replacing the 100-iteration
# snapshot whose fitted point was 1.9%/2.9% off.
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_OBS=mld_avg
export REPORT21_N_STEPS=10
export REPORT21_N_ITERS=200
export REPORT21_OUT_DIR=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-21-paper200/start_0
cd ~/code/Vercor-Autodiff
python Results/Report/scripts/report-21/snapshots.py
echo "=== REPORT21 SNAPSHOTS200 DONE ==="
