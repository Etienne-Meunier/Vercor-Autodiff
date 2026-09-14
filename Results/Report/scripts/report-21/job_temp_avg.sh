#!/bin/bash
# Report 21: report-20's N=10 / 100-iteration fit with the temp_avg observable.
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_OBS=temp_avg
export REPORT21_N_STEPS=10
export REPORT21_N_ITERS=100
export REPORT21_OUT_DIR=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-21-temp_avg
cd ~/code/Vercor-Autodiff
python Results/Report/scripts/report-21/fit.py
echo "=== REPORT21 temp_avg DONE ==="
