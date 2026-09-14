#!/bin/bash
# Report 21 paper figure: multistart corner 3, c_k=0.15, c_eps=0.40.
# Same recipe as the report-21 mld_avg run in every respect except the start.
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_OBS=mld_avg
export REPORT21_N_STEPS=10
export REPORT21_N_ITERS=100
export REPORT21_INIT_C_K=0.15
export REPORT21_INIT_C_EPS=0.40
export REPORT21_OUT_DIR=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-21-paper/start_3
cd ~/code/Vercor-Autodiff
python Results/Report/scripts/report-21/fit.py
echo "=== REPORT21 START 3 DONE ==="
