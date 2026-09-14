#!/bin/bash
# Report 21 paper figure: multistart corner 1, c_k=0.15, c_eps=1.00.
# Same recipe as the report-21 mld_avg run in every respect except the start.
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_OBS=mld_avg
export REPORT21_N_STEPS=10
export REPORT21_N_ITERS=100
export REPORT21_INIT_C_K=0.15
export REPORT21_INIT_C_EPS=1.00
export REPORT21_OUT_DIR=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-21-paper/start_1
cd ~/code/Vercor-Autodiff
python Results/Report/scripts/report-21/fit.py
echo "=== REPORT21 START 1 DONE ==="
