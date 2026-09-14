#!/bin/bash
# Report 21 paper figure: multistart corner 1 (c_k=0.15, c_eps=1.00) at 200 iterations.
#
# The 100-iteration set left start 3 with its gradient still pointing at truth and
# the cosine schedule already at zero, while starts 1 and 2 sat in the shallow
# secondary minimum near c_eps = 0.94 that the loss grid resolves. Doubling the
# budget separates those two explanations. decay_steps follows N_ITERS, so this is
# a different schedule from the 100-iteration runs, not a continuation of them --
# all four corners are rerun so the set stays internally comparable.
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_OBS=mld_avg
export REPORT21_N_STEPS=10
export REPORT21_N_ITERS=200
export REPORT21_INIT_C_K=0.15
export REPORT21_INIT_C_EPS=1.00
export REPORT21_OUT_DIR=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-21-paper200/start_1
cd ~/code/Vercor-Autodiff
python Results/Report/scripts/report-21/fit.py
echo "=== REPORT21 START200 1 DONE ==="
