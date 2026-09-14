#!/bin/bash
# Report 23: multistart corner 0 (c_k=0.05, c_eps=0.40), 200 iterations, top-3 T/S-difference
# observable. Report-22 pre-registered corners and 200-iteration recipe, observable swapped.
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_OBS=tsdiff_avg
export REPORT21_N_STEPS=10
export REPORT21_N_ITERS=200
export REPORT21_INIT_C_K=0.05
export REPORT21_INIT_C_EPS=0.40
export REPORT21_OUT_DIR=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-23-tsdiff-paper200/start_0
cd ~/code/Vercor-Autodiff
python Results/Report/scripts/report-21/fit.py
echo "=== REPORT23 START200 0 DONE ==="
