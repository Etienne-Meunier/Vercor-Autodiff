#!/bin/bash
# Report 21 paper figure: loss-landscape grid, c_k columns [12, 16).
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_OBS=mld_avg
export REPORT21_N_STEPS=10
export REPORT21_GRID_N=16
export REPORT21_GRID_I0=12
export REPORT21_GRID_I1=16
export REPORT21_PAPER_DIR=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-21-paper
cd ~/code/Vercor-Autodiff
python Results/Report/scripts/report-21/landscape_grid.py
echo "=== REPORT21 GRID 12-16 DONE ==="
