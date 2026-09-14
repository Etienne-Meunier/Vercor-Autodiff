#!/bin/bash
# Report 23: loss-landscape grid for the top-3 T/S-difference observable, c_k columns [12, 16).
# Same 16x16 scan as report-22 (report-21 scripts), observable swapped from mld_avg.
set -x
source ~/.bashrc
conda activate veros
export JAX_PLATFORMS=cpu
export REPORT21_OBS=tsdiff_avg
export REPORT21_N_STEPS=10
export REPORT21_GRID_N=16
export REPORT21_GRID_I0=12
export REPORT21_GRID_I1=16
export REPORT21_PAPER_DIR=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-23-tsdiff-paper
cd ~/code/Vercor-Autodiff
python Results/Report/scripts/report-21/landscape_grid.py
echo "=== REPORT23 GRID 12-16 DONE ==="
