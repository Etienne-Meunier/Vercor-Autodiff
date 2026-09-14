#!/bin/bash
# Smoke test of the paper_calibration package: a 3-day rollout, 2 iterations, 2 grid
# nodes, both observables. Checks that every entry point runs end to end; the numbers
# it produces are meaningless.
set -xe
PYTHON=$HOME/.conda/envs/veros/bin/python   # `conda activate` is not available on the nodes
export JAX_PLATFORMS=cpu
cd ~/code/Vercor-Autodiff/Results/Report/scripts/paper_calibration
SMOKE=$HOME/smoke_paper_calibration
rm -rf $SMOKE

$PYTHON calibrate.py --observable tsdiff_avg --days 3 --average-days 3 --iterations 2 \
    --c-k 0.15 --c-eps 0.40 --out $SMOKE/runs/start_3
$PYTHON calibrate.py --observable mld_avg --days 3 --average-days 3 --iterations 1 \
    --c-k 0.05 --c-eps 0.40 --out $SMOKE/runs/mld_start_0
$PYTHON snapshot.py --run $SMOKE/runs/start_3 --days 3 --average-days 3 --c-k 0.15 --c-eps 0.40
$PYTHON landscape.py scan --days 3 --average-days 3 --grid-size 2 --col-start 0 --col-end 2 \
    --out $SMOKE/grid
$PYTHON landscape.py merge --out $SMOKE/grid
$PYTHON figures/optimization.py --grid $SMOKE/grid --run $SMOKE/runs/start_3 \
    --out $SMOKE/figures/optimization.png
# figures/snapshot.py needs cartopy, which is not installed here; drawn locally.
echo "=== PAPER CALIBRATION SMOKE DONE ==="
