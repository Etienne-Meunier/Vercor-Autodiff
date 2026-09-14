#!/bin/bash
# Report 24 landscape grid: one c_k column chunk of one configuration.
# usage: job_grid.sh <observable> <restore|norestore> <col-start> <col-end>
set -xe
PYTHON=$HOME/.conda/envs/veros/bin/python   # `conda activate` is not available on the nodes
export JAX_PLATFORMS=cpu
OBSERVABLE=$1; RESTORING=$2; COL_START=$3; COL_END=$4
FLAG=--restore-to-climatology
[ "$RESTORING" = norestore ] && FLAG=--no-restore-to-climatology
OUT=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-24/${OBSERVABLE}_${RESTORING}/grid

cd ~/code/Vercor-Autodiff/Results/Report/scripts/paper_calibration
$PYTHON landscape.py scan --observable "$OBSERVABLE" $FLAG \
    --col-start "$COL_START" --col-end "$COL_END" --out "$OUT"
echo "=== REPORT24 GRID $OBSERVABLE $RESTORING $COL_START-$COL_END DONE ==="
