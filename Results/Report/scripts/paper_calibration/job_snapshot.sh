#!/bin/bash
# Report 24 snapshot fields for one finished descent.
# usage: job_snapshot.sh <observable> <restore|norestore> <start index 0-3>
set -xe
PYTHON=$HOME/.conda/envs/veros/bin/python   # `conda activate` is not available on the nodes
export JAX_PLATFORMS=cpu
OBSERVABLE=$1; RESTORING=$2; START=$3
FLAG=--restore-to-climatology
[ "$RESTORING" = norestore ] && FLAG=--no-restore-to-climatology
C_K=(0.05 0.15 0.05 0.15)
C_EPS=(0.40 1.00 1.00 0.40)
RUN=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-24/${OBSERVABLE}_${RESTORING}/runs/start_${START}

cd ~/code/Vercor-Autodiff/Results/Report/scripts/paper_calibration
$PYTHON snapshot.py --run "$RUN" --observable "$OBSERVABLE" $FLAG \
    --c-k "${C_K[$START]}" --c-eps "${C_EPS[$START]}"
echo "=== REPORT24 SNAPSHOT $OBSERVABLE $RESTORING START $START DONE ==="
