#!/bin/bash
# Report 24 descent: one pre-registered start of one configuration, 200 iterations.
# usage: job_fit.sh <observable> <restore|norestore> <start index 0-3>
set -xe
PYTHON=$HOME/.conda/envs/veros/bin/python   # `conda activate` is not available on the nodes
export JAX_PLATFORMS=cpu
OBSERVABLE=$1; RESTORING=$2; START=$3
FLAG=--restore-to-climatology
[ "$RESTORING" = norestore ] && FLAG=--no-restore-to-climatology

# corners of a box around the truth: c_k = 0.1 +/- 50%, c_eps = 0.7 +/- 0.3
C_K=(0.05 0.15 0.05 0.15)
C_EPS=(0.40 1.00 1.00 0.40)
OUT=$HOME/code/Vercor-Autodiff/Results/Report/figures/report-24/${OBSERVABLE}_${RESTORING}/runs/start_${START}

cd ~/code/Vercor-Autodiff/Results/Report/scripts/paper_calibration
$PYTHON calibrate.py --observable "$OBSERVABLE" $FLAG \
    --c-k "${C_K[$START]}" --c-eps "${C_EPS[$START]}" --iterations 200 --out "$OUT"
echo "=== REPORT24 FIT $OBSERVABLE $RESTORING START $START DONE ==="
