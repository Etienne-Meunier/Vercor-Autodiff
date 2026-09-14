#!/bin/bash
# Submit the 16 grid chunks of the 2x2 campaign, skipping any already queued.
JOB=$HOME/code/Vercor-Autodiff/Results/Report/scripts/paper_calibration/job_grid.sh
cd ~/code/Vercor-Autodiff
while read -r OBSERVABLE RESTORING TAG; do
    for COL_START in 0 4 8 12; do
        NAME="g24${TAG}${COL_START}"
        oarstat -u | grep -qw "$NAME" && continue
        oarsub -q production -l host=1,walltime=4:00:00 -n "$NAME" \
            "$JOB $OBSERVABLE $RESTORING $COL_START $((COL_START + 4))" | grep OAR_JOB_ID
    done
done <<ROWS
mld_avg restore mr
mld_avg norestore mn
tsdiff_avg restore tr
tsdiff_avg norestore tn
ROWS
oarstat -u
