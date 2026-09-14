#!/bin/bash
# Submit the 16 descents of the 2x2 campaign, skipping any already queued.
JOB=$HOME/code/Vercor-Autodiff/Results/Report/scripts/paper_calibration/job_fit.sh
cd ~/code/Vercor-Autodiff
while read -r OBSERVABLE RESTORING TAG; do
    for START in 0 1 2 3; do
        NAME="f24${TAG}${START}"
        oarstat -u | grep -qw "$NAME" && continue
        oarsub -q production -l host=1,walltime=12:00:00 -n "$NAME" \
            "$JOB $OBSERVABLE $RESTORING $START" | grep OAR_JOB_ID
    done
done <<ROWS
mld_avg restore mr
mld_avg norestore mn
tsdiff_avg restore tr
tsdiff_avg norestore tn
ROWS
oarstat -u | tail -20
