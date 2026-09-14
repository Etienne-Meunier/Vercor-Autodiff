#!/bin/bash
# Submit the snapshot job for the T/S, restoring-off configuration (start 0), if not queued.
JOB=$HOME/code/Vercor-Autodiff/Results/Report/scripts/paper_calibration/job_snapshot.sh
cd ~/code/Vercor-Autodiff
NAME=s24tn0
oarstat -u | grep -qw "$NAME" || oarsub -q production -l host=1,walltime=1:00:00 -n "$NAME" \
    -O "OAR.$NAME.%jobid%.stdout" -E "OAR.$NAME.%jobid%.stderr" \
    "$JOB tsdiff_avg norestore 0"
oarstat -u
