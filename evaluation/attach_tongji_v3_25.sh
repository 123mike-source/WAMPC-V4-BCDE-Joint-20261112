#!/usr/bin/env bash
set -euo pipefail
test -n "${TMUX:-}" || { echo 'Start this launcher inside tmux'; exit 2; }
project_root=/share/home/u21012/nixiran/WAMPC
run_root=${1:?Usage: attach_tongji_v3_25.sh RUN_ROOT EXISTING_JOB_ID}
job_id=${2:?An already running eight-GPU allocation is required}
[[ $job_id =~ ^[0-9]+$ ]] || exit 2
cd "$project_root"
test -f "$run_root/prepared.json"
# --jobid joins a step to the user's allocation; it neither creates nor cancels it.
exec srun --jobid="$job_id" --overlap -N 1 -n 1 --cpus-per-task=56 \
  --gres=gpu:l40:8 --cpu-bind=none --pty \
  bash "$project_root/evaluation/tongji_v3_in_job.sh" "$run_root"
