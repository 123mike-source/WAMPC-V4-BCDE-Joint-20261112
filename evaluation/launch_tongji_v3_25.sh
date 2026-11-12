#!/usr/bin/env bash
set -euo pipefail
test -n "${TMUX:-}" || { echo 'Start this launcher inside tmux'; exit 2; }
project_root=/share/home/u21012/nixiran/WAMPC
run_root=${1:?Usage: launch_tongji_v3_25.sh RUN_ROOT JOB_NAME}
job_name=${2:?A unique job name is required}
cd "$project_root"
test -f "$run_root/prepared.json"
exec srun -p L40 --job-name="$job_name" -N 1 --gres=gpu:l40:8 \
  --ntasks-per-node=1 --cpus-per-task=56 --mem=0 --time=12:00:00 \
  --account=p_p21012 --qos=cpu-4000_core-l40-32_card-a800-16_card \
  --pty bash "$project_root/evaluation/tongji_v3_in_job.sh" "$run_root"
