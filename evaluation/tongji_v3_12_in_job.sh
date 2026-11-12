#!/usr/bin/env bash
set -euo pipefail
retain_terminal() {
  status=$?
  trap - EXIT
  export WMMPC_LAST_EVAL_EXIT_CODE=$status
  printf '\nWAMPC 12-step exited with status %s; allocation %s is retained.\n' "$status" "${SLURM_JOB_ID:-unknown}"
  exec bash --noprofile --norc -i
}
trap retain_terminal EXIT
project_root=/share/home/u21012/nixiran/WAMPC
run_root=${1:?A prepared 12-step run root is required}
cd "$project_root"
source .runtime/activate_tongji_v3.sh
export WMMPC_DIFFUSION_INFERENCE_STEPS=12
export WMMPC_BUFFER_METRICS=1
export WMMPC_CARLA_RESULTS_ROOT="$run_root"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
python -u evaluation/run_full_diffusionv3_12_slurm.py --role node \
  --run-root "$run_root" --inference-steps 12 --lanes 2 \
  --carla-root "$project_root/.runtime/carla_tongji_0.9.15_sensor_guard" \
  --checkpoint "$project_root/ckps/full_diffusionv3/release/epoch_10.pt"
