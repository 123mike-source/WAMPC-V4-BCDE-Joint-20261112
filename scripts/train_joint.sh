#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate.sh
: "${SLURM_JOB_ID:?Run this script inside the existing eight-GPU allocation}"
config="${1:-configs/local_joint.json}"
output="${2:-ckps/full_diffusionv4/local_joint}"
shift "$(( $# > 1 ? 2 : $# ))"
export TMPDIR="$(mktemp -d /tmp/wampc-v4-joint.XXXXXXXX)"
export NCCL_DEBUG=WARN
mkdir -p "$output"
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv4.joint_train \
  --config "$config" --output "$output" "$@" 2>&1 | tee -a "$output/train.log"
# Returning from this child step leaves the user's interactive allocation intact.
