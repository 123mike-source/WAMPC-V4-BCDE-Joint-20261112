#!/usr/bin/env bash
set -euo pipefail
cd /share/home/u21012/nixiran/WAMPC
source .runtime/activate_tongji_v3.sh
export WMMPC_ROOT="$PWD"
export TMPDIR="$(mktemp -d /tmp/wampc-v4-joint-train.XXXXXXXX)"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
runtime=.runtime/v4_joint_train_20261112
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv4.joint_train \
  --config method/full_diffusionv4/config_joint.json \
  --output results/full_diffusionv4_joint_20261112/ddp_smoke \
  --resume --max-updates 9 > "$runtime/final_check.log" 2>&1
python "$runtime/audit_start.py"
date -Iseconds > "$runtime/formal_started.txt"
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv4.joint_train \
  --config method/full_diffusionv4/config_joint.json \
  --output ckps/full_diffusionv4/tongji_bcde_joint_seed901_20261112 \
  > "$runtime/train.log" 2>&1
date -Iseconds > "$runtime/formal_finished.txt"
# This child step exits on completion; the user's allocation remains allocated.
