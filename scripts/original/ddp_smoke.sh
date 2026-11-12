#!/usr/bin/env bash
set -euo pipefail
cd /share/home/u21012/nixiran/WAMPC
source .runtime/activate_tongji_v3.sh
export WMMPC_ROOT="$PWD"
export TMPDIR="$(mktemp -d /tmp/wampc-v4-joint-ddpcheck.XXXXXXXX)"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv4.joint_train --config method/full_diffusionv4/config_joint.json --output results/full_diffusionv4_joint_20261112/ddp_smoke --max-updates 5
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv4.joint_train --config method/full_diffusionv4/config_joint.json --output results/full_diffusionv4_joint_20261112/ddp_smoke --resume --max-updates 6
