#!/usr/bin/env bash
set -euo pipefail
cd /share/home/u21012/nixiran/WAMPC
source .runtime/activate_tongji_v3.sh
export WMMPC_ROOT="$PWD"
export TMPDIR
TMPDIR="$(mktemp -d /tmp/wampc-v4-joint-prepare.XXXXXXXX)"
export PYTHONUNBUFFERED=1
python -m method.full_diffusionv4.prepare_joint labels --output data/full_diffusionv4/pdm_seed901_bcde_v1 --workers 32
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv4.prepare_joint contexts --output data/full_diffusionv4/pdm_seed901_bcde_v1 --workers 4 --batch-size 16
