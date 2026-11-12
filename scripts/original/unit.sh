#!/usr/bin/env bash
set -euo pipefail
cd /share/home/u21012/nixiran/WAMPC
source .runtime/activate_tongji_v3.sh
export WMMPC_ROOT="$PWD"
export TMPDIR="$(mktemp -d /tmp/wampc-v4-joint-unit.XXXXXXXX)"
export PYTHONUNBUFFERED=1
python -m unittest discover -s tests -p test_full_diffusionv4_joint.py -v
