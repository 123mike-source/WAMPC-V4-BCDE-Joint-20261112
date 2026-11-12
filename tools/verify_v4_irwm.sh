#!/usr/bin/env bash
set -euo pipefail
V4_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V4_PROJECT_ROOT"
source .runtime/activate_tongji_v3.sh
export WMMPC_ROOT="$V4_PROJECT_ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m unittest discover -s tests -p test_full_diffusionv4.py -v
