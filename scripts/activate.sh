#!/usr/bin/env bash
# Source this file after activating the Python 3.10 environment.
_wampc_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WMMPC_ROOT="$_wampc_repo"
export WMMPC_BRIDGE_MODULES="$_wampc_repo/vendor/bridgedrive/modules"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export WMMPC_CARLA_NAVIGATION_PATH="$_wampc_repo/third_party/carla_navigation_0_9_15"
export CARLA_ROOT="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["WMMPC_ROOT"],"configs/paths.json")))["carla_root"])')"
export PYTHONPATH="$_wampc_repo:$_wampc_repo/third_party/carla_navigation_0_9_15:$_wampc_repo/third_party/carla_garage/team_code:$_wampc_repo/third_party/carla_garage/leaderboard:$_wampc_repo/third_party/carla_garage/scenario_runner:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-$_wampc_repo/.runtime/torch-cache}"
export HF_HOME="${HF_HOME:-$_wampc_repo/.runtime/huggingface}"
unset _wampc_repo
