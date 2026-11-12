# Source this file from bash to use the isolated Tongji V3 environment.
export WMMPC_PROJECT_ROOT=/share/home/u21012/nixiran/WAMPC
source "$WMMPC_PROJECT_ROOT/.venvs/full-diffusion-carla/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export WMMPC_DIFFUSION_INFERENCE_STEPS=25
export CARLA_ROOT=/share/home/u21012/gjc/carla
export WMMPC_CARLA_RESULTS_ROOT="$WMMPC_PROJECT_ROOT/results"
export WMMPC_DOWNLOAD_ROOT="$WMMPC_PROJECT_ROOT/.runtime/downloads"
export TMPDIR="$WMMPC_PROJECT_ROOT/.runtime/tmp"
export XDG_CACHE_HOME="$WMMPC_PROJECT_ROOT/.runtime/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export WMMPC_BRIDGE_MODULES=/share/home/u21012/nixiran/BridgeDrive/BridgeDrive_adaptation_DiffusionDrive/navsim/agents/diffusiondrive/modules
export PYTHONPATH="$WMMPC_PROJECT_ROOT:$WMMPC_PROJECT_ROOT/leaderboard:$WMMPC_PROJECT_ROOT/scenario_runner:$WMMPC_PROJECT_ROOT/third_party/carla_garage/team_code:$CARLA_ROOT/PythonAPI/carla"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$WMMPC_CARLA_RESULTS_ROOT" "$WMMPC_DOWNLOAD_ROOT" "$TMPDIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME"
