#!/usr/bin/env bash
# Run inside the user's existing allocation. This script does not manage it.
set -euo pipefail
cd /share/home/u21012/nixiran/WAMPC
source .runtime/activate_tongji_v3.sh
PYTHON=/share/home/u21012/nixiran/WAMPC/.venvs/full-diffusion-carla/bin/python
CONFIG=${1:-method/full_diffusionv5/config_train4hz.json}
STAGE=${2:-all}
OUTPUT=${3:-ckps/full_diffusionv5/tongji_train4hz_fresh_seed901}
if [[ -z ${SLURM_JOB_ID:-${LSB_JOBID:-}} ]]; then
  echo 'Run inside an existing compute allocation.' >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ $STAGE == labels ]]; then
  exec "$PYTHON" -m method.full_diffusionv5.prepare labels --config "$CONFIG" --workers "${LABEL_WORKERS:-8}"
fi
NPROC=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')
if (( NPROC < 1 )); then
  echo 'No CUDA device is visible in this allocation.' >&2
  exit 2
fi
RUN=("$PYTHON" -m torch.distributed.run --standalone --nproc-per-node "$NPROC")
if [[ $STAGE == all || $STAGE == train ]]; then
  EXPECTED=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["training"]["world_size"])' "$CONFIG")
  if (( NPROC != EXPECTED )); then
    echo "Visible GPUs ($NPROC) differ from training.world_size ($EXPECTED); prepare a matching config." >&2
    exit 2
  fi
fi
case "$STAGE" in
  all)
    INDEX=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["index_directory"])' "$CONFIG")
    if [[ ! -f $INDEX/manifest.json ]]; then
      "$PYTHON" -m method.full_diffusionv5.prepare labels --config "$CONFIG" --workers "${LABEL_WORKERS:-8}"
    fi
    "${RUN[@]}" -m method.full_diffusionv5.prepare features --config "$CONFIG" --workers "${CACHE_WORKERS:-4}" --batch-size "${CACHE_BATCH:-16}"
    "${RUN[@]}" -m method.full_diffusionv5.prepare contexts --config "$CONFIG" --workers "${CACHE_WORKERS:-4}" --batch-size "${CACHE_BATCH:-16}"
    ;;
  features|contexts)
    exec "${RUN[@]}" -m method.full_diffusionv5.prepare "$STAGE" --config "$CONFIG" --workers "${CACHE_WORKERS:-4}" --batch-size "${CACHE_BATCH:-16}"
    ;;
  train) ;;
  *) echo 'Stage must be labels, features, contexts, train or all.' >&2; exit 2 ;;
esac
OPTIONS=()
if [[ ${RESUME:-0} == 1 ]]; then OPTIONS+=(--resume); fi
if [[ -n ${MAX_UPDATES:-} ]]; then OPTIONS+=(--max-updates "$MAX_UPDATES"); fi
exec "${RUN[@]}" -m method.full_diffusionv5.train --config "$CONFIG" --output "$OUTPUT" "${OPTIONS[@]}"
