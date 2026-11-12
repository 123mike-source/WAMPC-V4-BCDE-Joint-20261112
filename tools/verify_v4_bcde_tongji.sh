#!/usr/bin/env bash
set -euo pipefail
V4_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V4_PROJECT_ROOT"
test -n "${SLURM_JOB_ID:-}"
source .runtime/activate_tongji_v3.sh
export WMMPC_ROOT="$V4_PROJECT_ROOT"
# Use node-local temporary files; shared NFS temp directories can retain open files.
export TMPDIR
TMPDIR="$(mktemp -d /tmp/wampc-v4-bcde.XXXXXXXX)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
V4_RESULTS=results/full_diffusionv4_bcde_20261112
mkdir -p "$V4_RESULTS"
python tools/fetch_irwm_bev_t.py \
  --source ckps/irwm_bev_t_large/tongji_bcde_seed901_20261111/best.pt \
  --output ckps/irwm_bev_t_large/release/best.pt \
  --sha256 b83eba4a33bb90031c43fea668087346a7e5a6af8aada53a8c5094f6f0703c2a \
  > "$V4_RESULTS/world_release.json"
python -m method.full_diffusionv4.bootstrap \
  --v3-checkpoint ckps/full_diffusionv3/release/epoch_10.pt \
  --world-checkpoint ckps/irwm_bev_t_large/release/best.pt \
  --output ckps/full_diffusionv4/initialization_bcde_20261112.pt \
  > "$V4_RESULTS/initialization.json"
python -m unittest discover -s tests -p test_full_diffusionv4.py -v \
  > "$V4_RESULTS/v4_tests.log" 2>&1
python -m unittest discover -s tests -p test_irwm_bev_t_large.py -v \
  > "$V4_RESULTS/world_tests.log" 2>&1
torchrun --standalone --nproc_per_node=8 tools/verify_v4_trained_world.py \
  --initial ckps/full_diffusionv4/initialization_bcde_20261112.pt \
  --output "$V4_RESULTS/cache_verification" \
  > "$V4_RESULTS/cache_verification.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python tools/smoke_v4_irwm.py \
  --world-checkpoint ckps/irwm_bev_t_large/release/best.pt \
  --output "$V4_RESULTS/full_policy_smoke" \
  > "$V4_RESULTS/full_policy_smoke.log" 2>&1
python - <<'PY'
from pathlib import Path
import json, os
base = Path('results/full_diffusionv4_bcde_20261112')
cache = json.loads((base / 'cache_verification/summary.json').read_text())
smoke = json.loads((base / 'full_policy_smoke/smoke.json').read_text())
assert cache['passed'] and cache['gpus'] == 8 and cache['sequences'] == 32
assert smoke['passed'] and not smoke['synthetic_world']
report = {'passed': True, 'job': os.environ['SLURM_JOB_ID'],
          'world_sha256': cache['world_sha256'], 'planner_requires_training': True,
          'allocation_action': 'none; existing allocation retained',
          'cache_verification': cache, 'full_policy_counts': smoke['counts']}
(base / 'completed.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
