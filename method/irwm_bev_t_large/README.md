# IRWM timed BEV: combined B+C+D+E

This independent module extends `irwm_bev_t`. Its input/output, fixed current-ego
coordinates and six half-second action intervals remain compatible. No original
world-model, V3, or existing checkpoint is overwritten.

The default combined model has 14,830,336 parameters, compared with 1,916,320 in
the original width-96 model (teacher and planner excluded). This includes the new
depth/multiscale modules, not just widening the original model.

## Architecture

- B: recurrent/path width 192, with a separate 384-channel context scale.
- C: four history residual blocks at 32x32 and two transition residual blocks per
  recurrent step. The six recurrent steps reuse the same dynamics parameters.
- D: history features at 64x64 / 32x32 / 16x16, coarse context fusion, and a 64x64
  detail skip into each future decoder. The expensive 192->192 decoder convolution
  runs at 32x32; high-resolution refinement operates with 64 output channels.
- E: base NMSE + .5 corridor NMSE + .05 cosine loss, plus .25 feature-change-region
  NMSE + .1 temporal-change NMSE + .1 multiscale NMSE. These weights are initial
  experimental settings, not measured optimal values.

Each prediction is still 64 channels x 64 x 64. `forward` only receives the three
observed teacher BEVs, their spatial masks, six metric displacement actions and
current speed. The inherited timed interface derives the auxiliary target speed
from the second displacement. Recorded future XY positions construct that explicit
action condition. Future teacher features and yaw do not enter predictor inputs;
poses also construct the aligned targets. Rollout feeds its own predictions back at all six steps.
The last residual readout is zero-initialized, preserving the copy-current start.

### Auxiliary objectives

Features are aligned to the fixed input ego frame before any loss is computed.

- Dynamic weights are detached, bounded weights from consecutive target feature
  changes, including current->first future. They are an approximate feature-change
  signal, not annotated moving-object segmentation. Completely static targets have
  zero dynamic loss.
- Temporal supervision compares predicted and target consecutive feature changes,
  normalized by feature power rather than near-zero motion energy.
- Multiscale supervision averages masked NMSE at pooled 32x32 and 16x16 scales.
  A pooled cell is valid only when its full source footprint is valid.
- Temporal/dynamic masks intersect validity at both adjacent instants. Invalid
  values are sanitized before arithmetic. Future GT BEV features remain loss-only.

All six loss components are included in `train.update`, logged separately and
available during validation. Best-checkpoint selection remains 44-route macro NMSE
so comparison with the original model retains the same principal metric. Validation
also reports auxiliary errors, each future horizon, route scores and copy-current.

## Data prerequisites

Reuse the original immutable TF++ BF16 feature cache and the filtered timed index,
including the exact original 44-route split and teacher identity. The original
`method/irwm_bev_t/prepare.py` documents the continuous -0.5..+3s filter. Set the
relocated index/cache paths in this module's config before training; no raw-sensor
fallback or substitute teacher is used.

The Tongji evaluation checkout currently lacks this cache/index. The new training
starts from random initialization and does not need the old world checkpoint.
This change implements and checks BCDE; it does not claim that a full training run
or driving evaluation has completed.

## Training inside an existing allocation

The default is 8 ranks x batch 4 x accumulation 1 = effective batch 32, AdamW 2e-4,
weight decay .01, 500-update warmup, cosine decay to 10%, 24 epochs, BF16 autocast.
The trainer supports different rank/microbatch/accumulation combinations when their
product stays 32; set `world_size` and the other fields before starting a new run.
The launcher does not allocate, cancel, resize, or release GPUs.

```bash
cd /share/home/u21012/nixiran/WAMPC
source .runtime/activate_tongji_v3.sh
export WMMPC_ROOT="$PWD"
export TMPDIR=/tmp
# Run this within the user's existing 8-GPU Slurm/tmux shell:
torchrun --standalone --nproc-per-node=8 -m method.irwm_bev_t_large.train \
  --config method/irwm_bev_t_large/config.json \
  --output results/irwm_bev_t_large_bcde_192_NEW_RUN
```

This is a new architecture trained from random initialization. Original 96-channel
weights cannot load strictly into it. Checkpoints record `method=irwm_bev_t_large`,
`architecture=multiscale_recurrent_192_v1`, exact config, dataset/index identities,
optimizer/scheduler, completed update boundary and per-rank RNG. `latest.pt` is
atomic, `latest.previous.pt` retains the preceding checkpoint, `best.pt` tracks
macro NMSE and epoch weights are retained. `--resume` requires matching config,
data and world size. A `STOP` file in the output or SIGTERM saves at an update
boundary; it never calls scontrol/scancel or exits the user's parent shell.

Each training rank also creates a private compute-node temporary directory for
worker sockets/files. This avoids the NFS open-file cleanup errors observed with
the evaluation environment's shared `.runtime/tmp` default. Spawn workers finish
before synthetic verification fixtures are removed.

## V4 integration

`method/full_diffusionv4/world.py` dispatches by checkpoint method/architecture and
strictly loads either original timed weights or this new architecture. It preserves
teacher SHA256 validation, frozen inference and prefix-masked rolling. The weight
fetch tool accepts both identities; use a separate destination such as
`ckps/irwm_bev_t_large/release/best.pt`. New weights require a new planner checkpoint
bound to their SHA256; existing V4 checkpoints retain their original world identity.

## Verification

`tools/verify_irwm_bcde.sh OUTPUT` runs finite compute-node tests: full default-width
forward/backward, pyramid shapes, 32x32 decoder computation, six-step API, action
intervention, prefix masks, auxiliary-loss effects, NaN masking, exact continuation,
V4 legacy/large checkpoint loading and two-rank CPU/Gloo accumulated updates.
Synthetic tensors/test weights are explicitly labelled and normal loading rejects
them. These checks measure numerical/interface behavior, not trained BEV accuracy.

When GPUs are already allocated, `tools/smoke_irwm_bcde_cuda.py` supports a finite
torchrun check of the full default-width model: BF16 forward/backward, actual NCCL
replica agreement, optimizer step count, memory, and batch-one six-map forward
timing. Eight ranks x batch four exercises effective batch 32 with synthetic BEV
targets. It does not save a trained weight or claim downstream driving accuracy.
