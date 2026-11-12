# Full Diffusion V4: six spatial points + six temporal intervals + timed IRWM

This module extends the existing V3 planner. V3 sources, checkpoints, results and
the active 25-step default remain separate from this new implementation.

The default world is now the **trained BCDE `irwm_bev_t_large`**, selected at
epoch 21 from the completed 24-epoch Tongji run. Its 44-route macro validation
NMSE is 0.0419037948. The release is pinned to SHA256
`b83eba4a33bb90031c43fea668087346a7e5a6af8aada53a8c5094f6f0703c2a`;
the original training checkpoint is retained under
`ckps/irwm_bev_t_large/tongji_bcde_seed901_20261111/best.pt`.

## Network and frame contract

- Spatial branch: six geometric points resampled at original node indices
  `[0,1.8,3.6,5.4,7.2,9]`, retaining the original covered path extent.
- Temporal branch: six consecutive metric XY displacements, at 0.5s intervals.
- Independent 256D queries, decoder stacks, time/behavior projections and future
  readouts; both branches receive observed conditions and the detached old plan.
- Diffusion state `[B,12,2]`, default 25 DDIM steps; temporal PID is unchanged.
- The six-point spatial output is interpolated to the old geometric lattice only
  inside the lateral controller adapter. Network outputs/labels/losses stay six-point.

The pretrained model is **WAMPC BCDE `irwm_bev_t_large`** (14,830,336
parameters, main/coarse widths 192/384). It takes three raw TF++ BEVs at
-0.5/-0.25/0s and predicts six
future BEVs `[B,6,64,64,64]`. Their coordinate frame stays at the current input
observation, width=x forward, height=y right, extent +/-32m.

## Adaptation of the reference

Reference repository: https://github.com/123mike-source/CVCI2_Res_V11_PlanA

Reference commit: `be6b7842adb8f28d2f05068d8f28206e2b3581ac`.

- Current BEV: centre, tangent +/-1m and normal +/-1m at each candidate point;
  content-based weights start with centre mass 0.8 and 0.05 on each neighbour.
- Future BEVs: five retained horizons at 0.5..2.5s, point budgets `[9,9,16,25,36]`
  (95 tokens), exact centre anchors, inner/outer sampling rings, motion-dependent
  learnable offsets and bounded tangent/normal radii. Initial half-windows are
  `[6m,3m]` times `[.5,.625,.75,.875,1]`; learned multipliers are bounded to [.5,1.5].
- Geometry-dependent distance bias, per-point reliability, separate scalar gates
  in each spatial/temporal decoder. New scalar gates start at zero and need training.
- Sampling and K/V preparation happen once per frame outside the DDIM loop.

The reference uses translated 200x200/256-channel future maps with matched centre
zero. This adaptation uses fixed-frame 64x64/64-channel maps: the centre is the
cumulative retained plan position. No extra translation and no x/y swap is applied.

## Causal rolling

`RollingPlanStore` stores detached decoded temporal predictions per route/scene.
For every observation it advances the old plan by **elapsed seconds**, interpolates
inside its known horizon, then applies `inverse(T_current) @ T_source` to endpoints.
It never removes a half-second node after a 0.05s CARLA tick and never extrapolates
the unknown tail. Unavailable action prefixes/maps are masked. The sixth horizon
is not read into the 95 tokens, following the five-retained-horizon reference.

World rollout is disabled until all three observations and a usable previous
prediction exist. Scene changes, backwards timestamps and gaps >0.5s reset the
prior. Stopped trajectories use ego-forward as the sampling-axis fallback.

`TimedWorldModel` dispatches by checkpoint method and architecture; the original
96-channel model remains supported. All parameters are strictly loaded and frozen,
with world/teacher SHA256, teacher preprocessing config, coordinate/action contract
and time-grid checks. Prefix masks reuse the existing `_evolve` interface.
Future GT images/poses/features are not model inputs.

## Weight preparation

Use the completed local BCDE run. The release copy and its provenance manifest
are created explicitly; the tool refuses to overwrite an existing destination:

```bash
source .runtime/activate_tongji_v3.sh
python tools/fetch_irwm_bev_t.py \
  --source ckps/irwm_bev_t_large/tongji_bcde_seed901_20261111/best.pt \
  --output ckps/irwm_bev_t_large/release/best.pt \
  --sha256 b83eba4a33bb90031c43fea668087346a7e5a6af8aada53a8c5094f6f0703c2a
```

The planner initialization embeds this world identity and refuses a different
checkpoint. Updating the world model requires explicitly rebinding and training
the planner. Random test weights are never substituted for the trained release.

## Normalization and explicit initialization

New heads cannot strictly load the old ten-point planner. A migration tool copies
matching weights, retains temporal weights and remaps spatial point-dependent
weights. New future modules start fresh. Its output is labelled `requires_training`.

Use the retained training index to measure the actual new labels; old variances
are not sufficient to infer variances of interpolated geometric points:

```bash
python tools/prepare_v4_normalization.py --index data/full_diffusion/pdm_lite_seed901
python -m method.full_diffusionv4.bootstrap \
  --v3-checkpoint ckps/full_diffusionv3/release/epoch_10.pt \
  --world-checkpoint ckps/irwm_bev_t_large/release/best.pt \
  --normalization data/full_diffusion/pdm_lite_seed901_v4_normalization.json \
  --output ckps/full_diffusionv4/initialization.pt
```

Bootstrap without `--normalization` is available for architecture initialization,
using explicitly labelled approximate V3 affine scales. The training entry point
requires actual measured V4 moments. It refuses mismatched labels/index hashes.

The Tongji joint-training index is prepared from the retained world-training split
at `data/full_diffusionv4/pdm_seed901_bcde_v1`: 222607 training samples,
1906 validation samples and the same 44 validation routes. Its manifest contains
measured six-point moments and source/label/index identities. The raw dataset remains
`/share/home/u21012/data/bench2drive-PDMlite`.
The statistics tool uses the configured raw data root; `--data-root` supports a
relocated copy without reusing an old machine's manifest path. Artifact identities
include this path. Measured moments are compared at the codec's float32 precision,
so a JSON float64-to-float32 conversion does not produce a false mismatch.

## Training

### Eight-GPU partial-world joint training

Inside the user's existing allocation:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m method.full_diffusionv4.joint_train \
  --config method/full_diffusionv4/config_joint.json \
  --output ckps/full_diffusionv4/tongji_bcde_joint_seed901_20261112
```

`config_joint.json` follows the pinned Plan A partial-tuning recipe. BCDE's final
transition residual block, detail decoder and action-fusion MLP are trainable;
observed/history/path encoders and other world weights stay frozen. The world
stays in eval mode, uses FP32 with TF32 acceleration, and retains the complete
five-step recursive gradient graph with nonreentrant activation checkpointing.
Planner losses supervise it through future-feature sampling; there is no
additional world BEV/OCC loss in this joint stage.

World LR is 1e-6; copied context/planner LRs are 1e-5/2e-5; new weights use 1e-4;
future gates/current-point logits/distance controls use 1e-3/1e-3/1e-2. Warmup is
500 updates, followed by cosine decay to 10% of each base LR. The planner uses
BF16, fused AdamW, gradient clipping 1.0 and future dropout 0.1. Training's
detached rolling prediction uses one DDIM step, matching the reference recipe;
validation and deployment use 25 steps.

Frozen current-observation memory/ego tokens are cached once. The original
teacher cache supplies only three observed BEVs; future measurements create
supervision labels only. Each global stream has a disjoint route pool. An epoch
is `ceil(train_samples/global_batch)` updates; streams continue across epochs
and can cycle their pool rather than enforcing an exact single pass. Checkpoints
save the consumed stream cursor independently of DataLoader prefetch, per-rank
rolling state/RNG, optimizer/scheduler, head and selected world tensors. All-rank
parameter checksums must agree before committing a checkpoint. Resume requires
the same config, dataset, batch and world size. Validation scores all 1906 frames
and reports macro averages over all 44 routes.

The inference loader restores selected world updates over the pinned original
checkpoint and uses the same FP32 five-map implementation. Training and all
diagnostics leave the existing Slurm allocation alive on success or failure.

### Earlier single-process frozen-world entry

```bash
# Inside an existing allocation, with config paths and measured moments ready:
python -m method.full_diffusionv4.train \
  --initial ckps/full_diffusionv4/initialization.pt \
  --output results/full_diffusionv4_irwm_s6t6_train_v1
```

This earlier entry point is single-process. It does not submit/resize/cancel GPU
allocations. Use one visible GPU; it explicitly rejects torchrun/DDP.

Each batch contains independent route streams in chronological order. The actual
previous full-noise sampled prediction supplies the next prior, with no GT action
teacher forcing. Perception/world run once; sampling reuses those frozen results.
Augmentation is disabled to preserve the pretrained world feature contract.

Loss retains V3's branch weighting: spatial L1 + 2*weighted temporal L1 +
0.4*behavior CE. Both decoder layers are supervised. Temporal weights stay
`[3,3,2,2,1,1]`. Validation reports sampled spatial/future ADE, displacement L1
and behavior accuracy. The frozen world model has no new OCC/feature loss here.

Checkpoints include optimizer, scheduler, sampler offset, rolling predictions,
all RNG states, normalization and teacher/world identities. SIGINT/SIGTERM saves
at a completed update boundary. `--resume` uses this run's `latest.pt`.

## CARLA and verification

Agent: `method/full_diffusionv4/inference/carla_agent.py`, entry point
`FullDiffusionV4Agent`; its checkpoint must be trained V4. Existing V3 evaluation
launchers retain their V3 agent selection.

```bash
python -m unittest discover -s tests -p test_full_diffusionv4.py -v
```

Tests cover real numerical geometry, 0.05s timing, masks, 95-token sample values,
independent branches, live future-gate gradients, DDIM call count, weight migration,
and strict original-model forward equivalence. World weights generated inside
unit tests are explicitly synthetic; they do not validate a trained checkpoint's
prediction accuracy or new closed-loop driving performance.

A finite L40 interface check using the real trained world is available inside an
existing allocation:

```bash
python tools/smoke_v4_irwm.py \
  --output results/full_diffusionv4_bcde_interface_UNIQUE_RUN \
  --world-checkpoint ckps/irwm_bev_t_large/release/best.pt
```

This diagnostic uses real TF++, V3 initialization and trained world weights, with
synthetic sensor inputs. The saved planner is explicitly marked as requiring
training; diagnostic gates are opened in this test only. Normal inference refuses
the untrained planner. The older `--synthetic-world` mode remains available for
isolated interface regression tests.

For independent checks on all eight GPUs and 32 held-out cached sequences:

```bash
torchrun --standalone --nproc_per_node=8 tools/verify_v4_trained_world.py \
  --initial ckps/full_diffusionv4/initialization_bcde_20261112.pt \
  --output results/full_diffusionv4_bcde_20261112/cache_verification
```

This checks trained-forward equivalence, prefix masks, missing-history bypass,
0.05s rolling, 95-token sampling, both branches' gradients and future-feature
influence, branch independence and 25-step caching. Logged future actions are used
as diagnostic conditions for feature accuracy; online rolling still uses previous
model predictions. Reported subset errors are sample means, not the full 44-route
macro score. The earlier single-process trainer is separate from the joint DDP entry.

The new future gates and six-point spatial head need training before driving
evaluation. Prefix-masked rolling uses the pretrained parameters but has a
different action context from full-horizon training. The finite checks quantify
this on real cached features; they do not establish closed-loop driving quality.
All verification tools use an existing allocation and leave its resources allocated.
