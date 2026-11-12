"""Finite multi-GPU integration checks using the trained BCDE world and real cache.

Launch with torchrun inside an existing Slurm step. Each GPU checks independent
held-out samples; this is not a distributed planner training run. No scheduler
allocation is submitted, cancelled or released by this tool.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_matrices(relative):
    import torch
    matrix = torch.eye(3, device=relative.device).expand(*relative.shape[:-1], 3, 3).clone()
    c, s = relative[..., 2].cos(), relative[..., 2].sin()
    matrix[..., 0, 0], matrix[..., 0, 1] = c, -s
    matrix[..., 1, 0], matrix[..., 1, 1] = s, c
    matrix[..., :2, 2] = relative[..., :2]
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=Path, default=ROOT / 'data/irwm_bev_t/pdm_seed901_dev44_dt05_tongji_v1')
    parser.add_argument('--samples-per-rank', type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.samples_per_rank < 1:
        raise RuntimeError('Use a finite step in an existing Slurm allocation')
    signal.alarm(600)
    import numpy as np
    import torch
    import torch.distributed as dist
    from method.full_diffusionv4.bootstrap import load_head
    from method.full_diffusionv4.diffusion import SpatialTemporalVPFullDiffusion
    from method.full_diffusionv4.geometry import align_history, motion_features
    from method.full_diffusionv4.rolling import advance_plan
    from method.full_diffusionv4.world import TimedWorldModel
    from method.irwm_bev_t_large.data import SequenceDataset, TeacherCache, collate, prepare
    from method.irwm_bev_t_large.loss import error_metrics

    rank, size = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(1)
    torch.manual_seed(901 + rank)
    if size > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    args.output.mkdir(parents=True, exist_ok=True)
    report_file = args.output / f'rank_{rank}.json'
    if report_file.exists():
        raise FileExistsError(report_file)
    head, config, state = load_head(args.initial, device, allow_untrained=True)
    if not state.get('requires_training'):
        raise ValueError('Use the explicitly marked V4 initialization for this diagnostic')
    teacher = ROOT / config['backbone_directory'] / 'model_0030_1.pth'
    world = TimedWorldModel(ROOT / config['world']['checkpoint'], teacher,
                            config['world']['checkpoint_sha256']).to(device).eval()
    if world.provenance['method'] != 'irwm_bev_t_large' or world.provenance['progress']['epoch'] != 21:
        raise AssertionError('The integration is not using the selected trained BCDE checkpoint')
    if state['world']['sha256'] != world.provenance['sha256']:
        raise AssertionError('Initialization has a different world identity')
    parameters = sum(p.numel() for p in world.parameters())
    if parameters != 14830336 or any(p.requires_grad for p in world.parameters()):
        raise AssertionError('World architecture/freeze differs')
    data = SequenceDataset(args.index, 'val', ROOT / 'data/irwm_bev/teacher_cache')
    teacher_cache = TeacherCache(device)
    indices = np.linspace(0, len(data) - 1, size * args.samples_per_rank, dtype=int)[rank::size]
    samples, last = [], None

    with torch.no_grad():
        for index in indices:
            batch = collate([data[int(index)]])
            raw = teacher_cache(batch['sensors'])[:, :3]
            inputs, supervision, _ = prepare(batch, teacher_cache, device)
            transforms = source_matrices(batch['source_in_anchor'][:, :3].to(device))
            frames = torch.ones(1, 3, dtype=torch.bool, device=device)
            valid = torch.ones(1, 6, dtype=torch.bool, device=device)
            delta, speed = inputs['temporal_delta'], inputs['speed']
            full, _, aligned = world(raw, transforms, frames, delta, valid, speed)
            _, pixels = align_history(raw, transforms, frames)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                direct = world.model(aligned, pixels, delta, speed)
            torch.testing.assert_close(full, direct, atol=0, rtol=0)

            # Online elapsed time leaves five known actions and an unknown tail.
            prefix = valid.clone()
            prefix[:, -1] = False
            masked, returned_valid, _ = world(raw, transforms, frames, delta, prefix, speed)
            changed = delta.clone()
            changed[:, -1] = 1e5
            again, _, _ = world(raw, transforms, frames, changed, prefix, speed)
            torch.testing.assert_close(masked, again, atol=0, rtol=0)
            if bool(returned_valid[:, -1].any()) or not bool(returned_valid[:, :5].all()):
                raise AssertionError('Unknown rolling tail became valid')
            inactive, _, _ = world(raw, transforms, frames, delta, torch.zeros_like(valid), speed)
            if inactive is not None:
                raise AssertionError('Cold start must bypass world rollout')
            cold_frames = frames.clone()
            cold_frames[:, 0] = False
            inactive, _, _ = world(raw, transforms, cold_frames, delta, valid, speed)
            if inactive is not None:
                raise AssertionError('Incomplete observations must bypass world rollout')

            full_metric = error_metrics(full, **supervision)
            prefix_supervision = {**supervision, 'query_valid': prefix}
            prefix_metric = error_metrics(masked, **prefix_supervision)
            copy_metric = error_metrics(aligned[:, -1, None].expand_as(full), **prefix_supervision)
            first5 = supervision['spatial_valid'][:, :5].flatten(2).any(-1)
            reduce5 = lambda value: float((value[:, :5] * first5).sum() / first5.sum().clamp_min(1))
            identity = torch.eye(3, dtype=torch.float64, device=device)
            shifted, shifted_valid = advance_plan(delta[0], identity, identity, .05)
            if shifted_valid.tolist() != [True] * 5 + [False] or not bool(torch.isfinite(shifted).all()):
                raise AssertionError('0.05s rolling must retain five known endpoints')
            samples.append({'index': int(index), 'route_id': int(batch['route_id'][0]),
                            'full_first5_nmse': reduce5(full_metric['nmse']),
                            'masked_tail_first5_nmse': reduce5(prefix_metric['nmse']),
                            'copy_current_first5_nmse': reduce5(copy_metric['nmse']),
                            'future_shape': list(full.shape)})
            last = (raw, transforms, frames, masked, delta, prefix, aligned, speed)

    # Check that real predicted BEVs reach both branches and can train new gates.
    raw, transforms, frames, maps, delta, valid, aligned, speed = last
    head.eval()
    memory = torch.randn(1, 5, 256, device=device)
    ego = torch.randn(1, 1, 256, device=device)
    motion = motion_features(aligned, frames, speed, delta, speed.new_tensor([.05]))
    observed = (raw[:, -1].float(), memory, ego, delta.cumsum(1)[:, 0], raw[:, :2].float(),
                transforms[:, :2], frames[:, :2], speed,
                torch.tensor([[1., 0., 0., 0., 0., 0.]], device=device))
    sampler_calls = []
    hook = head.future_sampler.register_forward_hook(lambda *_: sampler_calls.append(1))
    def condition(future_maps=maps):
        return head.prepare_condition(*observed, future_maps, delta, valid, motion)
    noisy = head.codec.encode(delta.cumsum(1), delta)
    time = torch.tensor([.5], device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prepared = condition()
        if prepared['future'][0]['tokens'].shape != (1, 95, 256):
            raise AssertionError('Future neighbourhood token budget changed')
        output = head(noisy, time, prepared)
    if output['predictions'].shape != (2, 1, 12, 2):
        raise AssertionError('Two independent six-point branch stacks are required')
    output['predictions'].float().square().mean().backward()
    layers = list(head.layers) + list(head.temporal_layers)
    gates = [float(layer.future.gate.grad) for layer in layers]
    if not all(np.isfinite(gates)) or not any(abs(v) > 0 for v in gates[:2]) or not any(abs(v) > 0 for v in gates[2:]):
        raise AssertionError(f'Both branches need finite nonzero future gate gradients: {gates}')
    if any(p.grad is not None for p in world.parameters()):
        raise AssertionError('Frozen world received planner gradients')
    head.zero_grad(set_to_none=True)
    # Exercise the learned readout diagnostically; the saved initialization stays zero-gated.
    for layer in layers:
        layer.future.gate.data.fill_(.1)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        linked = head(noisy, time, condition())['predictions']
        ablated = head(noisy, time, condition(torch.zeros_like(maps)))['predictions']
        influence = [(linked[:, :, :6] - ablated[:, :, :6]).abs().max(),
                     (linked[:, :, 6:] - ablated[:, :, 6:]).abs().max()]
        if not all(bool(torch.isfinite(v)) and float(v) > 0 for v in influence):
            raise AssertionError('Real future features did not affect both branches')
        prepared = condition()
        altered = noisy.clone()
        altered[:, :6] += .01
        original = head(noisy, time, prepared)['predictions']
        independent = head(altered, time, prepared)['predictions']
        torch.testing.assert_close(original[:, :, 6:], independent[:, :, 6:], atol=0, rtol=0)
        altered = noisy.clone()
        altered[:, 6:] += .01
        independent = head(altered, time, prepared)['predictions']
        torch.testing.assert_close(original[:, :, :6], independent[:, :, :6], atol=0, rtol=0)
        before, calls = len(sampler_calls), []
        diffusion = SpatialTemporalVPFullDiffusion(head.codec)
        def denoise(x, t):
            calls.append(1)
            return head(x, t, prepared)['predictions'][-1]
        sampled = diffusion.sample(denoise, 1, device, torch.Generator(device=device).manual_seed(901), 25)
        if len(calls) != 25 or len(sampler_calls) != before or sampled.shape != (1, 12, 2):
            raise AssertionError('Future sampling was repeated inside denoising or slots changed')
    hook.remove()
    report = {'passed': True, 'rank': rank, 'world_size': size, 'job': os.environ['SLURM_JOB_ID'],
              'gpu': torch.cuda.get_device_name(local), 'world': world.provenance,
              'world_parameters': parameters, 'samples': samples,
              'future_gate_gradients': gates, 'future_influence_max': [float(v) for v in influence],
              'spatial_points': 6, 'temporal_points': 6, 'denoiser_calls': len(calls),
              'sampler_calls_inside_denoising': len(sampler_calls) - before,
              'planner_requires_training': True,
              'limitations': 'Finite held-out cache check; real logged actions are diagnostic conditions. '
                             'Planner memory/ego tokens are synthetic. No planner training or CARLA score.'}
    report_file.write_text(json.dumps(report, indent=2) + '\n')
    if size > 1:
        dist.barrier()
    if rank == 0:
        reports = [json.loads((args.output / f'rank_{i}.json').read_text()) for i in range(size)]
        all_samples = [s for r in reports for s in r['samples']]
        summary = {'passed': all(r['passed'] for r in reports), 'gpus': size,
                   'job': os.environ['SLURM_JOB_ID'], 'sequences': len(all_samples),
                   'world_sha256': world.provenance['sha256'],
                   'sample_mean_first5_nmse': {k: float(np.mean([s[k] for s in all_samples])) for k in
                                              ('full_first5_nmse', 'masked_tail_first5_nmse', 'copy_current_first5_nmse')},
                   'metric_scope': 'sample mean over selected sequences and first five horizons; not 44-route macro',
                   'planner_requires_training': True, 'allocation_action': 'none; allocation retained'}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)
    if size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
