"""Finite L40 interface check; synthetic world weights are clearly labelled.

This validates the deployed full policy, actual TF++/V3 weights, BF16 execution,
history warmup and cached future integration. It does not evaluate driving or
claim a synthetic world model is a trained irwm_bev_t checkpoint.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world-checkpoint', type=Path)
    parser.add_argument('--synthetic-world', action='store_true')
    args = parser.parse_args()
    if bool(args.world_checkpoint) == args.synthetic_world:
        parser.error('Select one trained world checkpoint OR explicit synthetic interface testing')
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run in the existing Slurm allocation')
    if args.output.exists():
        raise FileExistsError(args.output)
    signal.alarm(180)
    import torch
    from method.full_diffusion.garage import FrozenGaragePerception
    from method.full_diffusionv4.bootstrap import initialize_from_v3
    from method.full_diffusionv4.inference.policy import FullDiffusionV4Network
    from method.full_diffusionv4.world import TimedWorldModel, sha256, timed_model_class
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Use exactly one GPU in a finite Slurm step')
    device = torch.device('cuda:0')
    args.output.mkdir(parents=True)
    teacher = ROOT / 'ckps/full_diffusion/backbone/all_towns_seed1/model_0030_1.pth'
    torch.manual_seed(901)
    world_path = args.world_checkpoint
    if args.synthetic_world:
        world_path = args.output / 'SYNTHETIC_world_interface_only.pt'
        model = timed_model_class()()
        torch.save({'model': model.state_dict(),
                    'config': {'method': 'irwm_bev_t', 'model': {'hidden': 96, 'history': 3, 'path_nodes': 6}},
                    'dataset': {'teacher_sha256': sha256(teacher), 'query_times_seconds': [.5, 1., 1.5, 2., 2.5, 3.]},
                    'synthetic_interface_test_only': True}, world_path)
        del model
    world = TimedWorldModel(world_path, teacher, allow_synthetic=args.synthetic_world)
    config = json.loads((ROOT / 'method/full_diffusionv4/config.json').read_text())
    config['world']['checkpoint'] = str(world_path.resolve())
    config['world']['checkpoint_sha256'] = sha256(world_path)
    v3_path = ROOT / 'ckps/full_diffusionv3/release/epoch_10.pt'
    source = torch.load(v3_path, map_location='cpu', weights_only=False)
    head, migration = initialize_from_v3(source, config)
    # Exercise the actual future readout, rather than leaving its bootstrap gates zero.
    for layer in list(head.layers) + list(head.temporal_layers):
        layer.future.gate.data.fill_(.1)
    initialization = args.output / 'SYNTHETIC_OR_UNTRAINED_planner_interface_only.pt'
    torch.save({'model': head.state_dict(), 'config': config, 'requires_training': True,
                'normalization': {'mean': head.codec.mean.tolist(), 'std': head.codec.std.tolist()},
                'world': world.provenance}, initialization)
    del source, head, world
    garage = FrozenGaragePerception()
    net = FullDiffusionV4Network(garage.net, initialization, device, allow_untrained=True,
                                allow_synthetic_world=args.synthetic_world)
    cfg = garage.config
    inputs = {'rgb': torch.rand(1, 3, cfg.cropped_height, cfg.cropped_width, device=device) * 255,
              'lidar_bev': torch.rand(1, (1 + int(cfg.use_ground_plane)) * cfg.lidar_seq_len,
                                      cfg.lidar_resolution_height, cfg.lidar_resolution_width, device=device),
              'target_point': torch.tensor([[10., 0.]], device=device),
              'ego_vel': torch.tensor([[5.]], device=device),
              'command': torch.tensor([[1., 0., 0., 0., 0., 0.]], device=device)}
    counts = {'head': 0, 'future_sampler': 0, 'world_evolve': 0}
    net.head.register_forward_hook(lambda *_: counts.__setitem__('head', counts['head'] + 1))
    net.head.future_sampler.register_forward_hook(lambda *_: counts.__setitem__('future_sampler', counts['future_sampler'] + 1))
    original = net.world.model._evolve
    def evolve(*positional, **keywords):
        counts['world_evolve'] += 1
        return original(*positional, **keywords)
    net.world.model._evolve = evolve
    frames = []
    begin = time.perf_counter()
    with torch.inference_mode():
        for frame in range(13):
            net.set_observation((0., 0., 0.), frame * .05, scene_id='synthetic-interface-scene')
            result = net(**inputs)
            if result[2].shape != (1, 6, 2) or net.last_temporal_delta.shape != (1, 6, 2):
                raise AssertionError('The policy did not preserve both six-point outputs')
            if not bool(torch.isfinite(result[2]).all()):
                raise AssertionError('Nonfinite policy output')
            frames.append({'time': frame * .05, 'world_active': net.last_world_active,
                           'future_valid': net.last_future_valid.cpu().tolist()[0]})
        controls = net.control_pid_direct(result[2][0], None, inputs['ego_vel'])
    torch.cuda.synchronize()
    if not any(record['world_active'] for record in frames):
        raise AssertionError('The world model never became active after history warmup')
    if counts['future_sampler'] != counts['world_evolve'] or counts['head'] != 13 * 25:
        raise AssertionError(f'World/future work was not cached per frame: {counts}')
    if not all(bool(torch.isfinite(torch.as_tensor(value))) for value in controls):
        raise AssertionError('Nonfinite controller output')
    # A new scene must not retain observations or a prediction from the old one.
    net.set_observation((0., 0., 0.), 1., scene_id='new-scene')
    if net.rolling.entries or net.history.entries:
        raise AssertionError('Scene reset leaked rolling state')
    report = {'passed': True, 'gpu': torch.cuda.get_device_name(0), 'job': os.environ['SLURM_JOB_ID'],
              'synthetic_world': args.synthetic_world, 'planner_requires_training': True,
              'input_kind': 'same-shaped synthetic sensors; real frozen TF++ and V3 initialization weights',
              'world_checkpoint_sha256': sha256(world_path), 'v3_checkpoint_sha256': sha256(v3_path),
              'inference_steps': 25, 'spatial_points': 6, 'temporal_points': 6, 'counts': counts,
              'frames': frames, 'controller_output': [float(value) for value in controls],
              'wall_seconds': time.perf_counter() - begin,
              'limitations': 'Interface validation only; no trained-world accuracy or CARLA driving score measured.'}
    (args.output / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
