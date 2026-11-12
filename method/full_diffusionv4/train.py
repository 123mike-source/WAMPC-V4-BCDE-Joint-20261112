"""Single-process V4 training with causal route streams and sampled rolling priors.

This entry point does not request/cancel allocations. Run it inside an existing
allocation. Frozen perception/world computation is shared by the loss pass and
the detached full-noise sample used as the next frame's prior.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from method.full_diffusion2.perception import FrozenPlannerPerception
from .bootstrap import ROOT, load_head
from .data import GarageWorldDataset, RouteStreamBatchSampler, validate_normalization_artifact
from .diffusion import SpatialTemporalVPFullDiffusion
from .geometry import motion_features
from .rolling import RollingPlanStore
from .world import TimedWorldModel, sha256

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def autocast(config, device):
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if device.type == 'cuda' and config['precision'] == 'bf16' else nullcontext())


@torch.no_grad()
def condition_inputs(batch, perception, world, rolling, config):
    bev, memory, ego = perception(batch)
    history = perception.encode_history(batch, chunk=config['history_encode_chunk'])
    prior, valid, age = rolling.prior(batch['scene_id'], batch['current_pose'], batch['timestamp'])
    raw = torch.cat((history, bev[:, None]), 1)
    transforms = torch.cat((batch['history_transforms'], torch.eye(3, device=bev.device).expand(len(bev), 1, 3, 3)), 1)
    frame_valid = torch.cat((batch['history_mask'], batch['history_mask'].new_ones(len(bev), 1)), 1)
    maps, future_valid, aligned = world(raw, transforms, frame_valid, prior, valid, batch['speed'])
    motion = motion_features(aligned, frame_valid, batch['speed'], prior, age)
    return (bev, memory, ego, batch['target_point'], history, batch['history_transforms'],
            batch['history_mask'], batch['speed'], batch['command'], maps, prior, future_valid, motion)


def losses_for(head, output, clean, batch, config):
    return head.loss_components(output, clean, batch['spatial_mask'], batch['temporal_mask'],
                                batch['behavior_targets'], batch['behavior_mask'], config['loss'])


@torch.no_grad()
def sample_and_update(head, diffusion, inputs, batch, rolling, generator, config):
    training = head.training
    head.eval()
    try:
        condition = head.prepare_condition(*inputs)
        sampled = diffusion.sample(lambda noisy, t: head(noisy, t, condition)['predictions'][-1],
                                   len(batch['path']), batch['path'].device, generator,
                                   config['diffusion']['inference_steps'])
        path, delta = head.codec.decode(sampled)
        rolling.update(batch['scene_id'], batch['current_pose'], batch['timestamp'], delta)
        return path, delta
    finally:
        head.train(training)


@torch.no_grad()
def validate(dataset, perception, world, head, diffusion, config, device, generator):
    rolling = RollingPlanStore(config['world']['max_prior_age_seconds'])
    sampler = RouteStreamBatchSampler(dataset.samples, config['training']['batch_size'], config['seed'])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(config['seed'] + 700001))
    sums = {'spatial_l1': 0., 'spatial_ade_m': 0., 'temporal_delta_l1': 0.,
            'future_ade_m': 0., 'behavior_correct': 0., 'behavior_count': 0.}
    counts = {'spatial': 0., 'temporal': 0., 'future': 0.}
    head.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast(config, device):
            inputs = condition_inputs(batch, perception, world, rolling, config)
            path, delta = sample_and_update(head, diffusion, inputs, batch, rolling, generator, config)
            condition = head.prepare_condition(*inputs)
        mask, temporal = batch['spatial_mask'], batch['temporal_mask']
        future = batch['future_position_mask'] & temporal.long().cumprod(1).bool()
        sums['spatial_l1'] += float(((path - batch['path']).abs().mean(-1) * mask).sum())
        sums['spatial_ade_m'] += float(((path - batch['path']).norm(dim=-1) * mask).sum())
        sums['temporal_delta_l1'] += float(((delta - batch['temporal_delta']).abs().mean(-1) * temporal).sum())
        sums['future_ade_m'] += float(((delta.cumsum(1) - batch['future_positions']).norm(dim=-1) * future).sum())
        predicted = condition['behavior_logits'].argmax(-1)
        sums['behavior_correct'] += float(((predicted == batch['behavior_targets']) & batch['behavior_mask']).sum())
        sums['behavior_count'] += float(batch['behavior_mask'].sum())
        counts['spatial'] += float(mask.sum())
        counts['temporal'] += float(temporal.sum())
        counts['future'] += float(future.sum())
    head.train()
    return {'spatial_l1': sums['spatial_l1'] / max(counts['spatial'], 1),
            'spatial_ade_m': sums['spatial_ade_m'] / max(counts['spatial'], 1),
            'temporal_delta_l1': sums['temporal_delta_l1'] / max(counts['temporal'], 1),
            'future_ade_m': sums['future_ade_m'] / max(counts['future'], 1),
            'behavior_accuracy': sums['behavior_correct'] / max(sums['behavior_count'], 1),
            'counts': counts, 'sequences': len(dataset)}


def save(path, head, optimizer, scheduler, config, dataset, world, rolling, generator, progress):
    state = {'format': 'full_diffusionv4_v1', 'model': head.state_dict(), 'config': config,
             'normalization': {'mean': head.codec.mean.tolist(), 'std': head.codec.std.tolist()},
             'world': world.provenance, 'dataset': dataset, 'progress': progress,
             'requires_training': progress['updates'] == 0,
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'rolling': rolling.state_dict(), 'generator': generator.get_state(),
             'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state() if next(head.parameters()).is_cuda else None,
                     'numpy': np.random.get_state(), 'python': random.getstate()}}
    temporary = path.with_suffix('.partial')
    torch.save(state, temporary)
    if path.exists():
        path.replace(path.with_name('latest.previous.pt'))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Use this causal trainer with one process; it does not support torchrun/DDP')
    device = torch.device(args.device)
    if device.type == 'cuda' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Start this trainer inside the existing Slurm allocation')
    checkpoint = args.output / 'latest.pt' if args.resume else args.initial
    head, config, state = load_head(checkpoint, device, allow_untrained=True)
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory or explicitly resume this run')
    settings = config['training']
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    artifact = json.loads((ROOT / config['normalization_file']).read_text())
    validate_normalization_artifact(artifact, config, state['normalization'])
    index = ROOT / config['index_directory']
    for name, digest in artifact['source_sha256'].items():
        if sha256(index / name) != digest:
            raise ValueError(f'Training index changed after normalization: {name}')
    metadata = {'method': 'full_diffusionv4', 'normalization': artifact,
                'index_manifest': json.loads((index / 'manifest.json').read_text()), 'data_root': config['data_root']}
    options = {'backbone_directory': ROOT / config['backbone_directory'], 'data_root': config['data_root'],
               'label_config': config['labels'], 'history_frames': 2, 'history_stride': config['history_stride']}
    dataset = GarageWorldDataset(index, **options)
    validation = GarageWorldDataset(index, split='val', **options)
    perception = FrozenPlannerPerception(ROOT / config['backbone_directory']).to(device).eval()
    world = TimedWorldModel(ROOT / config['world']['checkpoint'],
                            ROOT / config['backbone_directory'] / 'model_0030_1.pth',
                            config['world']['checkpoint_sha256']).to(device)
    if state['world']['sha256'] != world.provenance['sha256']:
        raise ValueError('Training and initialization world checkpoint identities differ')
    diffusion = SpatialTemporalVPFullDiffusion(head.codec, **{k: v for k, v in config['diffusion'].items() if k != 'inference_steps'})
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings['epochs'])
    generator = torch.Generator(device=device).manual_seed(config['seed'] + 100003)
    rolling = RollingPlanStore(config['world']['max_prior_age_seconds'])
    progress = {'epoch': 0, 'next_batch': 0, 'updates': 0, 'validation_pending': False}
    if args.resume:
        if state['dataset'] != metadata:
            raise ValueError('Resume data/normalization identity differs')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        rolling.load_state_dict(state['rolling'])
        generator.set_state(state['generator'])
        torch.set_rng_state(state['rng']['torch'])
        if state['rng']['cuda'] is not None:
            torch.cuda.set_rng_state(state['rng']['cuda'], device)
        np.random.set_state(state['rng']['numpy'])
        random.setstate(state['rng']['python'])
        progress = state['progress']
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if progress.get('validation_pending'):
        metrics = validate(validation, perception, world, head, diffusion, config, device, generator)
        progress.update(validation_pending=False, validation=metrics)
        save(args.output / 'latest.pt', head, optimizer, scheduler, config, metadata, world, rolling, generator, progress)
    start_epoch = progress['epoch']
    for epoch in range(start_epoch, settings['epochs']):
        dataset.epoch = epoch
        sampler = RouteStreamBatchSampler(dataset.samples, settings['batch_size'], config['seed'], epoch)
        offset = progress['next_batch'] if epoch == start_epoch else 0
        if offset == 0:
            rolling.reset()
        loader = DataLoader(dataset, batch_sampler=sampler.batches[offset:], num_workers=settings['workers'],
                            pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(config['seed'] + epoch + 200003),
                            **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if settings['workers'] else {}))
        head.train()
        for number, batch in enumerate(loader, offset):
            begin = time.perf_counter()
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(config, device):
                inputs = condition_inputs(batch, perception, world, rolling, config)
                clean = head.codec.encode(batch['path'], batch['temporal_delta'])
                noisy, t = diffusion.corrupt(clean)
                output = head(noisy, t, inputs)
                losses = losses_for(head, output, clean, batch, config)
            if not bool(torch.isfinite(losses['loss'])):
                raise FloatingPointError('Nonfinite V4 training loss')
            losses['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            with autocast(config, device):
                sample_and_update(head, diffusion, inputs, batch, rolling, generator, config)
            progress.update(epoch=epoch, next_batch=number + 1, updates=progress['updates'] + 1)
            record = {**progress, **{key: float(value.detach()) for key, value in losses.items()},
                      'gradient_norm': float(norm), 'seconds': time.perf_counter() - begin,
                      'future_ready': int(inputs[11][:, :5].any(-1).sum()), 'batch': len(batch['path'])}
            with (args.output / 'train.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            if progress['updates'] == 1 or progress['updates'] % settings['checkpoint_every'] == 0 or STOP:
                save(args.output / 'latest.pt', head, optimizer, scheduler, config, metadata, world, rolling, generator, progress)
                print(json.dumps(record), flush=True)
            if STOP:
                return
        scheduler.step()
        progress.update(epoch=epoch + 1, next_batch=0, validation_pending=True)
        save(args.output / 'latest.pt', head, optimizer, scheduler, config, metadata, world, rolling, generator, progress)
        metrics = validate(validation, perception, world, head, diffusion, config, device, generator)
        progress.update(validation_pending=False, validation=metrics)
        save(args.output / 'latest.pt', head, optimizer, scheduler, config, metadata, world, rolling, generator, progress)
        with (args.output / 'validation.jsonl').open('a') as handle:
            handle.write(json.dumps({'epoch': epoch + 1, **metrics}) + '\n')
        print(json.dumps({'epoch': epoch + 1, 'validation': metrics}), flush=True)


if __name__ == '__main__':
    main()
