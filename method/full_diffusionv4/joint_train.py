"""Eight-GPU causal planner training with reference-style partial world tuning."""
import argparse
from copy import deepcopy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .bootstrap import ROOT, initialize_from_v3
from .data import RouteStreamBatchSampler
from .joint import JointPlanner, optimizer_groups
from .joint_data import JointDataset, DistributedRouteStreams
from .batched_rolling import BatchedRollingPlanStore
from .world import sha256
from .world_tuning import PartialTimedWorld, TUNING, load_tuned_state

STOP = False


def stop(*_):
    global STOP
    STOP = True


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def barrier(device):
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) and k != 'timestamp' else v for k, v in batch.items()}


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def rng_state(generator):
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(), 'python': random.getstate(), 'generator': generator.get_state()}


def restore_rng(state, generator):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    generator.set_state(state['generator'])


def build(config, manifest, device):
    seed(config['seed'])
    original = ROOT / config['initialization']['checkpoint']
    if sha256(original) != config['initialization']['sha256']:
        raise ValueError('V3 initialization checksum differs')
    source = torch.load(original, map_location='cpu', weights_only=False)
    head, migration = initialize_from_v3(source, config, manifest['normalization'])
    del source
    world = PartialTimedWorld(ROOT / config['world']['checkpoint'],
                              ROOT / config['backbone_directory'] / 'model_0030_1.pth',
                              config['world']['checkpoint_sha256'], tuning=config['world']['tuning'])
    if (world.provenance['teacher_sha256'] != manifest['teacher_sha256']
            or world.provenance['teacher_config_sha256'] != manifest['teacher_config_sha256']):
        raise ValueError('Joint dataset and world have different frozen perception identities')
    policy = JointPlanner(head, world, config).to(device)
    if config['training'].get('channels_last', True):
        policy.world.model.to(memory_format=torch.channels_last)
        # Singleton kernels are both contiguous and channels-last; explicitly
        # canonicalize their strides before DDP builds gradient bucket views.
        for parameter in policy.world.model.parameters():
            if parameter.ndim == 4 and parameter.shape[-2:] == (1, 1):
                parameter.data = parameter.detach().clone(memory_format=torch.contiguous_format)
    return policy, migration


def save(output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling, generator,
         migration, device, best=False, diagnostic=False):
    rank, size = rank_world()
    trained = [p.detach().double() for p in policy.parameters() if p.requires_grad]
    checksum = torch.stack((sum(p.sum() for p in trained), sum(p.square().sum() for p in trained))).cpu().tolist()
    local = {'sampler': sampler.state_dict(), 'rolling': rolling.state_dict(), 'rng': rng_state(generator),
             'parameter_checksum': checksum}
    del trained
    local_states = [None] * size if rank == 0 else None
    if size > 1:
        dist.gather_object(local, local_states, dst=0)
    else:
        local_states = [local]
    if rank == 0:
        if any(s['parameter_checksum'] != local_states[0]['parameter_checksum'] for s in local_states):
            raise RuntimeError('DDP parameters differ between ranks at checkpoint')
        state = {'format': 'full_diffusionv4_v1', 'model': policy.head.state_dict(), 'config': config,
                 'normalization': {'mean': policy.head.codec.mean.tolist(), 'std': policy.head.codec.std.tolist()},
                 'world': policy.world.provenance, 'world_tuned_state': policy.world.export_tuned(),
                 'dataset': metadata, 'progress': deepcopy(progress), 'initialization': migration,
                 'requires_training': diagnostic or progress['updates'] == 0, 'diagnostic_only': diagnostic,
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rank_states': local_states}
        temporary = output / 'latest.partial'
        torch.save(state, temporary)
        latest = output / 'latest.pt'
        if latest.exists():
            latest.replace(output / 'latest.previous.pt')
        temporary.replace(latest)
        if best:
            shutil.copyfile(latest, output / 'best.partial')
            (output / 'best.partial').replace(output / 'best.pt')
    barrier(device)


@torch.no_grad()
def validate(policy, config, device, generator):
    rank, size = rank_world()
    dataset = JointDataset(ROOT / config['index_directory'], 'val')
    routes = sorted(set(int(r) for r in dataset.samples[:, 0]))
    assigned = routes[rank::size]
    dataset.samples = dataset.samples[np.isin(dataset.samples[:, 0], assigned)]
    batch_size = min(config['training']['batch_per_gpu'], max(1, len(assigned)))
    sampler = RouteStreamBatchSampler(dataset.samples, batch_size, config['seed'])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(config['seed'] + rank + 300003))
    rolling = BatchedRollingPlanStore(config['world']['max_prior_age_seconds'])
    totals = torch.zeros(len(dataset.routes), 5, dtype=torch.float64, device=device)
    policy.eval()
    for batch in loader:
        batch = move(batch, device)
        prior, valid, ages = rolling.prior(batch['scene_id'], batch['current_pose'], batch['timestamp'])
        condition, _ = policy.conditions(batch, prior, valid, ages)
        path, delta = policy.sample(condition, len(prior), device, generator, config['diffusion']['inference_steps'])
        rolling.update(batch['scene_id'], batch['current_pose'], batch['timestamp'], delta)
        measured = torch.stack(((path - batch['path']).float().norm(dim=-1).mean(1),
                                (delta.cumsum(1) - batch['future_positions']).float().norm(dim=-1).mean(1),
                                (delta - batch['temporal_delta']).float().abs().mean((1, 2)),
                                (condition['behavior_logits'].argmax(-1) == batch['behavior_targets']).double().mean(1)), -1).double()
        totals.index_add_(0, batch['route_id'], torch.cat((measured, torch.ones(len(prior), 1, device=device)), -1))
    if size > 1:
        dist.all_reduce(totals)
    present = totals[:, -1] > 0
    if int(present.sum()) != 44 or int(totals[:, -1].sum()) != 1906:
        raise ValueError('Validation did not score every retained route and sequence exactly once')
    macro = (totals[present, :4] / totals[present, -1:]).mean(0).tolist()
    metrics = dict(zip(('spatial_ade_m', 'future_ade_m', 'temporal_delta_l1_m', 'behavior_accuracy'), macro))
    policy.train()
    return {**metrics, 'routes': 44, 'sequences': 1906, 'aggregation': 'route_macro',
            'inference_steps': config['diffusion']['inference_steps']}


def synthetic_batch(batch_size, memory_shape, ego_shape, device):
    delta = torch.tensor([.5, 0.], device=device).expand(batch_size, 6, 2).clone()
    return {'raw_bev': torch.randn(batch_size, 3, 64, 64, 64, device=device).bfloat16(),
            'memory': torch.randn(batch_size, *memory_shape, device=device).bfloat16(),
            'ego': torch.randn(batch_size, *ego_shape, device=device).bfloat16(),
            'history_transforms': torch.eye(3, device=device).expand(batch_size, 3, 3, 3).clone(),
            'history_mask': torch.ones(batch_size, 3, dtype=torch.bool, device=device),
            'speed': torch.ones(batch_size, device=device), 'command': torch.zeros(batch_size, 6, device=device),
            'target_point': torch.ones(batch_size, 2, device=device), 'path': delta.cumsum(1),
            'temporal_delta': delta, 'spatial_mask': torch.ones(batch_size, 6, dtype=torch.bool, device=device),
            'temporal_mask': torch.ones(batch_size, 6, dtype=torch.bool, device=device),
            'behavior_targets': torch.full((batch_size, 3), 2, dtype=torch.long, device=device),
            'behavior_mask': torch.ones(batch_size, 3, dtype=torch.bool, device=device)}


def close_loader(iterator):
    # First finish already-prefetched transfers. Exiting workers while their
    # tensor queue feeder is still active can abort during C++ teardown.
    shutdown = getattr(iterator, '_shutdown_workers', None)
    if shutdown is not None:
        iterator._sampler_iter = iter(())
        for _ in iterator:
            pass
        shutdown()


def benchmark(policy, migration, config, output, device, candidates):
    rank, size = rank_world()
    context_path = next((ROOT / config['index_directory'] / 'context').glob('*.pt'))
    context = torch.load(context_path, map_location='cpu', weights_only=True)
    groups, _ = optimizer_groups(policy, migration)
    optimizer = torch.optim.AdamW(groups, fused=True)
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    policy.train()
    for layer in list(policy.head.layers) + list(policy.head.temporal_layers):
        layer.future.gate.data.fill_(.1)
    results = []
    for batch_size in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        elapsed, failed = 0., 0
        try:
            batch = synthetic_batch(batch_size, context['memory'].shape[1:], context['ego'].shape[1:], device)
            prior = batch['temporal_delta']
            valid = torch.ones(batch_size, 6, dtype=torch.bool, device=device)
            valid[:, -1] = False
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = policy(batch, prior, valid, torch.full((batch_size,), .5, device=device), generator)
                result['loss'].backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                if step > 0:
                    elapsed += time.perf_counter() - start
            del batch, result
        except torch.cuda.OutOfMemoryError:
            failed = 1
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        state = torch.tensor([failed, elapsed / 2, torch.cuda.max_memory_allocated() / 2**30], device=device)
        if size > 1:
            dist.all_reduce(state, op=dist.ReduceOp.MAX)
        item = {'batch_per_gpu': batch_size, 'oom': bool(state[0]), 'seconds': float(state[1]),
                'peak_allocated_gib': float(state[2]), 'samples_per_second': 0. if bool(state[0]) else batch_size * size / float(state[1])}
        results.append(item)
        if rank == 0:
            print(json.dumps({'event': 'benchmark', **item}), flush=True)
        if bool(state[0]):
            break
    usable = [x for x in results if not x['oom'] and x['peak_allocated_gib'] < 40.]
    if not usable:
        raise RuntimeError('No safe batch candidate passed the partial-world stress test')
    best = max(usable, key=lambda x: x['samples_per_second'])
    if rank == 0:
        (output / 'benchmark.json').write_text(json.dumps({'results': results, 'selected': best,
            'scope': 'all ranks run worst-case ready rollout; synthetic inputs; diagnostic parameters discarded'}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--batch-candidates', type=int, nargs='+', default=[16, 32, 64, 96])
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run only inside the existing Slurm allocation')
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    rank, size = rank_world()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    config = json.loads(args.config.read_text())
    settings = config['training']
    if config['world']['tuning'] != TUNING or size != settings['world_size']:
        raise ValueError('Unexpected partial tuning contract or DDP world size')
    directory = ROOT / config['index_directory']
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['label_config'] != config['labels']:
        raise ValueError('Prepared labels differ from the planner config')
    policy, migration = build(config, manifest, device)
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            raise FileExistsError('Use a separate output directory or resume')
        args.output.mkdir(parents=True, exist_ok=True)
    barrier(device)
    if args.benchmark:
        benchmark(policy, migration, config, args.output, device, args.batch_candidates)
        if size > 1:
            dist.destroy_process_group()
        return
    dataset = JointDataset(directory)
    sampler = DistributedRouteStreams(dataset.samples, settings['batch_per_gpu'], rank, size, config['seed'])
    groups, group_audit = optimizer_groups(policy, migration)
    optimizer = torch.optim.AdamW(groups, fused=True)
    steps_per_epoch = math.ceil(len(dataset) / (size * settings['batch_per_gpu']))
    total_updates = settings['epochs'] * steps_per_epoch
    warmup = min(settings['warmup_updates'], max(1, total_updates - 1))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(1, total_updates - warmup))
        return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    metadata = {'prepared_identity': manifest['identity'], 'train_samples': len(dataset),
                'val_samples': manifest['val_samples'], 'source_sha256': manifest['source_sha256'],
                'world_size': size, 'steps_per_epoch': steps_per_epoch}
    rolling = BatchedRollingPlanStore(config['world']['max_prior_age_seconds'])
    generator = torch.Generator(device=device).manual_seed(config['seed'] + rank + 100003)
    seed(config['seed'] + rank + 200003)
    progress = {'epoch': 0, 'step_in_epoch': 0, 'updates': 0, 'best_future_ade_m': None, 'validation_pending': False}
    if args.resume:
        state = torch.load(args.output / 'latest.pt', map_location='cpu', weights_only=False)
        if state['config'] != config or state['dataset'] != metadata or state['diagnostic_only'] != bool(args.max_updates):
            raise ValueError('Resume config, dataset, world size or diagnostic identity differs')
        policy.head.load_state_dict(state['model'], strict=True)
        load_tuned_state(policy.world, state['world_tuned_state'], config['world']['tuning'])
        optimizer.load_state_dict(state['optimizer'])
        for p, saved in optimizer.state.items():
            for key, value in saved.items():
                if torch.is_tensor(value):
                    saved[key] = value.to(p.device)
        scheduler.load_state_dict(state['scheduler'])
        local_state = state['rank_states'][rank]
        sampler.load_state_dict(local_state['sampler'])
        rolling.load_state_dict(local_state['rolling'])
        restore_rng(local_state['rng'], generator)
        progress = state['progress']
        del state
    if rank == 0:
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (args.output / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (args.output / 'optimizer_groups.json').write_text(json.dumps(group_audit, indent=2) + '\n')
        (args.output / 'startup.json').write_text(json.dumps({'job': os.environ['SLURM_JOB_ID'], 'world': policy.world.provenance,
            'global_batch': size * settings['batch_per_gpu'], 'steps_per_epoch': steps_per_epoch,
            'total_updates': total_updates, 'normalization': manifest['normalization'],
            'code_sha256': {f.name: sha256(f) for f in Path(__file__).parent.glob('*.py')},
            'reference': config['reference'], 'allocation_policy': 'retain after success or failure'}, indent=2) + '\n')
    ddp = DDP(policy, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True,
              bucket_cap_mb=64, find_unused_parameters=False) if size > 1 else policy
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=settings['workers_per_rank'],
                        pin_memory=True, persistent_workers=bool(settings['workers_per_rank']),
                        generator=torch.Generator().manual_seed(config['seed'] + rank + 400003),
                        **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if settings['workers_per_rank'] else {}))
    iterator = iter(loader)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    policy.train()
    if progress['validation_pending']:
        metrics = validate(policy, config, device, generator)
        score = metrics['future_ade_m']
        best = progress['best_future_ade_m'] is None or score < progress['best_future_ade_m']
        progress.update(validation_pending=False, validation=metrics,
                        best_future_ade_m=score if best else progress['best_future_ade_m'])
        save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
             generator, migration, device, best=best, diagnostic=bool(args.max_updates))
    for epoch in range(progress['epoch'], settings['epochs']):
        first = progress['step_in_epoch'] if epoch == progress['epoch'] else 0
        for step in range(first, steps_per_epoch):
            begin = time.perf_counter()
            batch = move(next(iterator), device)
            data_wait = time.perf_counter() - begin
            prior, valid, ages = rolling.prior(batch['scene_id'], batch['current_pose'], batch['timestamp'])
            optimizer.zero_grad(set_to_none=True)
            output = ddp(batch, prior, valid, ages, generator)
            if not bool(torch.isfinite(output['loss'])):
                raise FloatingPointError('Nonfinite joint planner/world loss')
            output['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            logging = progress['updates'] < 5 or (progress['updates'] + 1) % settings['log_every'] == 0
            world_norm = None
            if logging:
                world_norm = sum(p.grad.float().square().sum() for p in policy.world.parameters() if p.requires_grad and p.grad is not None).sqrt()
            optimizer.step()
            scheduler.step()
            rolling.update(batch['scene_id'], batch['current_pose'], batch['timestamp'], output['sampled_delta'])
            sampler.commit()
            progress.update(epoch=epoch, step_in_epoch=step + 1, updates=progress['updates'] + 1)
            if logging:
                values = torch.stack([output[k].detach().float() for k in ('loss', 'spatial_l1', 'temporal_delta_l1', 'behavior_ce', 'future_ready')])
                if size > 1:
                    dist.all_reduce(values)
                timing = torch.tensor([time.perf_counter() - begin, torch.cuda.max_memory_allocated() / 2**30,
                                       float(world_norm), data_wait], device=device)
                if size > 1:
                    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                if rank == 0:
                    record = {**progress, **dict(zip(('loss', 'spatial_l1', 'temporal_delta_l1', 'behavior_ce'), (values[:4] / size).tolist())),
                              'future_ready_global': int(values[4]), 'global_batch': size * len(prior),
                              'seconds_per_update': float(timing[0]), 'samples_per_second': size * len(prior) / float(timing[0]),
                              'data_wait_seconds': float(timing[3]),
                              'peak_allocated_gib': float(timing[1]), 'world_gradient_norm': float(timing[2]),
                              'gradient_norm': float(norm), 'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups}}
                    with (args.output / 'train.jsonl').open('a') as handle:
                        handle.write(json.dumps(record) + '\n')
                    print(json.dumps(record), flush=True)
            stop_now = torch.tensor(int(STOP or (args.output / 'STOP').exists()
                                        or (args.max_updates is not None and progress['updates'] >= args.max_updates)), device=device)
            if size > 1:
                dist.all_reduce(stop_now, op=dist.ReduceOp.MAX)
            if progress['updates'] == 1 or progress['updates'] % settings['checkpoint_every'] == 0 or bool(stop_now):
                save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
                     generator, migration, device, diagnostic=bool(args.max_updates))
            if bool(stop_now):
                if rank == 0:
                    (args.output / 'stopped.json').write_text(json.dumps(progress, indent=2) + '\n')
                close_loader(iterator)
                if size > 1:
                    dist.destroy_process_group()
                return
        progress.update(epoch=epoch + 1, step_in_epoch=0, validation_pending=True)
        save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
             generator, migration, device, diagnostic=bool(args.max_updates))
        metrics = validate(policy, config, device, generator)
        score = metrics['future_ade_m']
        best = progress['best_future_ade_m'] is None or score < progress['best_future_ade_m']
        progress.update(validation_pending=False, validation=metrics,
                        best_future_ade_m=score if best else progress['best_future_ade_m'])
        save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
             generator, migration, device, best=best, diagnostic=bool(args.max_updates))
        if rank == 0:
            with (args.output / 'validation.jsonl').open('a') as handle:
                handle.write(json.dumps({'epoch': epoch + 1, **metrics}) + '\n')
            print(json.dumps({'event': 'validation', 'epoch': epoch + 1, **metrics}), flush=True)
    if rank == 0:
        (args.output / 'completed.json').write_text(json.dumps(progress, indent=2) + '\n')
    close_loader(iterator)
    if size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
