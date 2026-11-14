"""Train V5 on single 4Hz route chains; each anchor gets a fresh world rollout."""
import argparse
from copy import deepcopy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv4.bootstrap import ROOT
from method.full_diffusionv4.data import RouteStreamBatchSampler
from method.full_diffusionv4.joint import detach_tree
from method.full_diffusionv4.joint_data import DistributedRouteStreams
from method.full_diffusionv4.joint_train import (barrier, close_loader, move, rank_world, restore_rng,
                                                 rng_state, seed)
from method.full_diffusionv4.world import sha256

from .data import V5Dataset
from .contract import CONTRACT, TRAINING_FORMAT, validate_config, validate_manifest
from .model import FutureAdaptedPlanner, align_future, build_world, generate_future, load_v3_head
from .rolling import QuarterSecondPlanStore

STOP = False


def stop(*_):
    global STOP
    STOP = True


class FreshWorldPlanner(nn.Module):
    """The detached t-0.25 prediction drives a new rollout at the current t."""
    def __init__(self, planner, world, diffusion, config):
        super().__init__()
        self.planner, self.world, self.config = planner, world, config
        self.diffusion = diffusion

    def train(self, mode=True):
        super().train(mode)
        self.planner.base.eval()
        self.world.eval()
        return self

    def future(self, x, prior, prior_valid, ages):
        available = prior_valid.any(-1)
        if bool((available & ((ages - .25).abs() > 1e-6)).any()):
            raise ValueError('A usable training prior must be exactly 0.25s old')
        if bool(prior_valid[:, -1].any()) or bool((available & ~prior_valid[:, :5].all(-1)).any()):
            raise ValueError('The advanced previous plan covers five future points; its sixth tail is invalid')
        future = generate_future(self.world, self.planner, x['raw_bev'], x['history_transforms'],
                                 x['history_mask'], prior.detach(), prior_valid, ages.detach(), x['speed'], x['ego'])
        if future is None:
            return None
        identity = torch.eye(3, device=prior.device).expand(len(prior), -1, -1)
        # Generation and conditioning happen at the same observation: cache age is always zero.
        return align_future(future, identity, torch.zeros(len(prior), device=prior.device))

    def condition(self, x, future):
        return self.planner.condition(x['raw_bev'][:, -1], x['memory'], x['ego'], x['target_point'],
                                      x['raw_bev'][:, :2], x['history_transforms'][:, :2],
                                      x['history_mask'][:, :2], x['speed'], x['command'], future)

    @torch.no_grad()
    def sample(self, condition, batch_size, device, generator, steps):
        result = self.diffusion.sample(lambda x, t: self.planner(x, t, condition)['predictions'][-1],
                                       batch_size, device, generator, steps)
        return self.planner.codec.decode(result)

    def forward(self, batch, prior, prior_valid, ages, generator):
        settings, weights = self.config['training'], self.config['loss']
        device, b = prior.device, len(prior)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            future = self.future(batch, prior, prior_valid, ages)
            ready = (future['valid'].any(-1) if future is not None
                     else torch.zeros(b, dtype=torch.bool, device=device))
            condition = self.condition(batch, future)
            clean = self.planner.codec.encode(batch['path'], batch['temporal_delta'])
            steps = torch.randint(self.diffusion.train_steps, (b,), device=device)
            steps = torch.where(torch.rand(b, device=device) < settings['max_noise_fraction'],
                                self.diffusion.train_steps - 1, steps)
            noisy, t = self.diffusion.corrupt(clean, steps)
            output = self.planner(noisy, t, condition)
            # Cold rows use frozen V3. They bootstrap the next prior, without an adapter loss.
            parts = self.planner.base.loss_components(
                output, clean, batch['spatial_mask'] & ready[:, None], batch['temporal_mask'] & ready[:, None],
                batch['behavior_targets'], batch['behavior_mask'] & ready[:, None], weights)
            total = weights['spatial_weight'] * parts['spatial_l1'] + weights['temporal_weight'] * parts['temporal_delta_l1']
            anchor = sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
            _, delta = self.sample(detach_tree(condition), b, device, generator, settings['rolling_inference_steps'])
        return {'loss': total + anchor, 'spatial_l1': parts['spatial_l1'].detach(),
                'temporal_l1': parts['temporal_delta_l1'].detach(), 'sampled_delta': delta.detach().float(),
                'future_ready': ready.float().sum(), 'prior_ready': prior_valid.any(-1).float().sum(),
                'prior_age_sum_seconds': ages.sum(), 'cache_age_max_seconds': ages.new_zeros(()),
                'valid_prior_points': prior_valid.float().sum()}


def build(config, manifest, device):
    seed(config['seed'])
    validate_config(config)
    validate_manifest(manifest, config)
    path = ROOT / config['v3']['checkpoint']
    if sha256(path) != config['v3']['sha256']:
        raise ValueError('V3 checkpoint checksum differs')
    base, v3config = load_v3_head(path)
    if v3config['model']['history_frames'] != 2 or v3config['observation_interval_seconds'] != .25:
        raise ValueError('The frozen V3 head must share the original three-observation 0.25s history')
    if v3config['labels'] != config['labels'] or manifest['label_config'] != config['labels']:
        raise ValueError('V5 labels/index must match the frozen V3 labels')
    world = build_world(config, ROOT, training=True)
    if (world.provenance['teacher_sha256'] != manifest['teacher_sha256']
            or world.provenance['teacher_config_sha256'] != manifest['teacher_config_sha256']):
        raise ValueError('Index and world use different frozen perception identities')
    planner = FutureAdaptedPlanner(base, config['future_sampler'])
    diffusion = SpatialTemporalVPFullDiffusion(base.codec, **{
        k: v for k, v in v3config['diffusion'].items() if k != 'inference_steps'})
    if config['diffusion']['inference_steps'] > diffusion.train_steps:
        raise ValueError('DDIM sampling steps exceed the diffusion training steps')
    return FreshWorldPlanner(planner, world, diffusion, config).to(device)


def optimizer_groups(policy, settings):
    rates, modules, groups = settings['learning_rates'], dict(policy.named_modules()), {}
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('world.'):
            group = 'world'
        elif name.endswith('.gate'):
            group = 'gate'
        elif name.endswith(('.distance_logit', '.time_logit')):
            group = 'distance'
        else:
            group = 'new'
        owner = modules[name.rpartition('.')[0]]
        plain = p.ndim > 1 and not isinstance(owner, (nn.LayerNorm, nn.Embedding))
        decay = (settings['weight_decay'] if group == 'new' else 1e-4 if group == 'world' else 0.) if plain else 0.
        key = (group, decay)
        if key not in groups:
            groups[key] = {'params': [], 'lr': rates[group], 'weight_decay': decay, 'name': f'{group}/wd{decay}'}
        groups[key]['params'].append(p)
    return list(groups.values())


def save(output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling, generator,
         device, best=False):
    rank, size = rank_world()
    trained = [p.detach().double() for p in policy.parameters() if p.requires_grad]
    checksum = torch.stack((sum(p.sum() for p in trained), sum(p.square().sum() for p in trained))).cpu().tolist()
    local = {'sampler': sampler.state_dict(), 'rolling': rolling.state_dict(), 'rng': rng_state(generator),
             'parameter_checksum': checksum}
    states = [None] * size if rank == 0 else None
    if size > 1:
        dist.gather_object(local, states, dst=0)
    else:
        states = [local]
    if rank == 0:
        if any(s['parameter_checksum'] != states[0]['parameter_checksum'] for s in states):
            raise RuntimeError('DDP parameters differ between ranks at checkpoint')
        state = {'format': TRAINING_FORMAT, 'training_contract': CONTRACT,
                 'config': config, 'adapter': policy.planner.adapter.state_dict(),
                 'world_tuned_state': policy.world.export_tuned() if config['world'].get('tuning') else None,
                 'world': policy.world.provenance, 'v3': config['v3'], 'dataset': metadata,
                 'progress': deepcopy(progress), 'gates': policy.planner.gates(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rank_states': states}
        torch.save(state, output / 'latest.partial')
        if (output / 'latest.pt').exists():
            (output / 'latest.pt').replace(output / 'latest.previous.pt')
        (output / 'latest.partial').replace(output / 'latest.pt')
        if best:
            shutil.copyfile(output / 'latest.pt', output / 'best.partial')
            (output / 'best.partial').replace(output / 'best.pt')
    barrier(device)


@torch.no_grad()
def validate(policy, config, device):
    """Use the same fresh-world 4Hz chain, with identical V5/V3 initial noise."""
    rank, size = rank_world()
    dataset = V5Dataset(ROOT / config['index_directory'], 'val')
    routes = sorted({int(r) for r in dataset.samples[:, 0]})
    keep = np.isin(dataset.samples[:, 0], routes[rank::size])
    dataset.samples = dataset.samples[keep]
    chains = dataset.chains()
    batch_size = min(config['training']['batch_per_gpu'], max(1, len(np.unique(chains[:, 0]))))
    loader = DataLoader(dataset, batch_sampler=RouteStreamBatchSampler(chains, batch_size, config['seed']),
                        num_workers=0)
    rolling = QuarterSecondPlanStore()
    generator = torch.Generator(device=device).manual_seed(config['seed'] + 500009 + rank)
    steps = config['diffusion']['inference_steps']
    # Groups: all target-valid frames, and the subset with usable world tokens.
    totals = torch.zeros(len(dataset.routes), 2, 5, dtype=torch.float64, device=device)
    policy.eval()
    for batch in loader:
        batch = move(batch, device)
        prior, valid, ages = rolling.prior(batch['chain_id'], batch['current_pose'], batch['timestamp'])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            future = policy.future(batch, prior, valid, ages)
            ready = future['valid'].any(-1) if future is not None else torch.zeros(len(prior), dtype=torch.bool, device=device)
            state = generator.get_state()
            path, delta = policy.sample(policy.condition(batch, future), len(prior), device, generator, steps)
            generator.set_state(state)
            base_path, base_delta = policy.sample(policy.condition(batch, None), len(prior), device, generator, steps)
            measured = torch.stack((
                masked_ade(path, batch['path'], batch['spatial_mask']),
                masked_ade(delta.cumsum(1), batch['future_positions'], batch['temporal_mask']),
                masked_ade(base_path, batch['path'], batch['spatial_mask']),
                masked_ade(base_delta.cumsum(1), batch['future_positions'], batch['temporal_mask']),
                torch.ones(len(prior), device=device)), -1).double()
            target_valid = batch['spatial_mask'].any(-1) & batch['temporal_mask'].any(-1)
            totals[:, 0].index_add_(0, batch['route_id'], measured * target_valid[:, None])
            totals[:, 1].index_add_(0, batch['route_id'], measured * (target_valid & ready)[:, None])
        rolling.update(batch['chain_id'], batch['current_pose'], batch['timestamp'], delta.float())
    if size > 1:
        dist.all_reduce(totals)
    metrics = {}
    for j, group in enumerate(('all', 'ready')):
        present = totals[:, j, -1] > 0
        macro = (totals[present, j, :4] / totals[present, j, -1:]).mean(0).tolist() if bool(present.any()) else [math.nan] * 4
        for name, value in zip(('v5_spatial_ade_m', 'v5_future_ade_m', 'v3_spatial_ade_m', 'v3_future_ade_m'), macro):
            metrics[f'{name}_{group}'] = value
        metrics[f'frames_{group}'] = int(totals[:, j, -1].sum())
        metrics[f'routes_{group}'] = int(present.sum())
    metrics['score_v5_future_ade_m'] = metrics['v5_future_ade_m_ready']
    metrics['gain_vs_v3_future_ade_m'] = metrics['v3_future_ade_m_ready'] - metrics['v5_future_ade_m_ready']
    if not math.isfinite(metrics['score_v5_future_ade_m']):
        raise RuntimeError('No finite world-ready validation score; refusing to select a best checkpoint')
    policy.train()
    return {**metrics, 'aggregation': 'route_macro, all frames and world-ready subset',
            'inference_steps': steps, 'training_contract': CONTRACT}


def masked_ade(prediction, target, mask):
    errors = (prediction.float() - target.float()).norm(dim=-1)
    return (errors * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config_train4hz.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int)
    args = parser.parse_args()
    if not (os.environ.get('SLURM_JOB_ID') or os.environ.get('LSB_JOBID')):
        raise RuntimeError('Run only inside an existing allocation')
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
    validate_config(config)
    settings = config['training']
    if size != settings['world_size']:
        raise ValueError('DDP world size differs from the config')
    directory = ROOT / config['index_directory']
    manifest = json.loads((directory / 'manifest.json').read_text())
    policy = build(config, manifest, device)
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            raise FileExistsError('Use a separate output directory or --resume')
        args.output.mkdir(parents=True, exist_ok=True)
    barrier(device)
    dataset = V5Dataset(directory)
    sampler = DistributedRouteStreams(dataset.chains(), settings['batch_per_gpu'], rank, size, config['seed'])
    optimizer = torch.optim.AdamW(optimizer_groups(policy, settings), fused=True)
    steps_per_epoch = math.ceil(len(dataset) / (size * settings['batch_per_gpu']))
    total_updates = settings['epochs'] * steps_per_epoch
    warmup = min(settings['warmup_updates'], max(1, total_updates - 1))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(1, total_updates - warmup))
        return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    metadata = {'index_identity': manifest['identity'], 'training_contract': CONTRACT,
                'train_items': len(dataset), 'world_size': size,
                'global_items_per_update': size * settings['batch_per_gpu'], 'steps_per_epoch': steps_per_epoch}
    rolling = QuarterSecondPlanStore()
    generator = torch.Generator(device=device).manual_seed(config['seed'] + rank + 100003)
    seed(config['seed'] + rank + 200003)
    progress = {'epoch': 0, 'step_in_epoch': 0, 'updates': 0, 'best_score': None, 'validation_pending': False}
    if args.resume:
        state = torch.load(args.output / 'latest.pt', map_location='cpu', weights_only=False)
        if state.get('format') != TRAINING_FORMAT or state.get('training_contract') != CONTRACT:
            raise ValueError('Only the same fresh-world 4Hz training checkpoint can resume this run')
        if state['config'] != config or state['dataset'] != metadata:
            raise ValueError('Resume config, index or world size differs')
        policy.planner.adapter.load_state_dict(state['adapter'], strict=True)
        if config['world'].get('tuning'):
            from method.full_diffusionv4.world_tuning import load_tuned_state
            load_tuned_state(policy.world, state['world_tuned_state'], config['world']['tuning'])
        optimizer.load_state_dict(state['optimizer'])
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
    ddp = DDP(policy, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False) if size > 1 else policy
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=settings['workers_per_rank'],
                        pin_memory=True, persistent_workers=bool(settings['workers_per_rank']),
                        **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if settings['workers_per_rank'] else {}))
    iterator = iter(loader)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    policy.train()

    def finish_epoch():
        metrics = validate(policy, config, device)
        score = metrics['score_v5_future_ade_m']
        best = progress['best_score'] is None or score < progress['best_score']
        progress.update(validation_pending=False, validation=metrics,
                        best_score=score if best else progress['best_score'])
        if rank == 0:
            with (args.output / 'validation.jsonl').open('a') as handle:
                handle.write(json.dumps({**progress, 'gates': policy.planner.gates()}) + '\n')
        save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
             generator, device, best=best)

    if progress['validation_pending']:
        finish_epoch()
    for epoch in range(progress['epoch'], settings['epochs']):
        first = progress['step_in_epoch'] if epoch == progress['epoch'] else 0
        for step in range(first, steps_per_epoch):
            begin = time.perf_counter()
            batch = move(next(iterator), device)
            data_wait = time.perf_counter() - begin
            prior, valid, ages = rolling.prior(batch['chain_id'], batch['current_pose'], batch['timestamp'])
            optimizer.zero_grad(set_to_none=True)
            output = ddp(batch, prior, valid, ages, generator)
            if not bool(torch.isfinite(output['loss'])):
                raise FloatingPointError('Nonfinite V5 adapter loss')
            output['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            rolling.update(batch['chain_id'], batch['current_pose'], batch['timestamp'], output['sampled_delta'])
            sampler.commit()
            progress.update(epoch=epoch, step_in_epoch=step + 1, updates=progress['updates'] + 1)
            if progress['updates'] < 5 or progress['updates'] % settings['log_every'] == 0:
                names = ['loss', 'spatial_l1', 'temporal_l1', 'future_ready', 'prior_ready',
                         'prior_age_sum_seconds', 'valid_prior_points', 'cache_age_max_seconds']
                values = torch.stack([output[k].detach().float() for k in names])
                if size > 1:
                    dist.all_reduce(values)
                if rank == 0:
                    summed = {'future_ready', 'prior_ready', 'prior_age_sum_seconds', 'valid_prior_points'}
                    stats = {k: float(v) / (1 if k in summed else size) for k, v in zip(names, values)}
                    stats['prior_age_mean_seconds'] = stats['prior_age_sum_seconds'] / max(1, stats['prior_ready'])
                    stats['valid_points_per_prior'] = stats['valid_prior_points'] / max(1, stats['prior_ready'])
                    record = {**progress, **stats,
                              'global_items': size * len(prior), 'seconds_per_update': time.perf_counter() - begin,
                              'data_wait_seconds': data_wait, 'gradient_norm': float(norm),
                              'gates': policy.planner.gates(),
                              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                              'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups}}
                    with (args.output / 'train.jsonl').open('a') as handle:
                        handle.write(json.dumps(record) + '\n')
                    print(json.dumps(record), flush=True)
            stop_now = torch.tensor(int(STOP or (args.output / 'STOP').exists()
                                        or (args.max_updates is not None and progress['updates'] >= args.max_updates)),
                                    device=device)
            if size > 1:
                dist.all_reduce(stop_now, op=dist.ReduceOp.MAX)
            if progress['updates'] == 1 or progress['updates'] % settings['checkpoint_every'] == 0 or bool(stop_now):
                save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
                     generator, device)
            if bool(stop_now):
                close_loader(iterator)
                if size > 1:
                    dist.destroy_process_group()
                return
        progress.update(epoch=epoch + 1, step_in_epoch=0, validation_pending=True)
        save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
             generator, device)
        finish_epoch()
    close_loader(iterator)
    if size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

