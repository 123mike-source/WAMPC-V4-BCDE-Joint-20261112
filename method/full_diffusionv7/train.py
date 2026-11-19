"""Train V7 with controller-aligned speed supervision and frozen BCDE."""
import argparse
from copy import deepcopy
from datetime import timedelta
import json
import hashlib
import math
import os
from pathlib import Path
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
from method.full_diffusionv4.joint import detach_tree
from method.full_diffusionv4.joint_data import DistributedRouteStreams
from method.full_diffusionv4.joint_train import (barrier, close_loader, move, rank_world, restore_rng,
                                                 rng_state, seed)
from method.full_diffusionv4.world import sha256
from method.training_selection import final_checkpoint, record_no_validation, validation_enabled

from method.full_diffusionv6.data import V6Dataset, collate_with_predecessors
from method.full_diffusionv6.model import (FutureAdaptedPlanner, align_future, build_random_head_from_v3_metadata,
                                                 build_world, generate_future)
from method.full_diffusionv6.optimization import make_scheduler, mask_inactive_gradients, optimizer_groups, optimizer_report
from method.full_diffusionv6.rolling import QuarterSecondPlanStore

from .contract import CONTRACT, METHOD, TRAINING_FORMAT, validate_config, validate_manifest
from .probe import TrainingProbe, atomic_json, verify_best
from .speed import speed_supervision

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
        self.planner.base.train(mode and self.planner.base_training_mode)
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
        was_training = self.planner.training
        self.planner.eval()  # Roll the prior with inference-time dropout, also during scratch training.
        try:
            result = self.diffusion.sample(lambda x, t: self.planner(x, t, condition)['predictions'][-1],
                                           batch_size, device, generator, steps)
            return self.planner.codec.decode(result)
        finally:
            self.planner.train(was_training)

    @torch.no_grad()
    def predict_predecessor(self, observation, generator):
        device = observation['raw_bev'].device
        was_training = self.planner.training
        self.planner.eval()
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                condition = self.condition(observation, future=None)
                _, delta = self.sample(condition, len(observation['raw_bev']), device, generator,
                                       self.config['training']['rolling_inference_steps'])
            return delta.float()
        finally:
            self.planner.train(was_training)

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
            parts = self.planner.base.loss_components(
                output, clean, batch['spatial_mask'], batch['temporal_mask'],
                batch['behavior_targets'], batch['behavior_mask'], weights)
            _, predicted_delta = self.planner.codec.decode(output['predictions'][-1].float())
            speed, speed_absolute_error, speed_valid = speed_supervision(
                self.planner.codec, predicted_delta, batch['temporal_delta'], batch['temporal_mask'],
                batch['speed'], weights['speed_supervision'], self.config['labels']['stop_speed_mps'])
            total = parts['loss'] + weights['speed_supervision']['weight'] * speed
            anchor = sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
            _, delta = self.sample(detach_tree(condition), b, device, generator, settings['rolling_inference_steps'])
        return {'loss': total + anchor, 'spatial_l1': parts['spatial_l1'].detach(),
                'temporal_l1': parts['temporal_delta_l1'].detach(), 'behavior_ce': parts['behavior_ce'].detach(),
                'base_loss': parts['loss'].detach(), 'speed_huber': speed.detach(),
                'speed_abs_error_sum': (speed_absolute_error * speed_valid).sum().detach(),
                'speed_valid': speed_valid.float().sum(),
                'sampled_delta': delta.detach().float(),
                'future_ready': ready.float().sum(), 'prior_ready': prior_valid.any(-1).float().sum(),
                'prior_age_sum_seconds': ages.sum(), 'cache_age_max_seconds': ages.new_zeros(()),
                'valid_prior_points': prior_valid.float().sum()}


@torch.no_grad()
def training_prior(rolling, policy, batch, support, generator):
    prior, valid, ages = rolling.prior(batch['chain_id'], batch['current_pose'], batch['timestamp'])
    if support is None:
        return prior, valid, ages, 0
    rows, observations = support
    needed = ~valid.any(-1).cpu()[rows]
    rows = rows[needed]
    if not len(rows):
        return prior, valid, ages, 0
    observation = move({k: v[needed] for k, v in observations.items()}, prior.device)
    now = batch['timestamp'][rows]
    if not torch.equal(observation['timestamp'], now - .25):
        raise ValueError('Backfill must come from the same route exactly one raw frame before the anchor')
    scenes = [batch['chain_id'][i] for i in rows.tolist()]
    prediction = policy.predict_predecessor(observation, generator)
    rolling.update(scenes, observation['current_pose'], observation['timestamp'], prediction)
    targets = rows.to(prior.device)
    filled = rolling.prior(scenes, batch['current_pose'][targets], now)
    prior[targets], valid[targets], ages[targets] = filled
    return prior, valid, ages, len(rows)


def build(config, manifest, device):
    seed(config['seed'])
    validate_config(config)
    validate_manifest(manifest, config)
    path = ROOT / config['v3']['checkpoint']
    if sha256(path) != config['v3']['sha256']:
        raise ValueError('V3 checkpoint checksum differs')
    base, v3config, different = build_random_head_from_v3_metadata(path)
    if v3config['model']['history_frames'] != 2 or v3config['observation_interval_seconds'] != .25:
        raise ValueError('The model metadata must use three observations at 0.25s intervals')
    if v3config['labels'] != config['labels'] or manifest['label_config'] != config['labels']:
        raise ValueError('V7 labels/index must match the V3 normalization metadata')
    world = build_world(config, ROOT)
    if world.training or any(p.requires_grad for p in world.parameters()):
        raise RuntimeError('BCDE must be frozen in eval mode')
    if (world.provenance['teacher_sha256'] != manifest['teacher_sha256']
            or world.provenance['teacher_config_sha256'] != manifest['teacher_config_sha256']):
        raise ValueError('Index and world use different frozen perception identities')
    planner = FutureAdaptedPlanner(base, config['future_sampler'], train_base=True, base_training_mode=True)
    diffusion = SpatialTemporalVPFullDiffusion(base.codec, **{
        k: v for k, v in v3config['diffusion'].items() if k != 'inference_steps'})
    if config['diffusion']['inference_steps'] > diffusion.train_steps:
        raise ValueError('DDIM sampling steps exceed the diffusion training steps')
    policy = FreshWorldPlanner(planner, world, diffusion, config)
    named = dict(base.named_parameters())
    digest = hashlib.sha256()
    for n, p in named.items():
        digest.update(n.encode()); digest.update(p.detach().cpu().numpy().tobytes())
    policy.initialization_report = {
        'method': METHOD, 'experiment': config['experiment'], 'format': TRAINING_FORMAT,
        'planner_initialization': 'random', 'pretrained_parameter_import': False, 'seed': config['seed'],
        'normalization_source': str(path), 'normalization_source_sha256': config['v3']['sha256'],
        'parameter_tensors': len(named), 'different_from_pretrained_tensors': different,
        'initial_parameter_sha256': digest.hexdigest(),
        'training_dropout_enabled': True, 'world_frozen': True, 'world': world.provenance}
    return policy.to(device)


def planner_training_state(planner):
    return {'adapter': planner.adapter.state_dict(),
            'planner_base': planner.base.state_dict() if planner.train_base else None}


def load_planner_training_state(planner, state):
    base = state.get('planner_base')
    if planner.train_base != (base is not None):
        raise ValueError('Checkpoint is missing the fine-tuned planner, or uses a different tuning policy')
    planner.adapter.load_state_dict(state['adapter'], strict=True)
    if base is not None:
        planner.base.load_state_dict(base, strict=True)


def save(output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling, generator, device):
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
        state = {'format': TRAINING_FORMAT, 'training_contract': CONTRACT, 'method': METHOD,
                 'experiment': config['experiment'],
                 'config': config, **planner_training_state(policy.planner),
                 'world_tuned_state': None,
                 'world': policy.world.provenance, 'v3': config['v3'], 'dataset': metadata,
                 'progress': deepcopy(progress), 'gates': policy.planner.gates(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rank_states': states}
        torch.save(state, output / 'latest.partial')
        if (output / 'latest.pt').exists():
            (output / 'latest.pt').replace(output / 'latest.previous.pt')
        (output / 'latest.partial').replace(output / 'latest.pt')
    barrier(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config_train4hz_scratch10.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--smoke-steps-per-epoch', type=int)
    args = parser.parse_args()
    if not (os.environ.get('SLURM_JOB_ID') or os.environ.get('LSB_JOBID')):
        raise RuntimeError('Run only inside an existing allocation')
    config_payload = args.config.read_bytes()
    config = json.loads(config_payload)
    validate_config(config)
    expected_output = ROOT / 'ckps' / METHOD / config['experiment']
    if args.output.resolve() != expected_output:
        raise ValueError(f'V7 output must be {expected_output}')
    args.output = expected_output
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
    settings = config['training']
    if size != settings['world_size']:
        raise ValueError('DDP world size differs from the config')
    directory = ROOT / config['index_directory']
    manifest_payload = (directory / 'manifest.json').read_bytes()
    pinned = config['dataset']
    if hashlib.sha256(manifest_payload).hexdigest() != pinned['manifest_sha256']:
        raise ValueError('V7 index manifest differs from the pinned data')
    manifest = json.loads(manifest_payload)
    if manifest['identity'] != pinned['index_identity'] or validation_enabled():
        raise ValueError('V7 requires the pinned all-training selection')
    validation_items = len(np.load(directory / 'val.npy', mmap_mode='r', allow_pickle=False))
    if validation_items != pinned['validation_items']:
        raise ValueError('Unexpected validation rows in the V7 index')
    if rank == 0:
        if not args.resume:
            args.output.mkdir(parents=True, exist_ok=False)
            provenance = {'method': METHOD, 'experiment': config['experiment'], 'checkpoint_format': TRAINING_FORMAT,
                          'fresh_start': True, 'job_id': os.environ.get('SLURM_JOB_ID'),
                          'config_path': str(args.config.resolve()),
                          'config_sha256': hashlib.sha256(config_payload).hexdigest(),
                          'source_manifest_sha256': os.environ.get('V7_SOURCE_MANIFEST_SHA256'),
                          'output': str(args.output)}
            (args.output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n')
            print(json.dumps(provenance), flush=True)
    barrier(device)
    dataset = V6Dataset(directory, backfill_missing_prior=True)
    selection = dataset.training_selection
    if (selection is None or selection['policy_sha256'] != pinned['policy_sha256']
            or selection['validation_policy'] != 'disabled' or len(dataset) != pinned['train_items']):
        raise ValueError('Resolved V7 training rows differ from the pinned selection')
    policy = build(config, manifest, device)
    sampler = DistributedRouteStreams(dataset.chains(), settings['batch_per_gpu'], rank, size, config['seed'])
    optimizer = torch.optim.AdamW(optimizer_groups(policy, settings), fused=True,
                                  betas=tuple(settings['adam_betas']), eps=settings['adam_eps'])
    steps_per_epoch = math.ceil(len(dataset) / (size * settings['batch_per_gpu']))
    if args.smoke_steps_per_epoch is not None:
        if (not config['experiment'].startswith('v7_smoke_') or settings['epochs'] != 1
                or not 1 <= args.smoke_steps_per_epoch <= 16):
            raise ValueError('Smoke updates can only shorten a separately named single-epoch smoke run')
        steps_per_epoch = args.smoke_steps_per_epoch
    total_updates = settings['epochs'] * steps_per_epoch
    scheduler = make_scheduler(optimizer, settings, total_updates)
    metadata = {'index_directory': str(directory), 'index_identity': manifest['identity'],
                'index_manifest_sha256': pinned['manifest_sha256'], 'training_contract': CONTRACT,
                'training_selection': selection, 'train_items': len(dataset),
                'validation_items': validation_items, 'world_size': size,
                'global_items_per_update': size * settings['batch_per_gpu'], 'steps_per_epoch': steps_per_epoch,
                'total_updates': total_updates, 'prior_backfill': 'previous_raw_frame_prediction_no_world_no_grad'}
    rolling = QuarterSecondPlanStore()
    generator = torch.Generator(device=device).manual_seed(config['seed'] + rank + 100003)
    seed(config['seed'] + rank + 200003)
    progress = {'epoch': 0, 'step_in_epoch': 0, 'updates': 0}
    record_no_validation(progress)
    progress['validation']['checkpoint_selection'] = 'fixed_training_probe'
    progress['checkpoint_selection'] = 'fixed_training_probe'
    if args.resume:
        state = torch.load(args.output / 'latest.pt', map_location='cpu', weights_only=False)
        if (state.get('format') != TRAINING_FORMAT or state.get('training_contract') != CONTRACT
                or state.get('method') != METHOD or state.get('experiment') != config['experiment']):
            raise ValueError('Only a checkpoint from the same V7 experiment can resume this run')
        if state['config'] != config or state['dataset'] != metadata:
            raise ValueError('Resume config, index or world size differs')
        load_planner_training_state(policy.planner, state)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        local_state = state['rank_states'][rank]
        sampler.load_state_dict(local_state['sampler'])
        rolling.load_state_dict(local_state['rolling'])
        restore_rng(local_state['rng'], generator)
        progress = state['progress']
        del state
    if rank == 0:
        (args.output / 'initialization_report.json').write_text(json.dumps(policy.initialization_report, indent=2) + '\n')
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (args.output / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (args.output / 'optimizer_groups.json').write_text(json.dumps(optimizer_report(optimizer), indent=2) + '\n')
    probe = TrainingProbe(dataset, config, args.output, device, training_prior)
    reference = probe.reference(policy, load_planner_training_state, planner_training_state)
    if rank == 0:
        print('V7_REFERENCE_PROBE ' + json.dumps(reference, allow_nan=False), flush=True)
    ddp = DDP(policy, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False) if size > 1 else policy
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=settings['workers_per_rank'],
                        pin_memory=True, persistent_workers=bool(settings['workers_per_rank']),
                        collate_fn=collate_with_predecessors,
                        **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if settings['workers_per_rank'] else {}))
    iterator = iter(loader)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    policy.train()

    for epoch in range(progress['epoch'], settings['epochs']):
        first = progress['step_in_epoch'] if epoch == progress['epoch'] else 0
        for step in range(first, steps_per_epoch):
            begin = time.perf_counter()
            batch, support = next(iterator)
            batch = move(batch, device)
            data_wait = time.perf_counter() - begin
            prior, valid, ages, backfilled = training_prior(rolling, policy, batch, support, generator)
            optimizer.zero_grad(set_to_none=True)
            output = ddp(batch, prior, valid, ages, generator)
            if not bool(torch.isfinite(output['loss'])):
                raise FloatingPointError('Nonfinite V7 training loss')
            output['loss'].backward()
            mask_inactive_gradients(optimizer, progress['updates'])
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            used_rates = {g['name']: g['lr'] for g in optimizer.param_groups}
            optimizer.step()
            scheduler.step()
            rolling.update(batch['chain_id'], batch['current_pose'], batch['timestamp'], output['sampled_delta'])
            sampler.commit()
            progress.update(epoch=epoch, step_in_epoch=step + 1, updates=progress['updates'] + 1)
            if progress['updates'] < 5 or progress['updates'] % settings['log_every'] == 0:
                names = ['loss', 'base_loss', 'spatial_l1', 'temporal_l1', 'behavior_ce', 'speed_huber',
                         'speed_abs_error_sum', 'speed_valid', 'future_ready', 'prior_ready',
                         'prior_age_sum_seconds', 'valid_prior_points', 'cache_age_max_seconds']
                values = torch.stack([output[k].detach().float() for k in names]
                                     + [prior.new_tensor(backfilled)])
                names.append('prior_backfilled')
                if size > 1:
                    dist.all_reduce(values)
                if rank == 0:
                    summed = {'future_ready', 'prior_ready', 'prior_age_sum_seconds', 'valid_prior_points',
                              'prior_backfilled', 'speed_abs_error_sum', 'speed_valid'}
                    stats = {k: float(v) / (1 if k in summed else size) for k, v in zip(names, values)}
                    stats['prior_age_mean_seconds'] = stats['prior_age_sum_seconds'] / max(1, stats['prior_ready'])
                    stats['valid_points_per_prior'] = stats['valid_prior_points'] / max(1, stats['prior_ready'])
                    stats['prior_cold'] = size * len(prior) - stats['prior_ready']
                    stats['control_interval_denoiser_mae_mps'] = stats['speed_abs_error_sum'] / max(1, stats['speed_valid'])
                    record = {'experiment': config['experiment'], **progress, **stats,
                              'global_items': size * len(prior), 'seconds_per_update': time.perf_counter() - begin,
                              'data_wait_seconds': data_wait, 'gradient_norm': float(norm),
                              'gates': policy.planner.gates(),
                              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                              'learning_rates': used_rates,
                              'planner_tuning_enabled': policy.planner.train_base}
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
        progress.update(epoch=epoch + 1, step_in_epoch=0)
        probe.consider(policy, config, metadata, progress, reference, planner_training_state,
                       TRAINING_FORMAT, CONTRACT)
        save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
             generator, device)
    if rank == 0:
        final_checkpoint(args.output, progress, settings['epochs'])
        best = verify_best(args.output, settings['epochs'])
        if not (args.output / 'final.pt').is_file() or progress['updates'] != total_updates:
            raise ValueError('V7 training did not complete its full update budget')
        atomic_json(args.output / 'train_completed.json', dict(
            experiment=config['experiment'], job_id=os.environ['SLURM_JOB_ID'], progress=progress,
            best_epoch=best['epoch'], best_updates=best['updates'], best_sha256=best['checkpoint_sha256'],
            best_checkpoint=str(args.output / 'best.pt'), reference_probe=reference))
    barrier(device)
    close_loader(iterator)
    if size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
