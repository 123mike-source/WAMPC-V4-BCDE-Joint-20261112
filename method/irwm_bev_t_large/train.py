"""Slurm-compatible BCDE training; uses existing allocations and cached targets."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler

from . import ARCHITECTURE
from .data import SequenceDataset, TeacherCache, collate, prepare
from .loss import METRICS, error_metrics, loss_components
from .model import ActionBEVWorldModel
from .runtime import configure_worker_tmp

ROOT = Path(__file__).resolve().parents[2]
STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def distributed():
    return dist.is_available() and dist.is_initialized()


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if distributed() else (0, 1)


def prepare_output(output, resume=False):
    """One rank checks/creates output before peers can race against its writes."""
    rank, _ = rank_world()
    errors = [None]
    if rank == 0:
        try:
            if not resume and output.exists() and any(output.iterdir()):
                raise FileExistsError('Use a new output directory or resume this exact BCDE run')
            output.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            errors[0] = str(error)
    if distributed():
        dist.broadcast_object_list(errors, src=0)
    if errors[0] is not None:
        raise FileExistsError(errors[0])
    if distributed():
        dist.barrier()


def autocast(config, device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and config['precision'] == 'bf16' else nullcontext()


def update(prepared_batches, policy, optimizer, scheduler, config, device):
    optimizer.zero_grad(set_to_none=True)
    totals = {}
    for number, (inputs, supervision, _) in enumerate(prepared_batches):
        sync = policy.no_sync() if isinstance(policy, DDP) and number + 1 < len(prepared_batches) else nullcontext()
        with sync:
            with autocast(config, device):
                prediction = policy(**inputs)
            values = loss_components(prediction, **supervision, settings=config['loss'])
            if not bool(torch.isfinite(values['loss'])):
                raise FloatingPointError('Nonfinite BCDE training loss')
            (values['loss'] / len(prepared_batches)).backward()
        for name, value in values.items():
            totals[name] = totals.get(name, value.detach().new_zeros(())) + value.detach() / len(prepared_batches)
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return totals, norm.detach()


def rng_state(device):
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def save_checkpoint(output, model, optimizer, scheduler, progress, config, manifest, device,
                    metrics=None, best=False, epoch_file=False, synthetic=False):
    rank, world = rank_world()
    states = [None] * world
    if distributed():
        dist.all_gather_object(states, rng_state(device))
    else:
        states[0] = rng_state(device)
    if rank == 0:
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'progress': dict(progress), 'config': config,
                 'dataset': manifest, 'rng_by_rank': states, 'metrics': metrics,
                 'synthetic_interface_test_only': synthetic}
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / 'latest.partial'
        torch.save(state, temporary)
        latest = output / 'latest.pt'
        if latest.exists():
            latest.replace(output / 'latest.previous.pt')
        temporary.replace(latest)
        if best:
            # An independent atomic snapshot avoids filesystem hardlink assumptions.
            target = output / 'best.partial'
            torch.save(state, target)
            target.replace(output / 'best.pt')
        if epoch_file:
            weights = {key: value for key, value in state.items() if key not in ('optimizer', 'scheduler', 'rng_by_rank')}
            torch.save(weights, output / f"epoch_{progress['epoch']:04}.pt")
    if distributed():
        dist.barrier()


def load_checkpoint(path, model, optimizer, scheduler, config, manifest, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    rank, world = rank_world()
    if (state['config'] != config or state['dataset'] != manifest
            or len(state['rng_by_rank']) != world or state.get('synthetic_interface_test_only')):
        raise ValueError('Resume requires this architecture, objective, dataset, and world size')
    if state['progress']['next_batch'] % config['accumulation_steps']:
        raise ValueError('Resume must use a completed optimizer update boundary')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng_by_rank'][rank], device)
    return dict(state['progress'])


@torch.no_grad()
def validate(dataset, model, teacher, config, device):
    rank, world = rank_world()
    selected = list(range(rank, len(dataset), world))
    loader = DataLoader(torch.utils.data.Subset(dataset, selected), batch_size=config['batch_per_gpu'],
                        num_workers=0, collate_fn=collate,
                        generator=torch.Generator().manual_seed(config['seed'] + 500003))
    count = len(METRICS)
    totals = torch.zeros(len(dataset.routes), count * 2 + 1, device=device, dtype=torch.float64)
    horizons = torch.zeros(6, count * 2 + 1, device=device, dtype=torch.float64)
    model.eval()
    for batch in loader:
        inputs, supervision, routes = prepare(batch, teacher, device)
        with autocast(config, device):
            prediction = model(**inputs)
        measured = error_metrics(prediction, **supervision, settings=config['loss'])
        baseline = error_metrics(inputs['history'][:, -1, None].expand_as(prediction),
                                 **supervision, settings=config['loss'])
        values = torch.stack([measured[name] for name in METRICS] + [baseline[name] for name in METRICS], -1).double()
        valid = supervision['query_valid'] & supervision['spatial_valid'].flatten(2).any(-1)
        sequence = (values * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        totals.index_add_(0, routes, torch.cat((sequence, torch.ones(len(routes), 1, device=device)), 1))
        horizons[:, :-1] += (values * valid[..., None]).sum(0)
        horizons[:, -1] += valid.sum(0)
    if distributed():
        dist.all_reduce(totals)
        dist.all_reduce(horizons)
    present = totals[:, -1] > 0
    if int(present.sum()) != 44:
        raise ValueError(f'Expected all 44 original validation routes; got {int(present.sum())}')
    macro = (totals[present, :-1] / totals[present, -1, None]).mean(0).cpu().tolist()
    result = {'routes': 44, 'sequences': int(totals[:, -1].sum()),
              'model': dict(zip(METRICS, macro[:count])),
              'copy_current': dict(zip(METRICS, macro[count:])),
              'by_query': [{'time_seconds': .5 * (i + 1), 'count': int(horizons[i, -1]),
                            **dict(zip(METRICS, (horizons[i, :count] / horizons[i, -1].clamp_min(1)).cpu().tolist()))}
                           for i in range(6)],
              'per_route': {dataset.routes[i]: dict(zip(METRICS, (totals[i, :count] / totals[i, -1]).cpu().tolist()))
                            for i in present.nonzero().flatten().tolist()}}
    model.train()
    if not math.isfinite(result['model']['nmse']):
        raise FloatingPointError('Nonfinite validation NMSE')
    return result


def check_config(config, world):
    if config['method'] != 'irwm_bev_t_large' or config['architecture'] != ARCHITECTURE:
        raise ValueError('Use the separate BCDE method and architecture identity')
    if (world != config['world_size'] or world * config['batch_per_gpu'] * config['accumulation_steps'] != config['effective_batch']
            or config['effective_batch'] != 32):
        raise ValueError('Actual world size and microbatch/accumulation must give effective batch 32')
    for name in ('nmse', 'corridor_nmse', 'cosine', 'dynamic_nmse', 'temporal_nmse', 'multiscale_nmse'):
        if config['loss'][name] < 0:
            raise ValueError('Loss weights must be nonnegative')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside an existing Slurm allocation; this entry point never submits or cancels jobs')
    configure_worker_tmp()
    config = json.loads(args.config.read_text())
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    check_config(config, world)
    device = torch.device('cuda', local) if args.device == 'cuda' else torch.device('cpu')
    torch.set_num_threads(1)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=15))
    rank, world = rank_world()
    prepare_output(args.output, args.resume)
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    index, cache = ROOT / config['index'], ROOT / config['cache']
    train = SequenceDataset(index, 'train', cache)
    validation = SequenceDataset(index, 'val', cache)
    manifest = {**train.manifest, 'bcde_index_identity': {
        name: hashlib.sha256((index / name).read_bytes()).hexdigest()
        for name in ('manifest.json', 'routes.json', 'split.json', 'train.npy', 'val.npy')}}
    teacher = TeacherCache(device)
    model = ActionBEVWorldModel(**config['model']).to(device)
    policy = DDP(model, device_ids=[local] if device.type == 'cuda' else None, broadcast_buffers=False) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, seed=seed, drop_last=True)
    batches = BatchSampler(sampler, config['batch_per_gpu'], drop_last=True)
    accumulation = config['accumulation_steps']
    updates_per_epoch = len(batches) // accumulation
    if updates_per_epoch < 1:
        raise ValueError('Training index is too small for a complete optimizer update')
    total = updates_per_epoch * config['epochs']
    warmup = config['warmup_updates']
    def factor(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    progress = {'epoch': 0, 'next_batch': 0, 'updates': 0, 'best_nmse': None, 'validation_pending': False}
    if args.resume:
        progress = load_checkpoint(args.output / 'latest.pt', model, optimizer, scheduler, config, manifest, device)
    if rank == 0:
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (args.output / 'dataset.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (args.output / 'code_sha256.json').write_text(json.dumps({
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob('*.py')}, indent=2) + '\n')
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def finish_validation():
        metrics = validate(validation, model, teacher, config, device)
        score = metrics['model']['nmse']
        best = progress['best_nmse'] is None or score < progress['best_nmse']
        progress.update(validation_pending=False, best_nmse=min(score, progress['best_nmse']) if progress['best_nmse'] is not None else score)
        save_checkpoint(args.output, model, optimizer, scheduler, progress, config, manifest, device, metrics, best, True)
        if rank == 0:
            with (args.output / 'validation.jsonl').open('a') as handle:
                handle.write(json.dumps({'epoch': progress['epoch'], 'best': best, **metrics}) + '\n')
        return metrics

    if progress['validation_pending']:
        finish_validation()
    start_epoch = progress['epoch']
    for epoch in range(start_epoch, config['epochs']):
        sampler.set_epoch(epoch)
        offset = progress['next_batch'] if epoch == start_epoch else 0
        loader = DataLoader(train, batch_sampler=itertools.islice(batches, offset, updates_per_epoch * accumulation),
                            num_workers=config['workers_per_rank'], pin_memory=device.type == 'cuda', collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed + epoch * world + rank + 100003),
                            **({'multiprocessing_context': 'spawn', 'prefetch_factor': config['prefetch_factor']} if config['workers_per_rank'] else {}))
        iterator = iter(loader)
        model.train()
        for number in range(offset // accumulation, updates_per_epoch):
            begin = time.perf_counter()
            prepared = [prepare(next(iterator), teacher, device) for _ in range(accumulation)]
            losses, norm = update(prepared, policy, optimizer, scheduler, config, device)
            del prepared
            progress.update(epoch=epoch, next_batch=(number + 1) * accumulation, updates=progress['updates'] + 1)
            stop = torch.tensor(int(STOP or (args.output / 'STOP').exists()), device=device)
            if distributed():
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if progress['updates'] == 1 or progress['updates'] % config['log_every'] == 0:
                values = torch.stack([losses[name] for name in losses])
                if distributed():
                    dist.all_reduce(values)
                if rank == 0:
                    record = {**progress, **dict(zip(losses, (values / world).cpu().tolist())),
                              'seconds_per_update_rank0': time.perf_counter() - begin,
                              'gradient_norm_rank0': float(norm), 'lr': optimizer.param_groups[0]['lr'],
                              'effective_batch': config['effective_batch']}
                    with (args.output / 'metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps(record) + '\n')
                    print(json.dumps(record), flush=True)
            if progress['updates'] == 1 or progress['updates'] % config['checkpoint_every'] == 0 or bool(stop):
                save_checkpoint(args.output, model, optimizer, scheduler, progress, config, manifest, device)
            if bool(stop):
                if distributed():
                    dist.destroy_process_group()
                return
        progress.update(epoch=epoch + 1, next_batch=0, validation_pending=True)
        save_checkpoint(args.output, model, optimizer, scheduler, progress, config, manifest, device)
        metrics = finish_validation()
        if rank == 0:
            print(json.dumps({'epoch': epoch + 1, 'validation': metrics['model']}), flush=True)
    if rank == 0:
        (args.output / 'completed.json').write_text(json.dumps(progress, indent=2) + '\n')
    if distributed():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
