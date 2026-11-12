"""Prepare measured labels and frozen current-frame context on the existing split."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch

from .bootstrap import ROOT
from .geometry import spatial_ten_to_six
from .world import sha256

PROCESSOR = None


def worker_init(backbone):
    global PROCESSOR
    torch.set_num_threads(1)
    from method.full_diffusion.garage import load_config, make_preprocessor
    PROCESSOR = make_preprocessor(load_config(backbone), training=False)


def prepare_route(task):
    from method.full_diffusion.data import read_measurement, geometric_path
    from method.full_diffusion2.data import frame_pose
    from method.full_diffusionv3.data import future_motion_targets
    import transfuser_utils
    rid, name, source, directory, data_root, anchors, labels = task
    output = Path(directory) / 'labels' / f'{rid:05}.npz'
    with np.load(Path(source) / 'metadata' / f'{rid:05}.npz') as f:
        frames, poses, speeds = f['frames'], f['poses'], f['speed']
    records = defaultdict(list)
    for anchor in anchors:
        frame = int(frames[anchor])
        current = read_measurement(Path(data_root) / name / 'measurements' / f'{frame:04}.json.gz')
        future = {2 * k: {'pos_global': poses[anchor + 2 * k, :2], 'speed': speeds[anchor + 2 * k]}
                  for k in range(1, 7)}
        targets = future_motion_targets(current, future, {k: v for k, v in labels.items() if k != 'spatial_source_indices'})
        spatial = spatial_ten_to_six(torch.from_numpy(geometric_path(PROCESSOR, current))).numpy()
        pose = frame_pose(current)
        observed = []
        for a in (anchor - 2, anchor - 1, anchor):
            x, y, yaw = poses[a]
            c, s = np.cos(yaw), np.sin(yaw)
            observed.append(np.linalg.solve(pose, np.array([[c, -s, x], [s, c, y], [0, 0, 1]])))
        values = {**targets, 'frame': frame, 'history_frames': frames[anchor - 2:anchor + 1],
                  'path': spatial, 'spatial_mask': np.ones(6, bool), 'speed': np.float32(current['speed']),
                  'command': np.asarray(transfuser_utils.command_to_one_hot(current['command']), dtype=np.float32),
                  'target_point': np.asarray(PROCESSOR.augment_target_point(np.asarray(current['target_point']),
                                                                           y_augmentation=0., yaw_augmentation=0.), dtype=np.float32),
                  'current_pose': pose, 'history_transforms': np.asarray(observed, dtype=np.float32),
                  'history_mask': np.ones(3, bool)}
        for key, value in values.items():
            records[key].append(value)
    arrays = {k: np.asarray(v) for k, v in records.items()}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError(f'Nonfinite labels in {name}')
    with output.with_suffix('.partial').open('wb') as handle:
        np.savez(handle, **arrays)
    output.with_suffix('.partial').replace(output)
    target = np.concatenate((arrays['path'], arrays['temporal_delta']), 1).astype(np.float64)
    return rid, len(anchors), target.sum(0), np.square(target).sum(0), sha256(output)


def labels(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Prepare on the allocated compute node')
    config = json.loads(args.config.read_text())
    source = args.source.resolve()
    original = json.loads((source / 'manifest.json').read_text())
    if (args.output / 'manifest.json').exists():
        manifest = json.loads((args.output / 'manifest.json').read_text())
        if manifest['source_sha256'] != {n: sha256(source / n) for n in manifest['source_sha256']}:
            raise ValueError('Prepared index differs from source')
        print(json.dumps({'event': 'reuse_prepared_labels', 'manifest': manifest}), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'labels').mkdir(exist_ok=True)
    routes = json.loads((source / 'routes.json').read_text())
    source_samples = {s: np.load(source / f'{s}.npy') for s in ('train', 'val')}
    groups, parts = defaultdict(set), {}
    for split, samples in source_samples.items():
        for rid, anchor in samples:
            rid = int(rid)
            groups[rid].add(int(anchor))
            if rid in parts and parts[rid] != split:
                raise ValueError('Train and validation route overlap')
            parts[rid] = split
    tasks = [(rid, routes[rid], str(source), str(args.output), config['data_root'], sorted(group), config['labels'])
             for rid, group in sorted(groups.items())]
    total, square, count, identities = np.zeros((12, 2)), np.zeros((12, 2)), 0, {}
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init, initargs=(str(ROOT / config['backbone_directory']),)) as pool:
        for i, (rid, n, sums, squares, digest) in enumerate(pool.map(prepare_route, tasks), 1):
            identities[f'{rid:05}.npz'] = digest
            if parts[rid] == 'train':
                total += sums; square += squares; count += n
            if i % 100 == 0:
                print(json.dumps({'event': 'labels', 'routes': i, 'total': len(tasks),
                                  'seconds': time.perf_counter() - started}), flush=True)
    new_samples = {s: np.asarray([(rid, slot) for rid, anchors in sorted(groups.items()) if parts[rid] == s
                                  for slot in range(len(anchors))], dtype=np.int32) for s in ('train', 'val')}
    for split, samples in new_samples.items():
        np.save(args.output / f'{split}.npy', samples)
    (args.output / 'routes.json').write_text(json.dumps(routes))
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(square / count - mean ** 2, 0)), .1)
    teacher_identity = original['teacher_sha256'][:16] + '_' + original['teacher_config_sha256'][:12] + '_bf16_v1'
    manifest = {'method': 'full_diffusionv4_joint_v1', 'data_root': config['data_root'],
                'backbone_directory': str(ROOT / config['backbone_directory']), 'label_config': config['labels'],
                'teacher_cache': str(ROOT / 'data/irwm_bev/teacher_cache' / teacher_identity),
                'teacher_sha256': original['teacher_sha256'], 'teacher_config_sha256': original['teacher_config_sha256'],
                'source_index': str(source), 'source_sha256': {n: sha256(source / n) for n in
                                                            ('manifest.json', 'routes.json', 'train.npy', 'val.npy', 'split.json')},
                'train_samples': len(new_samples['train']), 'val_samples': len(new_samples['val']),
                'train_routes': sum(p == 'train' for p in parts.values()), 'validation_routes': sum(p == 'val' for p in parts.values()),
                'labels_sha256': identities, 'prepared_index_sha256': {n: sha256(args.output / n) for n in
                                                                     ('routes.json', 'train.npy', 'val.npy')},
                'normalization': {'mean': mean.tolist(), 'std': std.tolist()},
                'normalization_source': 'all retained unaugmented six-point training labels only',
                'future_input_policy': 'GT future positions/speeds create labels only; observed cache frames only are loaded at training'}
    manifest['identity'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'event': 'labels_complete', 'train': manifest['train_samples'], 'val': manifest['val_samples'],
                      'identity': manifest['identity'], 'seconds': time.perf_counter() - started}), flush=True)


def contexts(args):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from method.full_diffusion2.perception import FrozenPlannerPerception
    from .joint_data import JointDataset
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Fill frozen contexts only inside the allocation')
    rank, world, local = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl')
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda', local)
    dataset = JointDataset(args.output, sensors=True)
    val = JointDataset(args.output, split='val', sensors=True)
    target = args.output / 'context'
    target.mkdir(exist_ok=True)
    perception = FrozenPlannerPerception(dataset.manifest['backbone_directory']).to(device).eval()
    counts = 0
    for data in (dataset, val):
        grouped = defaultdict(list)
        for i, (rid, _) in enumerate(data.samples):
            grouped[int(rid)].append(i)
        selected = [indices for rid, indices in sorted(grouped.items()) if rid % world == rank and not (target / f'{rid:05}.pt').exists()]
        # Group several routes per loader to amortize worker startup.
        for start in range(0, len(selected), 64):
            indices = [i for group in selected[start:start + 64] for i in group]
            loader = DataLoader(Subset(data, indices), batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=True, **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if args.workers else {}))
            collected = defaultdict(list)
            with torch.inference_mode():
                for batch in loader:
                    sensor = {k: batch[k].to(device, non_blocking=True) for k in ('rgb', 'lidar', 'speed', 'command', 'target_point')}
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        _, memory, ego = perception(sensor)
                    for i, (rid, slot) in enumerate(zip(batch['route_id'].tolist(), batch['slot'].tolist())):
                        collected[rid].append((slot, memory[i].bfloat16().cpu(), ego[i].bfloat16().cpu()))
                    counts += len(batch['rgb'])
            for rid, records in collected.items():
                ordered = sorted(records)
                if [x[0] for x in ordered] != list(range(len(ordered))):
                    raise ValueError('Incomplete or duplicated frozen context slots')
                state = {'memory': torch.stack([x[1] for x in ordered]), 'ego': torch.stack([x[2] for x in ordered]),
                         'dataset_sha256': data.manifest['identity']}
                temporary = target / f'{rid:05}.partial'
                torch.save(state, temporary)
                temporary.replace(target / f'{rid:05}.pt')
            print(json.dumps({'event': 'contexts', 'rank': rank, 'new_frames': counts, 'route_chunks': start // 64 + 1}), flush=True)
    if world > 1:
        dist.barrier(device_ids=[local])
    if rank == 0:
        routes = set(int(x) for data in (dataset, val) for x in data.samples[:, 0])
        if any(not (target / f'{r:05}.pt').is_file() for r in routes):
            raise ValueError('Frozen context cache is incomplete')
        complete = {'dataset_sha256': dataset.manifest['identity'], 'teacher_sha256': dataset.manifest['teacher_sha256'],
                    'routes': len(routes), 'frames': len(dataset) + len(val), 'precision': 'bf16',
                    'source': 'frozen TF++ current observation memory and ego; no future sensor inputs'}
        (target / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
        print(json.dumps({'event': 'contexts_complete', **complete}), flush=True)
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['labels', 'contexts'])
    parser.add_argument('--source', type=Path, default=ROOT / 'data/irwm_bev_t/pdm_seed901_dev44_dt05_tongji_v1')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    (labels if args.stage == 'labels' else contexts)(args)


if __name__ == '__main__':
    main()
