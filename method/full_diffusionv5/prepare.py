"""Prepare single-anchor 4Hz labels, missing teacher features and current contexts.

All eligible anchors remain in the index, including anchors whose teacher
features still need filling. Missing caches must be completed before training.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from method.full_diffusionv4.bootstrap import ROOT
from method.full_diffusionv4.world import sha256
from .contract import CONTRACT, INDEX_FORMAT, validate_config, validate_manifest

HISTORY, FUTURE = 2, 12
PROCESSOR = None


def require_allocation():
    if not (os.environ.get('SLURM_JOB_ID') or os.environ.get('LSB_JOBID')):
        raise RuntimeError('Prepare inside an existing compute allocation')


def worker_init(backbone):
    global PROCESSOR
    torch.set_num_threads(1)
    from method.full_diffusion.garage import load_config, make_preprocessor
    PROCESSOR = make_preprocessor(load_config(backbone), training=False)


def anchor_indices(frames, skip_first):
    good = np.zeros(len(frames), dtype=bool)
    width = HISTORY + FUTURE + 1
    if len(frames) >= width:
        windows = np.lib.stride_tricks.sliding_window_view(frames, width)
        good[HISTORY:len(frames) - FUTURE] = (np.diff(windows, axis=1) == 1).all(1)
    return np.flatnonzero(good & (frames >= skip_first))


def prepare_route(task):
    from method.full_diffusion.data import geometric_path, read_measurement
    from method.full_diffusion2.data import frame_pose
    from method.full_diffusionv3.data import future_motion_targets
    import transfuser_utils
    rid, name, source, directory, data_root, teacher_root, skip_first, labels = task
    with np.load(Path(source) / 'metadata' / f'{rid:05}.npz') as f:
        frames, poses, speeds = f['frames'].astype(np.int64), f['poses'], f['speed']
    cache = Path(teacher_root) / name
    existing = ({int(p.name[:-4]) for p in os.scandir(cache) if p.name.endswith('.npy')}
                if cache.is_dir() else set())
    anchors = anchor_indices(frames, skip_first)
    records = defaultdict(list)
    for anchor in anchors:
        frame = int(frames[anchor])
        current = read_measurement(Path(data_root) / name / 'measurements' / f'{frame:04}.json.gz')
        future = {2 * k: {'pos_global': poses[anchor + 2 * k, :2], 'speed': speeds[anchor + 2 * k]}
                  for k in range(1, 7)}
        targets = future_motion_targets(current, future, labels)
        pose = frame_pose(current)
        observed = []
        for a in (anchor - 2, anchor - 1, anchor):
            x, y, yaw = poses[a]
            c, s = np.cos(yaw), np.sin(yaw)
            observed.append(np.linalg.solve(pose, np.array([[c, -s, x], [s, c, y], [0, 0, 1]])))
        values = {**targets, 'frame': frame, 'history_frames': frames[anchor - 2:anchor + 1],
                  'path': geometric_path(PROCESSOR, current), 'spatial_mask': np.ones(10, bool),
                  'speed': np.float32(current['speed']),
                  'command': np.asarray(transfuser_utils.command_to_one_hot(current['command']), dtype=np.float32),
                  'target_point': np.asarray(PROCESSOR.augment_target_point(
                      np.asarray(current['target_point']), y_augmentation=0., yaw_augmentation=0.), dtype=np.float32),
                  'current_pose': pose, 'history_transforms': np.asarray(observed, dtype=np.float32),
                  'history_mask': np.ones(3, bool)}
        for key, value in values.items():
            records[key].append(value)
    items, missing = [], 0
    for s, a in enumerate(anchors):
        f = int(frames[a])
        items.append((rid, s, f))
        missing += int(not all(x in existing for x in range(f - 2, f + 1)))
    output = Path(directory) / 'labels' / f'{rid:05}.npz'
    if records:
        arrays = {k: np.asarray(v) for k, v in records.items()}
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError(f'Nonfinite labels in {name}')
        with output.with_suffix('.partial').open('wb') as handle:
            np.savez(handle, **arrays)
        output.with_suffix('.partial').replace(output)
    digest = sha256(output) if records else None
    return rid, len(anchors), np.asarray(items, dtype=np.int64).reshape(-1, 3), missing, digest


def labels(args):
    require_allocation()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if sha256(ROOT / config['v3']['checkpoint']) != config['v3']['sha256']:
        raise ValueError('V3 checkpoint checksum differs')
    v3 = torch.load(ROOT / config['v3']['checkpoint'], map_location='cpu', weights_only=False)
    if v3['config']['labels'] != config['labels']:
        raise ValueError('V5 labels must be exactly the frozen V3 labels')
    del v3
    source = (ROOT / config['source_index']).resolve()
    output = (ROOT / config['index_directory']).resolve()
    original = json.loads((source / 'manifest.json').read_text())
    if original.get('frame_dt_s') != .25:
        raise ValueError('Source observations must be recorded at 4Hz')
    if output.exists():
        raise FileExistsError(f'Use a new V5 index directory: {output}')
    (output / 'labels').mkdir(parents=True)
    routes = json.loads((source / 'routes.json').read_text())
    parts = {}
    for split in ('train', 'val'):
        for rid in np.unique(np.load(source / f'{split}.npy')[:, 0]):
            if parts.setdefault(int(rid), split) != split:
                raise ValueError('Train and validation route overlap')
    teacher_root = (ROOT / 'data/irwm_bev/teacher_cache' / (
        original['teacher_sha256'][:16] + '_' + original['teacher_config_sha256'][:12] + '_bf16_v1'))
    tasks = [(rid, routes[rid], str(source), str(output), config['data_root'], str(teacher_root),
              int(original['skip_first']), config['labels']) for rid in sorted(parts)]
    items = {'train': [], 'val': []}
    anchors = {'train': 0, 'val': 0}
    missing = {'train': 0, 'val': 0}
    identities = {}
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init, initargs=(str(ROOT / config['backbone_directory']),)) as pool:
        for i, (rid, count, route_items, absent, digest) in enumerate(pool.map(prepare_route, tasks), 1):
            split = parts[rid]
            anchors[split] += count
            missing[split] += absent
            items[split].append(route_items)
            if digest:
                identities[f'{rid:05}.npz'] = digest
            if i % 100 == 0:
                print(json.dumps({'event': 'labels', 'routes': i, 'total': len(tasks),
                                  'seconds': time.perf_counter() - started}), flush=True)
    for split, values in items.items():
        np.save(output / f'{split}.npy', np.concatenate(values).astype(np.int32))
    (output / 'routes.json').write_text(json.dumps(routes))
    counts = {split: len(np.load(output / f'{split}.npy')) for split in items}
    manifest = {'method': INDEX_FORMAT, 'training_contract': CONTRACT, 'data_root': config['data_root'],
                'backbone_directory': str(ROOT / config['backbone_directory']), 'label_config': config['labels'],
                'teacher_cache': str(teacher_root), 'teacher_sha256': original['teacher_sha256'],
                'teacher_config_sha256': original['teacher_config_sha256'],
                'source_index': str(source), 'source_sha256': {n: sha256(source / n) for n in
                                                            ('manifest.json', 'routes.json', 'train.npy', 'val.npy')},
                'cadence': {'recorded_hz': 4, 'anchor_stride_frames': 1, 'chain_stride_frames': 1,
                            'item_frames': [0], 'history_frame_offsets': [-2, -1, 0],
                            'cache_ages_seconds': [0.0], 'prior_age_seconds': .25},
                'anchors': anchors, 'items': counts, 'items_missing_teacher_features': missing,
                'routes': {split: sum(s == split for s in parts.values()) for split in ('train', 'val')},
                'labels_sha256': identities,
                'prepared_index_sha256': {n: sha256(output / n) for n in ('routes.json', 'train.npy', 'val.npy')},
                'future_input_policy': 'future poses/speeds create labels only; observed frames only are inputs'}
    manifest['identity'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'event': 'labels_complete', 'anchors': anchors, 'items': counts,
                      'items_missing_teacher_features': missing, 'identity': manifest['identity'],
                      'seconds': time.perf_counter() - started}), flush=True)


class AnchorSensors(Dataset):
    def __init__(self, directory, manifest, tasks):
        from method.full_diffusion.data import GarageGeoDataset
        self.loader = GarageGeoDataset(directory, data_root=manifest['data_root'],
                                       backbone_directory=manifest['backbone_directory'], training=False)
        self.root = Path(manifest['data_root'])
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        rid, slot, frame = self.tasks[index]
        current = self.loader.load_frame(self.root / self.loader.routes[rid], frame, random.Random(901))
        return {k: current[k] for k in ('rgb', 'lidar', 'speed', 'command', 'target_point')} | {
            'route_id': rid, 'slot': slot, 'frame': frame}


def verify_teacher(manifest):
    backbone = Path(manifest['backbone_directory'])
    if (sha256(backbone / 'model_0030_1.pth') != manifest['teacher_sha256']
            or sha256(backbone / 'config.json') != manifest['teacher_config_sha256']):
        raise ValueError('Frozen perception weights/config differ from the existing teacher cache')


def context_source(config, manifest):
    """Only reuse contexts with the same teacher, route numbering and index identity."""
    name = config.get('context_source_index')
    if not name:
        return None
    source = ROOT / name
    old = json.loads((source / 'manifest.json').read_text())
    marker = json.loads((source / 'context/complete.json').read_text())
    if (old['teacher_sha256'] != manifest['teacher_sha256']
            or old['teacher_config_sha256'] != manifest['teacher_config_sha256']
            or marker['dataset_sha256'] != old['identity']):
        raise ValueError('Source current-context cache has a different teacher or index identity')
    routes = json.loads((source / 'routes.json').read_text())
    expected = json.loads((ROOT / config['index_directory'] / 'routes.json').read_text())
    if routes != expected:
        raise ValueError('Source context route numbering differs')
    return source, old['identity']


def reusable_context(source, rid):
    if source is None:
        return None
    directory, identity = source
    path = directory / 'context' / f'{rid:05}.pt'
    if not path.is_file():
        return None
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['dataset_sha256'] != identity:
        raise ValueError(f'Stale source context: {path}')
    with np.load(directory / 'labels' / f'{rid:05}.npz') as f:
        frame_ids = f['frame'].copy()
    if len(state['memory']) != len(frame_ids) or len(state['ego']) != len(frame_ids):
        raise ValueError(f'Source context frame count differs: {path}')
    return {**state, 'frames': frame_ids}


def missing_features(directory, manifest, route_ids, verify=True):
    """Return only absent observations; cache availability never changes the index."""
    routes = json.loads((directory / 'routes.json').read_text())
    tasks, required = [], 0
    for rid in route_ids:
        with np.load(directory / 'labels' / f'{rid:05}.npz') as f:
            needed = sorted(set(map(int, f['history_frames'].ravel())))
        required += len(needed)
        folder = Path(manifest['teacher_cache']) / routes[rid]
        existing = ({int(p.name[:-4]) for p in os.scandir(folder) if p.name.endswith('.npy')}
                    if folder.is_dir() else set())
        for frame_id in needed:
            if frame_id not in existing:
                tasks.append((rid, frame_id, frame_id))
            elif verify:
                path = folder / f'{frame_id:04}.npy'
                feature = np.load(path, mmap_mode='r', allow_pickle=False)
                if feature.dtype != np.uint16 or feature.shape != (64, 64, 64):
                    raise ValueError(f'Existing teacher feature is corrupt: {path}')
                del feature
    return tasks, required


def features(args):
    """Fill the union of observed history frames; never use the old LSF-only cache launcher."""
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from method.full_diffusion2.perception import FrozenPlannerPerception
    require_allocation()
    config = json.loads(args.config.read_text())
    validate_config(config)
    directory = ROOT / config['index_directory']
    manifest = json.loads((directory / 'manifest.json').read_text())
    validate_manifest(manifest, config)
    verify_teacher(manifest)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl')
    device = torch.device('cuda', local)
    routes = sorted({int(r) for split in ('train', 'val') for r in np.load(directory / f'{split}.npy')[:, 0]})
    selected = [r for r in routes if r % world == rank]
    tasks, required = missing_features(directory, manifest, selected)
    target = directory / 'features'
    target.mkdir(exist_ok=True)
    print(json.dumps({'event': 'feature_scan', 'rank': rank, 'required': required, 'missing': len(tasks)}), flush=True)
    if tasks:
        perception = FrozenPlannerPerception(manifest['backbone_directory']).to(device).eval()
        loader = DataLoader(AnchorSensors(directory, manifest, tasks), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=True,
                            **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if args.workers else {}))
        route_names = json.loads((directory / 'routes.json').read_text())
        with torch.inference_mode():
            for batch in loader:
                sensor = {k: batch[k].to(device, non_blocking=True)
                          for k in ('rgb', 'lidar', 'speed', 'command', 'target_point')}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    bev, _, _ = perception(sensor)
                if bev.shape[1:] != (64, 64, 64) or not bool(torch.isfinite(bev).all()):
                    raise ValueError('Invalid newly extracted teacher BEV')
                for i, (rid, frame_id) in enumerate(zip(batch['route_id'].tolist(), batch['frame'].tolist())):
                    path = Path(manifest['teacher_cache']) / route_names[rid] / f'{frame_id:04}.npy'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(f'.{rank}.{os.getpid()}.partial')
                    with temporary.open('wb') as handle:
                        np.save(handle, bev[i].bfloat16().cpu().contiguous().view(torch.uint16).numpy(), allow_pickle=False)
                    temporary.replace(path)
    remaining, _ = missing_features(directory, manifest, selected, verify=False)
    if remaining:
        raise ValueError('Teacher feature filling is incomplete')
    (target / f'rank{rank}.json').write_text(json.dumps({
        'dataset_sha256': manifest['identity'], 'required': required, 'filled': len(tasks)}))
    if world > 1:
        dist.barrier(device_ids=[local])
    if rank == 0:
        states = [json.loads((target / f'rank{r}.json').read_text()) for r in range(world)]
        if any(s['dataset_sha256'] != manifest['identity'] for s in states):
            raise ValueError('Feature shard belongs to another index')
        (target / 'complete.json').write_text(json.dumps({
            'dataset_sha256': manifest['identity'], 'required_frames': sum(s['required'] for s in states),
            'filled_frames': sum(s['filled'] for s in states), 'precision': 'bf16',
            'verification': 'all required files present; pre-existing NPY headers validated; newly extracted BEVs finite'}, indent=2) + '\n')
    if world > 1:
        dist.destroy_process_group()


def contexts(args):
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from method.full_diffusion2.perception import FrozenPlannerPerception
    require_allocation()
    config = json.loads(args.config.read_text())
    validate_config(config)
    directory = ROOT / config['index_directory']
    manifest = json.loads((directory / 'manifest.json').read_text())
    validate_manifest(manifest, config)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl')
    device = torch.device('cuda', local)
    target = directory / 'context'
    target.mkdir(exist_ok=True)
    routes = sorted({int(r) for split in ('train', 'val') for r in np.load(directory / f'{split}.npy')[:, 0]})
    selected = [r for r in routes if r % world == rank and not (target / f'{r:05}.pt').exists()]
    verify_teacher(manifest)
    source = context_source(config, manifest)
    perception = None
    frames = 0
    reused = 0
    for start in range(0, len(selected), 64):
        tasks = []
        collected = defaultdict(list)
        for rid in selected[start:start + 64]:
            with np.load(directory / 'labels' / f'{rid:05}.npz') as f:
                frame_ids = f['frame'].copy()
            old = reusable_context(source, rid)
            old_slots = {} if old is None else {int(f): s for s, f in enumerate(old['frames'])}
            for slot, frame_id in enumerate(frame_ids):
                if int(frame_id) in old_slots:
                    s = old_slots[int(frame_id)]
                    collected[rid].append((slot, old['memory'][s].clone(), old['ego'][s].clone()))
                    reused += 1
                else:
                    tasks.append((rid, slot, int(frame_id)))
            del old
        if tasks and perception is None:
            perception = FrozenPlannerPerception(manifest['backbone_directory']).to(device).eval()
        loader = DataLoader(AnchorSensors(directory, manifest, tasks), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=True,
                            **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if args.workers else {}))
        with torch.inference_mode():
            for batch in loader:
                sensor = {k: batch[k].to(device, non_blocking=True)
                          for k in ('rgb', 'lidar', 'speed', 'command', 'target_point')}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, memory, ego = perception(sensor)
                for i, (rid, slot) in enumerate(zip(batch['route_id'].tolist(), batch['slot'].tolist())):
                    collected[rid].append((slot, memory[i].bfloat16().cpu(), ego[i].bfloat16().cpu()))
                frames += len(batch['rgb'])
        for rid, records in collected.items():
            ordered = sorted(records, key=lambda x: x[0])
            with np.load(directory / 'labels' / f'{rid:05}.npz') as f:
                expected = len(f['frame'])
            if [x[0] for x in ordered] != list(range(expected)):
                raise ValueError('Incomplete or duplicated context slots')
            state = {'memory': torch.stack([x[1].bfloat16() for x in ordered]),
                     'ego': torch.stack([x[2].bfloat16() for x in ordered]),
                     'dataset_sha256': manifest['identity']}
            torch.save(state, target / f'{rid:05}.partial')
            (target / f'{rid:05}.partial').replace(target / f'{rid:05}.pt')
        print(json.dumps({'event': 'contexts', 'rank': rank, 'frames_extracted': frames, 'frames_reused': reused,
                          'routes_done': min(start + 64, len(selected)), 'routes': len(selected)}), flush=True)
    if world > 1:
        dist.barrier(device_ids=[local])
    if rank == 0:
        if any(not (target / f'{r:05}.pt').is_file() for r in routes):
            raise ValueError('Context cache is incomplete')
        (target / 'complete.json').write_text(json.dumps({
            'dataset_sha256': manifest['identity'], 'routes': len(routes), 'precision': 'bf16',
            'source': 'frozen TF++ current-frame memory/ego of every 4Hz anchor'}, indent=2) + '\n')
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['labels', 'features', 'contexts'])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config_train4hz.json'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    {'labels': labels, 'features': features, 'contexts': contexts}[args.stage](args)


if __name__ == '__main__':
    main()

