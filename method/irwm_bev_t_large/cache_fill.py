"""Eight independent GPU shards of the unchanged frozen TF++ BF16 cache contract."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .prepare import ROOT, sha, write_json
from .runtime import configure_worker_tmp


def original_data():
    name = '_wampc_geometric_teacher_cache_reference'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'method/irwm_bev/data.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def identity(manifest):
    return manifest['teacher_sha256'][:16] + '_' + manifest['teacher_config_sha256'][:12] + '_bf16_v1'


def scan(index, cache, output, final=False):
    manifest = json.loads((index / 'manifest.json').read_text())
    routes = json.loads((index / 'routes.json').read_text())
    required = np.load(index / 'required_frames.npy')
    if sha(index / 'required_frames.npy') != manifest['required_frames_sha256']:
        raise ValueError('Required-frame index identity mismatch')
    directory = cache / identity(manifest)
    directory.mkdir(parents=True, exist_ok=True)
    missing = []
    # Exact np.save uint16 64^3 format: 128-byte v1 header plus tensor bytes.
    expected_size = 128 + 64 * 64 * 64 * 2
    offsets = np.searchsorted(required[:, 0], np.arange(len(routes) + 1))
    for rid, route in enumerate(routes):
        folder = directory / route
        existing = {entry.name: entry.stat().st_size for entry in os.scandir(folder)
                    if entry.is_file() and entry.name.endswith('.npy')} if folder.exists() else {}
        for _, frame in required[offsets[rid]:offsets[rid + 1]]:
            filename = f'{int(frame):04}.npy'
            if filename not in existing:
                missing.append((rid, int(frame)))
            elif existing[filename] != expected_size:
                # Accept valid alternate NumPy headers, reject corruption before training.
                bits = np.load(folder / filename, mmap_mode='r', allow_pickle=False)
                if bits.shape != (64, 64, 64) or bits.dtype != np.uint16:
                    raise ValueError('Invalid pre-existing cache file: ' + str(folder / filename))
        if (rid + 1) % 500 == 0:
            print(json.dumps({'event': 'cache_scan', 'routes': rid + 1, 'missing': len(missing)}), flush=True)
    report = {'required_frames': len(required), 'missing_frames': len(missing), 'routes': len(routes),
              'teacher_sha256': manifest['teacher_sha256'],
              'teacher_config_sha256': manifest['teacher_config_sha256'],
              'index_manifest_sha256': sha(index / 'manifest.json'),
              'required_frames_sha256': manifest['required_frames_sha256'],
              'cache_root': str(directory), 'coverage': manifest['cache_coverage'],
              'teacher_reference_sha256': sha(ROOT / 'method/irwm_bev/data.py'),
              'checked_unix': time.time(), 'verification': 'all required frame paths and np.save byte sizes; extractor checks shape/dtype/finite values'}
    output.mkdir(parents=True, exist_ok=True)
    if final:
        if missing:
            raise ValueError(f'Cache still misses {len(missing)} required frames')
        report['completed_unix'] = time.time()
        write_json(directory / 'complete.json', report)
        write_json(output / 'cache_complete.json', report)
    else:
        needed = len(missing) * expected_size
        if shutil.disk_usage(cache).free < needed + (10 << 30):
            raise OSError(f'Insufficient free storage for {needed / 2**30:.1f} GiB cache')
        np.save(output / 'missing.npy', np.asarray(missing, dtype=np.int32).reshape(-1, 2))
        report.update(missing_tasks_sha256=sha(output / 'missing.npy'), expected_new_gib=needed / 2**30)
        write_json(output / 'scan.json', report)
    print(json.dumps({'event': 'cache_verified' if final else 'cache_missing_ready', **report}), flush=True)


class CacheFrames(Dataset):
    def __init__(self, index, cache, tasks, rank, world):
        self.index, self.cache = str(index), str(cache)
        self.tasks = np.load(tasks, mmap_mode='r')[rank::world]
        self.source = None

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, number):
        if self.source is None:
            self.source = original_data().SequenceDataset(self.index, 'train', self.cache)
        rid, frame = map(int, self.tasks[number])
        return self.source.sensor(rid, frame)


def collate_frames(values):
    return [[value] for value in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Cache building must run inside an existing compute allocation')
    configure_worker_tmp()
    index, cache, output = args.index.resolve(), args.cache.resolve(), args.output.resolve()
    index.relative_to(ROOT / 'data')
    cache.relative_to(ROOT / 'data')
    cache.mkdir(parents=True, exist_ok=True)
    if args.scan or args.verify:
        scan(index, cache, output, args.verify)
        return
    local, rank, world = (int(os.environ.get(key, default)) for key, default in
                          [('LOCAL_RANK', '0'), ('RANK', '0'), ('WORLD_SIZE', '1')])
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda', local)
    from method.full_diffusion.garage import FrozenGaragePerception, BACKBONE
    manifest = json.loads((index / 'manifest.json').read_text())
    if sha(BACKBONE / 'model_0030_1.pth') != manifest['teacher_sha256'] or sha(BACKBONE / 'config.json') != manifest['teacher_config_sha256']:
        raise ValueError('Frozen teacher identity changed')
    dataset = CacheFrames(index, cache, output / 'missing.npy', rank, world)
    loader = DataLoader(dataset, batch_size=16, num_workers=args.workers, collate_fn=collate_frames,
                        multiprocessing_context='spawn' if args.workers else None,
                        **({'prefetch_factor': 1} if args.workers else {}), pin_memory=True, timeout=180 if args.workers else 0)
    teacher = original_data().TeacherCache(FrozenGaragePerception().to(device), device, chunk=4)
    start, done = time.monotonic(), 0
    for number, sensors in enumerate(loader):
        teacher(sensors)
        done += len(sensors)
        if number % 20 == 0:
            report = {'rank': rank, 'world_size': world, 'done': done, 'total': len(dataset),
                      'seconds': time.monotonic() - start, 'updated_unix': time.time(), 'gpu': torch.cuda.get_device_name(local)}
            write_json(output / f'shard{rank}.progress.json', report)
            print(json.dumps({'event': 'cache_progress', **report}), flush=True)
    report = {'rank': rank, 'world_size': world, 'frames': done, 'seconds': time.monotonic() - start,
              'tasks_sha256': sha(output / 'missing.npy'), 'completed_unix': time.time()}
    write_json(output / f'shard{rank}.complete.json', report)
    print(json.dumps({'event': 'cache_shard_complete', **report}), flush=True)


if __name__ == '__main__':
    main()
