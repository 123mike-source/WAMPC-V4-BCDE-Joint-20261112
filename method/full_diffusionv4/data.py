"""Six geometric labels, causal pose metadata and chronological route streams."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Sampler

from method.full_diffusionv3.data import GarageSpatialTemporalDataset as V3Dataset
from method.full_diffusion2.data import frame_pose
from .geometry import SPATIAL_SOURCE_INDICES, spatial_ten_to_six


class GarageWorldDataset(V3Dataset):
    def __init__(self, *args, label_config, **kwargs):
        settings = dict(label_config)
        indices = tuple(settings.pop('spatial_source_indices', SPATIAL_SOURCE_INDICES))
        if indices != SPATIAL_SOURCE_INDICES:
            raise ValueError('Six-point geometry differs from the V4 codec/controller contract')
        # The pretrained world model learned unaugmented raw frozen-teacher maps.
        kwargs['training'] = False
        super().__init__(*args, label_config=settings, **kwargs)
        self.processor.config.augment = False

    def __getitem__(self, index):
        result = super().__getitem__(index)
        rid, frame = map(int, self.samples[index])
        route = self.data_root / self.routes[rid]
        measurement = self.read_measurement_file(route / 'measurements' / f'{frame:04}.json.gz')
        result['path'] = spatial_ten_to_six(result['path'])
        result['spatial_mask'] = torch.ones(6, dtype=torch.bool)
        result['scene_id'] = self.routes[rid]
        result['current_pose'] = torch.from_numpy(frame_pose(measurement)).double()
        result['timestamp'] = frame * self.label_config['raw_measurement_interval_seconds']
        result['sample_index'] = int(index)
        return result


class RouteStreamBatchSampler(Sampler):
    """At most one sample from each route per batch, chronological within routes."""
    def __init__(self, samples, batch_size, seed=901, epoch=0):
        groups = defaultdict(list)
        for i, (route, frame) in enumerate(samples):
            groups[int(route)].append((int(frame), i))
        routes = sorted(groups)
        random.Random(seed + epoch).shuffle(routes)
        queue = [iter([i for _, i in sorted(groups[route])]) for route in routes]
        self.batches = []
        active, cursor = [], 0
        while active or cursor < len(queue):
            while len(active) < batch_size and cursor < len(queue):
                active.append(queue[cursor])
                cursor += 1
            batch, retained = [], []
            for stream in active:
                try:
                    batch.append(next(stream))
                    retained.append(stream)
                except StopIteration:
                    pass
            active = retained
            if batch:
                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def validate_normalization_artifact(artifact, config, normalization):
    """Compare measured moments at the codec's stored float32 precision."""
    if (artifact['method'] != 'full_diffusionv4' or artifact['label_config'] != config['labels']
            or Path(artifact['data_root']).resolve() != Path(config['data_root']).resolve()):
        raise ValueError('V4 normalization labels or data root differ from this training run')
    for name in ('mean', 'std'):
        measured = torch.as_tensor(artifact['normalization'][name], dtype=torch.float32)
        stored = torch.as_tensor(normalization[name], dtype=torch.float32)
        if (measured.shape != (12, 2) or not bool(torch.isfinite(measured).all())
                or not torch.equal(measured, stored)
                or (name == 'std' and not bool((measured > 0).all()))):
            raise ValueError('Prepare actual six-point training moments, then bootstrap using that artifact')


def prepare_normalization(index, backbone, labels, output, data_root=None):
    """Measure new six-point target moments; do not interpolate old variances."""
    from method.full_diffusion.data import geometric_path, read_measurement
    from method.full_diffusion.garage import load_config, make_preprocessor
    from method.full_diffusionv3.data import future_motion_targets, raw_offset
    index, output = Path(index), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((index / 'manifest.json').read_text())
    routes = json.loads((index / 'routes.json').read_text())
    samples = np.load(index / 'train.npy', mmap_mode='r')
    processor = make_preprocessor(load_config(backbone), training=False)
    settings = {k: v for k, v in labels.items() if k != 'spatial_source_indices'}
    sums, squares, count = np.zeros((12, 2)), np.zeros((12, 2)), np.zeros(12, np.int64)
    data_root = Path(data_root or manifest['data_root']).resolve()
    for rid, frame in samples:
        route = data_root / routes[int(rid)] / 'measurements'
        current = read_measurement(route / f'{frame:04}.json.gz')
        spatial = spatial_ten_to_six(torch.from_numpy(geometric_path(processor, current))).numpy()
        future = {}
        for k in range(1, 7):
            offset = raw_offset(.5 * k, settings)
            path = route / f'{int(frame) + offset:04}.json.gz'
            future[offset] = read_measurement(path) if path.is_file() else None
        targets = future_motion_targets(current, future, settings)
        values = np.concatenate((spatial, targets['temporal_delta'])).astype(np.float64)
        valid = np.r_[np.ones(6, bool), targets['temporal_mask']]
        sums[valid] += values[valid]
        squares[valid] += values[valid] ** 2
        count += valid
    if (count <= 0).any():
        raise ValueError('Some new target slots have no valid training labels')
    mean = sums / count[:, None]
    std = np.maximum(np.sqrt(np.maximum(squares / count[:, None] - mean ** 2, 0)), .1)
    artifact = {'method': 'full_diffusionv4', 'label_config': labels, 'train_frames': len(samples),
                'data_root': str(data_root),
                'valid_slot_counts': count.tolist(), 'normalization': {'mean': mean.tolist(), 'std': std.tolist()},
                'source_sha256': {name: hashlib.sha256((index / name).read_bytes()).hexdigest()
                                  for name in ('manifest.json', 'routes.json', 'train.npy', 'val.npy')}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + '\n')
    return artifact
