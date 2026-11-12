"""Prepared six-point labels, immutable observed features and causal DDP streams."""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class JointDataset(Dataset):
    def __init__(self, directory, split='train', sensors=False):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.routes = json.loads((self.directory / 'routes.json').read_text())
        self.samples = np.load(self.directory / f'{split}.npy', mmap_mode='r')
        self.sensors = sensors
        if sensors:
            from method.full_diffusion.data import GarageGeoDataset
            self.loader = GarageGeoDataset(self.directory, data_root=self.manifest['data_root'],
                                           backbone_directory=self.manifest['backbone_directory'], training=False)
        else:
            context = json.loads((self.directory / 'context/complete.json').read_text())
            if context['dataset_sha256'] != self.manifest['identity']:
                raise ValueError('Frozen planner context cache has a different dataset identity')
        self.teacher_root = Path(self.manifest['teacher_cache'])

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=256)
    def metadata(self, route):
        with np.load(self.directory / 'labels' / f'{route:05}.npz') as f:
            return {k: f[k].copy() for k in f.files}

    @lru_cache(maxsize=256)
    def context(self, route):
        state = torch.load(self.directory / 'context' / f'{route:05}.pt', map_location='cpu', weights_only=True)
        if state['dataset_sha256'] != self.manifest['identity']:
            raise ValueError('Route context has a different dataset identity')
        return state

    @lru_cache(maxsize=512)
    def feature(self, route, frame):
        path = self.teacher_root / self.routes[route] / f'{frame:04}.npy'
        bits = np.load(path, allow_pickle=False)
        if bits.dtype != np.uint16 or bits.shape != (64, 64, 64):
            raise ValueError(f'Invalid observed teacher feature: {path}')
        return torch.from_numpy(bits).view(torch.bfloat16)

    def __getitem__(self, index):
        route, slot = map(int, self.samples[index])
        m = self.metadata(route)
        result = {k: torch.from_numpy(v[slot].copy()) if v[slot].ndim else torch.as_tensor(v[slot].copy())
                  for k, v in m.items() if k not in ('frame', 'history_frames')}
        result.update(scene_id=self.routes[route], timestamp=float(m['frame'][slot]) * .25,
                      sample_index=index, route_id=route, slot=slot)
        if self.sensors:
            current = self.loader.load_frame(Path(self.manifest['data_root']) / self.routes[route],
                                             int(m['frame'][slot]), random.Random(901))
            result.update(rgb=current['rgb'], lidar=current['lidar'])
        else:
            result['raw_bev'] = torch.stack([self.feature(route, int(f)) for f in m['history_frames'][slot]])
            context = self.context(route)
            result['memory'], result['ego'] = context['memory'][slot], context['ego'][slot]
        return result


class DistributedRouteStreams:
    """Fixed disjoint route pools per stream; commit only consumed batches.

    Like the reference's SceneStreamBatchSampler, epochs are sample budgets and
    route streams continue across epochs. Prefetch never advances checkpoint state.
    """
    def __init__(self, samples, batch_size, rank=0, world_size=1, seed=901):
        groups = defaultdict(list)
        for index, (route, slot) in enumerate(samples):
            groups[int(route)].append((int(slot), index))
        self.scenes = [sorted(group) for _, group in sorted(groups.items())]
        self.batch_size, self.rank, self.world_size, self.seed = batch_size, rank, world_size, seed
        if len(self.scenes) < batch_size * world_size:
            raise ValueError('Need one distinct route pool per global stream')
        self.consumed, self.committed, self.pending = 0, None, {}

    def initial(self):
        order = list(range(len(self.scenes)))
        random.Random(self.seed).shuffle(order)
        total = self.batch_size * self.world_size
        return [{'pool': order[(self.rank * self.batch_size + i)::total], 'route': 0, 'frame': 0,
                 'cycle': 0, 'global_slot': self.rank * self.batch_size + i} for i in range(self.batch_size)]

    def __iter__(self):
        streams, cursor = deepcopy(self.committed if self.committed is not None else self.initial()), self.consumed
        while True:
            batch = []
            for stream in streams:
                if stream['route'] == len(stream['pool']):
                    stream['cycle'] += 1
                    random.Random(self.seed + 100003 * stream['cycle'] + stream['global_slot']).shuffle(stream['pool'])
                    stream['route'] = stream['frame'] = 0
                sequence = self.scenes[stream['pool'][stream['route']]]
                batch.append(sequence[stream['frame']][1])
                stream['frame'] += 1
                if stream['frame'] == len(sequence):
                    stream['route'] += 1
                    stream['frame'] = 0
            cursor += 1
            self.pending[cursor] = deepcopy(streams)
            yield batch

    def commit(self):
        self.consumed += 1
        self.committed = self.pending.pop(self.consumed)

    def state_dict(self):
        return {'consumed': self.consumed, 'streams': deepcopy(self.committed),
                'identity': [self.batch_size, self.rank, self.world_size, self.seed]}

    def load_state_dict(self, value):
        if value['identity'] != [self.batch_size, self.rank, self.world_size, self.seed]:
            raise ValueError('Stream resume requires identical batch, rank, world size and seed')
        self.consumed, self.committed = value['consumed'], deepcopy(value['streams'])
        self.pending = {}
