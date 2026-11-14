"""One current anchor per item, with BEV history at t-0.5, t-0.25 and t."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .contract import validate_manifest

FRAME_KEYS = ('path', 'spatial_mask', 'temporal_delta', 'temporal_mask', 'future_positions',
              'behavior_targets', 'behavior_mask', 'speed', 'command', 'target_point',
              'current_pose', 'history_transforms', 'history_mask')


class V5Dataset(Dataset):
    def __init__(self, directory, split='train'):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        validate_manifest(self.manifest)
        self.routes = json.loads((self.directory / 'routes.json').read_text())
        self.samples = np.load(self.directory / f'{split}.npy')
        if self.samples.ndim != 2 or self.samples.shape[1] != 3:
            raise ValueError('Expected (route, label slot, current frame) rows')
        for marker in ('features/complete.json', 'context/complete.json'):
            state = json.loads((self.directory / marker).read_text())
            if state['dataset_sha256'] != self.manifest['identity']:
                raise ValueError(f'Cache belongs to a different 4Hz index: {marker}')
        self.teacher_root = Path(self.manifest['teacher_cache'])

    def __len__(self):
        return len(self.samples)

    def chains(self):
        """One chronological chain per route; both frame parities belong to it."""
        return self.samples[:, [0, 2]].astype(np.int64)

    @lru_cache(maxsize=64)
    def metadata(self, route):
        with np.load(self.directory / 'labels' / f'{route:05}.npz') as f:
            return {k: f[k].copy() for k in f.files}

    @lru_cache(maxsize=64)
    def context(self, route):
        state = torch.load(self.directory / 'context' / f'{route:05}.pt', map_location='cpu', weights_only=True)
        if state['dataset_sha256'] != self.manifest['identity']:
            raise ValueError('Route context belongs to a different 4Hz index')
        return state

    @lru_cache(maxsize=96)
    def feature(self, route, frame):
        path = self.teacher_root / self.routes[route] / f'{frame:04}.npy'
        bits = np.load(path, allow_pickle=False)
        if bits.dtype != np.uint16 or bits.shape != (64, 64, 64):
            raise ValueError(f'Invalid observed teacher feature: {path}')
        return torch.from_numpy(bits).view(torch.bfloat16)

    def __getitem__(self, index):
        route, slot, frame = map(int, self.samples[index])
        m = self.metadata(route)
        if int(m['frame'][slot]) != frame or list(m['history_frames'][slot]) != [frame - 2, frame - 1, frame]:
            raise ValueError(f'Item {index} has a mismatched current frame or observation history')
        result = {k: torch.from_numpy(np.asarray(m[k][slot]).copy()) for k in FRAME_KEYS}
        result['raw_bev'] = torch.stack([self.feature(route, f) for f in (frame - 2, frame - 1, frame)])
        context = self.context(route)
        result['memory'], result['ego'] = context['memory'][slot], context['ego'][slot]
        result.update(chain_id=self.routes[route], timestamp=frame * .25,
                      route_id=route, sample_index=index)
        return result
