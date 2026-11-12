"""Reuse the original immutable cache/index and isolate future target construction."""
import torch

from .legacy import module

_source = module('data')


class SequenceDataset(_source.SequenceDataset):
    """Physical import path keeps spawn-worker pickling independent of aliases."""


class TeacherCache(_source.TeacherCache):
    pass


def collate(samples):
    return _source.collate(samples)


align_to_anchor = module('geometry').align_to_anchor
path_corridor = module('geometry').path_corridor


@torch.no_grad()
def prepare(batch, teacher, device):
    features = teacher(batch['sensors'])
    batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
             for key, value in batch.items() if key != 'sensors'}
    b, n, c, h, w = features.shape
    aligned, valid = align_to_anchor(features.flatten(0, 1), batch['source_in_anchor'].flatten(0, 1))
    aligned = aligned.reshape(b, n, c, h, w)
    valid = valid.reshape(b, n, 1, h, w)
    inputs = {'history': aligned[:, :3], 'history_valid': valid[:, :3],
              'temporal_delta': batch['temporal_delta'], 'speed': batch['speed']}
    supervision = {'target': aligned[:, 3:], 'spatial_valid': valid[:, 3:],
                   'query_valid': batch['query_valid'],
                   'corridor': path_corridor(batch['path'], batch['path_valid'], batch['query_xy'], batch['query_valid']),
                   'current': aligned[:, 2], 'current_valid': valid[:, 2]}
    return inputs, supervision, batch['route_id']
