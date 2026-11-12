"""Explicit V3 -> V4 initialization; the result is marked as untrained."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from .geometry import SPATIAL_SOURCE_INDICES, interpolate_nodes, spatial_ten_to_six
from .model import SpatialTemporalFullDiffusionHead
from .world import TimedWorldModel, sha256


ROOT = Path(__file__).resolve().parents[2]


def initialize_from_v3(source, config, normalization=None):
    if (source['config'].get('method') != 'full_diffusionv3'
            or source['config'].get('architecture') != 'independent_dual_query64'):
        raise ValueError('Initialization needs the independently trained V3 dual-query checkpoint')
    old = source['model']
    supplied_normalization = normalization is not None
    if normalization is None:
        # A conservative affine scale for initialization, not measured V4 moments.
        # Training has a separate tool to measure actual resampled-target moments.
        normalization = {name: torch.cat((spatial_ten_to_six(old['codec.' + name][:10]),
                                         old['codec.' + name][10:]), 0).tolist()
                         for name in ('mean', 'std')}
    head = SpatialTemporalFullDiffusionHead(**config['model'], normalization=normalization)
    target = head.state_dict()
    copied, transformed, fresh = [], [], []
    for key in target:
        if key in ('codec.mean', 'codec.std', 'slot_ids', 'type_ids'):
            continue
        if key in old and old[key].shape == target[key].shape:
            target[key] = old[key].clone()
            copied.append(key)
        else:
            fresh.append(key)
    target['slot_embedding.weight'] = torch.cat((
        interpolate_nodes(old['slot_embedding.weight'][:10].T[..., None], range(10), SPATIAL_SOURCE_INDICES)
        .squeeze(-1).T, old['slot_embedding.weight'][10:]), 0)
    expansion = interpolate_nodes(torch.eye(6), SPATIAL_SOURCE_INDICES, range(10))
    target['path_query_encoder.0.weight'] = torch.einsum(
        'dpc,ps->dsc', old['path_query_encoder.0.weight'].reshape(256, 10, 64), expansion).reshape(256, 6 * 64)
    transformed += ['slot_embedding.weight', 'path_query_encoder.0.weight']
    for number in range(config['model']['layers']):
        base = f'layers.{number}.'
        for suffix in ('cross_bev_attention.attention_weights.weight', 'cross_bev_attention.attention_weights.bias'):
            key = base + suffix
            values = old[key]
            if values.ndim == 2:
                target[key] = interpolate_nodes(values.T[..., None], range(10), SPATIAL_SOURCE_INDICES).squeeze(-1).T
            else:
                target[key] = interpolate_nodes(values[:, None], range(10), SPATIAL_SOURCE_INDICES).squeeze(-1)
            transformed.append(key)
        for suffix in ('task_decoder.4.weight', 'task_decoder.4.bias'):
            key = base + suffix
            values = old[key]
            if values.ndim == 2:
                values = values.reshape(10, 2, -1).permute(2, 0, 1)
                target[key] = spatial_ten_to_six(values).permute(1, 2, 0).reshape(12, -1)
            else:
                target[key] = spatial_ten_to_six(values.reshape(10, 2)).flatten()
            transformed.append(key)
    head.load_state_dict(target, strict=True)
    return head, {'copied': copied, 'transformed': transformed,
                  'new': sorted(set(fresh) - set(transformed)),
                  'normalization_source': 'supplied_v4_moments' if supplied_normalization else 'v3_bootstrap_affine'}


def load_head(checkpoint, device='cpu', allow_untrained=False):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = deepcopy(state['config'])
    if (config.get('method') != 'full_diffusionv4'
            or config.get('architecture') != 'independent_dual_query64_irwm_fixed_frame_v1'):
        raise ValueError('V4 requires its own six-point world-conditioned checkpoint')
    if state.get('requires_training', False) and not allow_untrained:
        raise ValueError('This checkpoint is initialization only; train the new six-point future modules first')
    head = SpatialTemporalFullDiffusionHead(**config['model'], normalization=state['normalization'])
    head.load_state_dict(state['model'], strict=True)
    return head.to(device), config, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3-checkpoint', type=Path, required=True)
    parser.add_argument('--world-checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--normalization', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    teacher = ROOT / config['backbone_directory'] / 'model_0030_1.pth'
    world = TimedWorldModel(args.world_checkpoint, teacher, config['world']['checkpoint_sha256'])
    config['world']['checkpoint'] = str(args.world_checkpoint.resolve())
    config['world']['checkpoint_sha256'] = world.provenance['sha256']
    torch.manual_seed(config['seed'])
    source = torch.load(args.v3_checkpoint, map_location='cpu', weights_only=False)
    normalization = None
    if args.normalization:
        artifact = json.loads(args.normalization.read_text())
        if artifact['method'] != 'full_diffusionv4' or artifact['label_config'] != config['labels']:
            raise ValueError('V4 normalization label identity differs')
        normalization = artifact['normalization']
        config['normalization_file'] = str(args.normalization.resolve())
    head, audit = initialize_from_v3(source, config, normalization)
    audit['normalization_source'] = 'measured_v4_train_targets' if args.normalization else 'v3_bootstrap_affine'
    checkpoint = {'format': 'full_diffusionv4_v1', 'model': head.state_dict(), 'config': config,
                  'normalization': {'mean': head.codec.mean.tolist(), 'std': head.codec.std.tolist()},
                  'world': world.provenance, 'requires_training': True, 'initialization': audit,
                  'source_v3_sha256': sha256(args.v3_checkpoint),
                  'dataset': source['dataset'], 'progress': {'epoch': 0, 'updates': 0}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.partial')
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    print(json.dumps({'checkpoint': str(args.output), 'requires_training': True, 'world': world.provenance,
                      'copied_tensors': len(audit['copied']), 'transformed_tensors': len(audit['transformed']),
                      'new_tensors': len(audit['new'])}, indent=2))


if __name__ == '__main__':
    main()
