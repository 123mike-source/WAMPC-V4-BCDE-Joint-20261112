"""Joint planning loss and partially trainable causal world rollout."""
from copy import deepcopy
import torch
from torch import nn

from .diffusion import SpatialTemporalVPFullDiffusion
from .geometry import motion_features
from .world_tuning import PartialTimedWorld, TUNING


def detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detach_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(detach_tree(v) for v in value)
    return value


class JointPlanner(nn.Module):
    def __init__(self, head, world, config):
        super().__init__()
        self.head, self.world, self.config = head, world, deepcopy(config)
        self.diffusion = SpatialTemporalVPFullDiffusion(head.codec, **{
            k: v for k, v in config['diffusion'].items() if k != 'inference_steps'})

    def train(self, mode=True):
        super().train(mode)
        # Eval mode controls world dropout/norm behavior, independently of gradients.
        self.world.eval()
        return self

    def conditions(self, batch, prior, prior_valid, ages, future_dropout=0.):
        valid = prior_valid
        if future_dropout:
            valid = valid & (torch.rand(len(valid), 1, device=prior.device) >= future_dropout)
        maps, valid, aligned = self.world(batch['raw_bev'], batch['history_transforms'],
                                          batch['history_mask'], prior, valid, batch['speed'])
        motion = motion_features(aligned.detach(), batch['history_mask'], batch['speed'], prior, ages)
        with torch.autocast(device_type=prior.device.type, dtype=torch.bfloat16, enabled=prior.is_cuda):
            condition = self.head.prepare_condition(
                batch['raw_bev'][:, -1], batch['memory'], batch['ego'], batch['target_point'],
                batch['raw_bev'][:, :2], batch['history_transforms'][:, :2], batch['history_mask'][:, :2],
                batch['speed'], batch['command'], maps, prior, valid, motion)
        return condition, valid

    @torch.no_grad()
    def sample(self, condition, batch_size, device, generator, steps):
        training = self.head.training
        self.head.eval()
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                result = self.diffusion.sample(lambda x, t: self.head(x, t, condition)['predictions'][-1],
                                               batch_size, device, generator, steps)
                return self.head.codec.decode(result)
        finally:
            self.head.train(training)

    def forward(self, batch, prior, prior_valid, ages, generator):
        settings = self.config['training']
        condition, valid = self.conditions(batch, prior, prior_valid, ages, settings['future_dropout'])
        device, b = batch['path'].device, len(batch['path'])
        clean = self.head.codec.encode(batch['path'], batch['temporal_delta'])
        steps = torch.randint(self.diffusion.train_steps, (b,), device=device)
        steps = torch.where(torch.rand(b, device=device) < settings['max_noise_fraction'], self.diffusion.train_steps - 1, steps)
        noisy, t = self.diffusion.corrupt(clean, steps)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            output = self.head(noisy, t, condition)
            losses = self.head.loss_components(output, clean, batch['spatial_mask'], batch['temporal_mask'],
                                                batch['behavior_targets'], batch['behavior_mask'], self.config['loss'])
        # As in the reference, make cold starts/dropout safe for all DDP ranks.
        anchor = sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
        losses['loss'] = losses['loss'] + anchor
        path, delta = self.sample(detach_tree(condition), b, device, generator, settings['rolling_inference_steps'])
        return {**losses, 'sampled_path': path.detach(), 'sampled_delta': delta.detach(),
                'future_ready': valid[:, :5].any(-1).float().sum(),
                'max_noise_count': (steps == self.diffusion.train_steps - 1).float().sum()}


def optimizer_groups(policy, initialization):
    """Reference differential LRs; every trainable tensor is assigned once."""
    migrated = set(initialization.get('copied', [])) | set(initialization.get('transformed', []))
    rates = {'world': 1e-6, 'pretrained_context': 1e-5, 'pretrained_planner': 2e-5,
             'new': 1e-4, 'gate': 1e-3, 'distance': 1e-2, 'current_points': 1e-3}
    rates.update(policy.config['training'].get('learning_rates', {}))
    modules, groups, audit = dict(policy.named_modules()), {}, {}
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('world.'):
            group = 'world'
        elif name.endswith('.future.gate'):
            group = 'gate'
        elif name.endswith('.future.distance_logit'):
            group = 'distance'
        elif name.endswith('.current_point_logits'):
            group = 'current_points'
        elif name.removeprefix('head.') not in migrated:
            group = 'new'
        elif name.startswith(('head.layers.', 'head.temporal_layers.', 'head.path_query_encoder.', 'head.temporal_query_encoder.')):
            group = 'pretrained_planner'
        else:
            group = 'pretrained_context'
        owner_name, _, leaf = name.rpartition('.')
        owner = modules[owner_name]
        no_decay = (leaf == 'bias' or isinstance(owner, (nn.LayerNorm, nn.GroupNorm, nn.Embedding))
                    or group in ('gate', 'distance', 'current_points'))
        decay = 0. if no_decay else (.01 if group == 'new' else .0001)
        key = (group, decay)
        if key not in groups:
            groups[key] = {'params': [], 'lr': rates[group], 'weight_decay': decay, 'name': f'{group}/wd{decay}'}
            audit[groups[key]['name']] = {'lr': rates[group], 'weight_decay': decay, 'parameters': 0, 'names': []}
        groups[key]['params'].append(p)
        audit[groups[key]['name']]['parameters'] += p.numel()
        audit[groups[key]['name']]['names'].append(name)
    return list(groups.values()), audit
