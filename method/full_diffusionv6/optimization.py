"""Discriminative planner fine-tuning, with a fixed DDP parameter set."""
import math

from torch import nn

BASE_GROUPS = ('base_output', 'base_decoder_late', 'base_decoder_early',
               'base_query', 'base_context', 'base_behavior')
ADAPTER_GROUPS = ('new', 'gate', 'distance')


def parameter_group(name, planner):
    if name.startswith('planner.adapter.'):
        if name.endswith('.gate'):
            return 'gate'
        if name.endswith(('.distance_logit', '.time_logit')):
            return 'distance'
        return 'new'
    if not name.startswith('planner.base.'):
        raise ValueError(f'Unexpected trainable parameter: {name}')
    relative = name.removeprefix('planner.base.')
    module = relative.split('.')[0]
    if module in ('layers', 'temporal_layers'):
        layer = int(relative.split('.')[1])
        if '.task_decoder.' in relative:
            return 'base_output'
        stack = planner.base.layers if module == 'layers' else planner.base.temporal_layers
        return 'base_decoder_late' if layer == len(stack) - 1 else 'base_decoder_early'
    if module in ('type_embedding', 'slot_embedding', 'path_query_encoder',
                  'temporal_query_encoder', 'time_mlp', 'temporal_time_mlp'):
        return 'base_query'
    if module in ('temporal', 'target_point_encoder'):
        return 'base_context'
    if module in ('behavior_head', 'behavior_condition', 'temporal_behavior_condition'):
        return 'base_behavior'
    raise ValueError(f'Unclassified V3 parameter: {name}')


def group_start_update(group, settings):
    tuning = settings['planner_tuning']
    if group in ('base_context', 'base_behavior'):
        return tuning['context_start_update']
    if group in BASE_GROUPS:
        return tuning['planner_start_update']
    return 0


def optimizer_groups(policy, settings):
    rates, modules, groups = settings['learning_rates'], dict(policy.named_modules()), {}
    seen = set()
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        group = parameter_group(name, policy.planner)
        if id(p) in seen:
            raise ValueError(f'Duplicate optimizer parameter: {name}')
        seen.add(id(p))
        owner = modules[name.rpartition('.')[0]]
        decay = 0.
        if p.ndim > 1 and not isinstance(owner, (nn.LayerNorm, nn.GroupNorm, nn.Embedding)):
            decay = settings['weight_decay'] if group == 'new' else settings['pretrained_weight_decay'] if group in BASE_GROUPS else 0.
        key = (group, decay)
        if key not in groups:
            groups[key] = {'params': [], 'param_names': [], 'lr': rates[group], 'weight_decay': decay,
                           'name': f'{group}/wd{decay}', 'module_group': group,
                           'start_update': group_start_update(group, settings)}
        groups[key]['params'].append(p)
        groups[key]['param_names'].append(name)
    expected = {id(p) for p in policy.parameters() if p.requires_grad}
    if seen != expected or not seen:
        raise ValueError('Optimizer must cover every trainable parameter exactly once')
    return list(groups.values())


def lr_multiplier(step, group, settings, total_updates):
    """step is the number of already completed updates (zero before update 1)."""
    start = group_start_update(group, settings)
    if step < start:
        return 0.
    warmup = min(settings['warmup_updates'], max(1, total_updates - start))
    elapsed = step - start
    if elapsed < warmup:
        return (elapsed + 1) / warmup
    # Scratch runs hit the requested minimum on the final optimizer update.
    endpoint = 1 if settings['planner_tuning']['schema'] == 'random_planner_v1' else 0
    progress = min(1., (elapsed - warmup) / max(1, total_updates - start - warmup - endpoint))
    return settings['minimum_lr_ratio'] + (1 - settings['minimum_lr_ratio']) * .5 * (1 + math.cos(math.pi * progress))


def make_scheduler(optimizer, settings, total_updates):
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, [lambda step, group=g['module_group']:
                               lr_multiplier(step, group, settings, total_updates)
                               for g in optimizer.param_groups])


def mask_inactive_gradients(optimizer, completed_updates):
    """No Adam moments or weight decay for delayed groups, including after resume."""
    for group in optimizer.param_groups:
        if completed_updates < group['start_update']:
            for p in group['params']:
                p.grad = None


def optimizer_report(optimizer):
    return [{k: v for k, v in group.items() if k != 'params'} |
            {'parameters': sum(p.numel() for p in group['params']), 'tensors': len(group['params'])}
            for group in optimizer.param_groups]
