"""Physical times for training; the closed-loop schedule is deliberately separate."""
import math
import re

METHOD = 'full_diffusionv6'
ARCHITECTURE = 'dual_query_dit_future_adapter_v6'
TRAINING_FORMAT = 'full_diffusionv6_train4hz_scratch_v1'
# V6 reuses the immutable 4Hz label and perception cache index made for V5.
INDEX_FORMAT = 'full_diffusionv5_index_train4hz_v1'
CONTRACT = {
    'version': 'fresh_world_4hz_v1',
    'anchor_interval_seconds': 0.25,
    'history_times_seconds': [-0.5, -0.25, 0.0],
    'prior_age_seconds': 0.25,
    'cache_age_seconds': 0.0,
    'future_map_times_seconds': [0.5, 1.0, 1.5, 2.0, 2.5],
    'prior_source': 'detached_model_temporal_prediction',
    'pose_source': 'observed_current_ego_pose',
    'tail_policy': 'mask_outside_previous_plan',
}


def validate_config(config):
    if config.get('method') != METHOD or config.get('architecture') != ARCHITECTURE:
        raise ValueError('V6 requires a V6 training config')
    if not re.fullmatch(r'v6_[a-z0-9_]+', config.get('experiment', '')):
        raise ValueError('V6 requires an explicit v6_ experiment name')
    if config.get('planner_initialization') != 'random':
        raise ValueError('V6 requires a random planner')
    if config['training'].get('backfill_missing_prior') is not True:
        raise ValueError('V6 requires no-grad previous-raw-frame prior backfill')
    pin = config['dataset']
    if (not all(re.fullmatch(r'[a-f0-9]{64}', pin.get(key, ''))
                for key in ('policy_sha256', 'manifest_sha256', 'index_identity'))
            or type(pin.get('train_items')) is not int or pin['train_items'] <= 0
            or pin.get('validation_items') != 0):
        raise ValueError('V6 requires a pinned all-training index with zero validation anchors')
    if config.get('training_contract') != CONTRACT:
        raise ValueError('Training requires fresh_world_4hz_v1; old cached/parity-chain configs cannot be reused')
    if (not math.isclose(config['observation_interval_seconds'], .25)
            or not math.isclose(config['labels']['raw_measurement_interval_seconds'], .25)
            or not math.isclose(config['labels']['temporal_interval_seconds'], .5)
            or config['labels']['temporal_points'] != 6):
        raise ValueError('Expected 0.25s observations and six 0.5s temporal targets')
    if config['world'].get('tuning') is not None:
        raise ValueError('This training version keeps the pretrained BCDE world frozen')
    from .optimization import ADAPTER_GROUPS, BASE_GROUPS
    settings = config['training']
    if type(settings.get('backfill_missing_prior', False)) is not bool:
        raise ValueError('backfill_missing_prior must be a boolean')
    tuning = settings.get('planner_tuning', {})
    if tuning.get('schema') != 'random_planner_v1' or tuning.get('enabled') is not True:
        raise ValueError('V6 requires the scratch optimizer policy')
    for key in ('planner_start_update', 'context_start_update'):
        if type(tuning.get(key)) is not int or tuning[key] < 0:
            raise ValueError(f'{key} must be a nonnegative update count')
    if tuning['planner_start_update'] != 0 or tuning['context_start_update'] != 0:
        raise ValueError('A random planner must train all groups from update one')
    for group in ADAPTER_GROUPS + BASE_GROUPS:
        rate = settings['learning_rates'].get(group)
        if not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f'Expected a finite positive learning rate for {group}')
    if (type(settings.get('warmup_updates')) is not int or settings['warmup_updates'] < 1
            or not 0 < settings.get('minimum_lr_ratio', 0) <= 1
            or not math.isfinite(settings.get('pretrained_weight_decay', float('nan')))
            or settings['pretrained_weight_decay'] < 0):
        raise ValueError('Invalid fine-tuning warmup, minimum learning rate or weight decay')
    betas = settings.get('adam_betas', [])
    if (len(betas) != 2 or not all(isinstance(x, (int, float)) and 0 <= x < 1 for x in betas)
            or not math.isfinite(settings.get('adam_eps', float('nan'))) or settings['adam_eps'] <= 0):
        raise ValueError('Invalid AdamW beta or epsilon')
    if tuning['enabled'] and config['loss'].get('behavior_weight', 0) <= 0:
        raise ValueError('An unfrozen behavior head requires its supervised classification loss')
    steps = config['training']['rolling_inference_steps']
    if type(steps) is not int or steps < 1 or steps != config['diffusion']['inference_steps']:
        raise ValueError('Rolling prior generation and validation must use the same positive DDIM step count')
    if 'age_weights' in config['loss']:
        raise ValueError('Fresh-world training has one loss per anchor, without cached-age loss weights')


def validate_manifest(manifest, config=None):
    if manifest.get('method') != INDEX_FORMAT or manifest.get('training_contract') != CONTRACT:
        raise ValueError('Prepare a new single-anchor 4Hz index; the old triplet index is incompatible')
    if config is not None and manifest['label_config'] != config['labels']:
        raise ValueError('Training labels and index labels differ')
