"""Physical times for training; the closed-loop schedule is deliberately separate."""
import math

TRAINING_FORMAT = 'full_diffusionv5_train4hz_v1'
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
    if config.get('training_contract') != CONTRACT:
        raise ValueError('Training requires fresh_world_4hz_v1; old cached/parity-chain configs cannot be reused')
    if (not math.isclose(config['observation_interval_seconds'], .25)
            or not math.isclose(config['labels']['raw_measurement_interval_seconds'], .25)
            or not math.isclose(config['labels']['temporal_interval_seconds'], .5)
            or config['labels']['temporal_points'] != 6):
        raise ValueError('Expected 0.25s observations and six 0.5s temporal targets')
    if config['world'].get('tuning') is not None:
        raise ValueError('This training version keeps the pretrained BCDE world frozen')
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
