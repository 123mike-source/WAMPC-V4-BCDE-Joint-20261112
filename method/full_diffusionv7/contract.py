"""Versioned V7 objective contract; the observation and rollout contract is V6's."""
from copy import deepcopy
import math
import re

from method.full_diffusionv6 import contract as v6

METHOD = 'full_diffusionv7'
ARCHITECTURE = v6.ARCHITECTURE
TRAINING_FORMAT = 'full_diffusionv7_train4hz_speed_v1'
CONTRACT = v6.CONTRACT
validate_manifest = v6.validate_manifest


def validate_config(config):
    if config['method'] != METHOD or not re.fullmatch(r'v7_[a-z0-9_]+', config['experiment']):
        raise ValueError('Expected a versioned V7 config')
    inherited = deepcopy(config)
    inherited.update(method=v6.METHOD, experiment='v6_inherited_contract')
    v6.validate_config(inherited)
    settings = config['loss']['speed_supervision']
    factors = settings['interval_weights']
    if (len(factors) != 6 or factors.index(max(factors)) != 1
            or not all(math.isfinite(v) and v > 0 for v in [*factors, settings['weight'],
                       settings['huber_beta_mps'], settings['transition_mps']])
            or not 1 <= settings['event_multiplier'] <= 3):
        raise ValueError('Expected positive speed weights with the control interval weighted most')
    probe = config['training_probe']
    if (probe['samples_per_group'] <= 0 or not 1 <= probe['max_spatial_ade_ratio'] <= 1.2
            or not re.fullmatch(r'[0-9a-f]{64}', probe['reference_sha256'])):
        raise ValueError('Invalid training probe or V6 reference')
