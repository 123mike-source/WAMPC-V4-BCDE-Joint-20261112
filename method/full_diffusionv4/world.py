"""Strict checkpoint loading for WAMPC's timed TF++ world model."""
import hashlib
import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn

from .geometry import FUTURE_TIMES, align_history, cumulative_valid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def timed_model_class(method='irwm_bev_t', architecture=None):
    """Reuse tensor-only source without the old package's LSF/CARLA init guard."""
    if method == 'irwm_bev_t_large':
        from method.irwm_bev_t_large import ARCHITECTURE
        if architecture != ARCHITECTURE:
            raise ValueError('Unknown large timed-world architecture')
        return importlib.import_module('method.irwm_bev_t_large.model').ActionBEVWorldModel
    if method != 'irwm_bev_t':
        raise ValueError('Unknown timed-world method')
    name = 'method.full_diffusionv4._timed_source'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'irwm_bev_t')]
        package.__package__ = name
        sys.modules[name] = package
    return importlib.import_module(name + '.model').ActionBEVWorldModel


class TimedWorldModel(nn.Module):
    def __init__(self, checkpoint, teacher_checkpoint, expected_sha256=None, allow_synthetic=False):
        super().__init__()
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Pretrained irwm_bev_t checkpoint is required: {checkpoint}')
        digest = sha256(checkpoint)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError('IRWM checkpoint checksum differs')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state.get('synthetic_interface_test_only') and not allow_synthetic:
            raise ValueError('Synthetic interface-test world weights cannot be used as a pretrained model')
        method = state['config'].get('method')
        model_class = timed_model_class(method, state['config'].get('architecture'))
        if (state['config']['model'].get('path_nodes') != 6
                or state['config']['model'].get('history') != 3):
            raise ValueError('World checkpoint must use three observations and six timed actions')
        dataset = state['dataset']
        teacher_hash = sha256(teacher_checkpoint)
        if dataset.get('teacher_sha256') != teacher_hash:
            raise ValueError('World model and planner use different frozen perception teachers')
        if dataset.get('query_times_seconds') != list(FUTURE_TIMES):
            raise ValueError('World checkpoint has a different future time grid')
        teacher_config_hash = dataset.get('teacher_config_sha256')
        if teacher_config_hash:
            teacher_config = Path(teacher_checkpoint).with_name('config.json')
            if not teacher_config.is_file() or sha256(teacher_config) != teacher_config_hash:
                raise ValueError('World model and planner use different frozen perception configs')
        contract = dataset.get('action_contract')
        if contract:
            expected = {'action_shape': [6, 2], 'feature_shape': [6, 64, 64, 64],
                        'times_seconds': list(FUTURE_TIMES),
                        'history_times_seconds': [-.5, -.25, 0.],
                        'feature_frame': 'fixed input observation ego frame',
                        'axes': 'width=x forward; height=y right', 'extent_metres': [-32, 32]}
            if any(contract.get(key) != value for key, value in expected.items()):
                raise ValueError('World checkpoint has an incompatible fixed-frame BEV/action contract')
        self.model = model_class(**state['config']['model'])
        self.model.load_state_dict(state['model'], strict=True)
        self.model.requires_grad_(False)
        self.provenance = {'checkpoint': str(checkpoint.resolve()), 'sha256': digest,
                           'teacher_sha256': teacher_hash, 'teacher_config_sha256': teacher_config_hash,
                           'method': method, 'model_config': state['config']['model'],
                           'action_contract': contract, 'progress': state.get('progress'),
                           'metrics': state.get('metrics'), 'source': 'WAMPC ' + method,
                           'architecture': state['config'].get('architecture')}
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, history, transforms, history_valid, prior_delta, prior_valid, speed):
        b = len(history)
        if history.shape[1:] != (3, 64, 64, 64):
            raise ValueError('IRWM needs the raw three-frame frozen teacher BEVs')
        aligned, spatial_valid = align_history(history, transforms, history_valid)
        action_valid = cumulative_valid(prior_valid.bool()) & history_valid.all(-1)[:, None]
        ready = action_valid[:, :5].any(-1)
        if not bool(ready.any()):
            return None, action_valid, aligned
        action = torch.where(action_valid[..., None], prior_delta.float(), 0.)
        path = action.cumsum(1)
        times = self.model.times[None].expand(b, -1)
        target_speed = action[:, 1].norm(dim=-1) / .5
        with torch.autocast(device_type=history.device.type, dtype=torch.bfloat16,
                            enabled=history.is_cuda):
            # Reuse the checkpoint's unchanged parameters through _evolve;
            # both supported architectures share this frame/time/mask contract.
            predicted = self.model._evolve(aligned[ready], spatial_valid[ready], path[ready], times[ready],
                                           action_valid[ready], path[ready], times[ready], action_valid[ready],
                                           speed.float().reshape(b)[ready], target_speed[ready])
        maps = history.new_zeros(b, 6, 64, 64, 64, dtype=predicted.dtype)
        maps[ready] = predicted
        if not bool(torch.isfinite(maps).all()):
            raise FloatingPointError('IRWM produced nonfinite future BEVs')
        return maps, action_valid, aligned
