"""Plan A style partial world tuning, adapted to the recurrent BCDE architecture.

Frozen observed/history encoders; last transition block, spatial decoder and
action fusion receive FP32 planning gradients through all five consumed maps.
"""
from copy import deepcopy

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from .geometry import align_history, cumulative_valid
from .world import TimedWorldModel

TUNING = {'schema': 'bcde_last_transition_decoder_action_v1',
          'parameter_prefixes': ['model.transition.1.', 'model.decode.', 'model.action.'],
          'precision': 'fp32_tf32', 'module_mode': 'eval_with_gradients',
          'gradient_rollout': 'full_five_step_bptt', 'activation_checkpoint': 'nonreentrant_step',
          'observed_encoders': 'frozen', 'supervision': 'planner_loss_only'}


def tuning_names(world, tuning):
    if tuning != TUNING:
        raise ValueError('Unknown partial BCDE tuning contract')
    names = {n for n, p in world.named_parameters() if any(n.startswith(s) for s in tuning['parameter_prefixes'])}
    if not names or any(not any(n.startswith(prefix) for n in names) for prefix in tuning['parameter_prefixes']):
        raise ValueError('The selected BCDE modules do not exist')
    return names


def load_tuned_state(world, state, tuning):
    names = tuning_names(world, tuning)
    if set(state) != names:
        raise ValueError('Joint checkpoint must contain exactly the approved tuned world parameters')
    current = dict(world.named_parameters())
    with torch.no_grad():
        for name in sorted(names):
            value = state[name]
            if value.shape != current[name].shape or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid tuned world tensor: ' + name)
            current[name].copy_(value)


class PartialTimedWorld(TimedWorldModel):
    def __init__(self, *args, tuning=None, activation_checkpoint=True, **kwargs):
        super().__init__(*args, **kwargs)
        if self.provenance['method'] != 'irwm_bev_t_large':
            raise ValueError('Partial recurrent tuning requires the BCDE model')
        self.tuning = deepcopy(tuning if tuning is not None else TUNING)
        self.tuned_names = tuning_names(self, self.tuning)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in self.tuned_names)
        self.activation_checkpoint = activation_checkpoint
        self.provenance['tuning'] = self.tuning
        self.provenance['trainable_parameters'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.eval()

    def export_tuned(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if n in self.tuned_names}

    def step(self, state, predicted, detail, action, dt, keep):
        m = self.model
        scale, shift = m.action(action).chunk(2, 1)
        encoded = m.state_input(predicted)
        encoded = encoded * (1 + .1 * scale[:, :, None, None].tanh()) + shift[:, :, None, None]
        reset, update = m.gates(torch.cat((state, encoded), 1)).sigmoid().chunk(2, 1)
        proposal = m.candidate(torch.cat((reset * state, encoded), 1))
        next_state = m.transition((1 - update) * state + update * proposal)
        change = m.decode(next_state, detail, predicted)
        next_prediction = predicted + change * dt.clamp(max=4.)[:, None, None, None]
        mask = keep[:, None, None, None]
        return torch.where(mask, next_state, state), torch.where(mask, next_prediction, predicted)

    def evolve(self, history, spatial_valid, delta, valid, speed):
        from method.irwm_bev_t_large.model import metric_grid
        m, b = self.model, len(history)
        path = delta.cumsum(1)
        times = m.times[None].expand(b, -1)
        target_speed = delta[:, 1].norm(dim=-1) / .5
        # The reference freezes observed/history encoding in partial mode.
        with torch.no_grad():
            spatial = metric_grid(history.device).permute(2, 0, 1)[None].expand(b, -1, -1, -1) / 32.
            visible = torch.cat((torch.where(spatial_valid, history, 0.), spatial_valid.float()), 2).flatten(1, 2)
            state, detail = m.history_encoder(torch.cat((visible, spatial), 1))
            nodes = torch.cat((path.new_zeros(b, 1, 2), torch.where(valid[..., None], path, 0.)), 1)
            intervals = torch.cat((nodes[:, :1], nodes[:, 1:] - nodes[:, :-1]), 1)
            token_times = torch.cat((times.new_zeros(b, 1), torch.where(valid, times, 0.)), 1)
            token_valid = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=history.device), valid), 1)
            tokens_in = torch.cat((nodes / 32., intervals / 5., token_times[..., None] / 6.,
                                   token_valid[..., None].float()), -1)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
                tokens = m.path_encoder(m.path_input(tokens_in) + m.path_position, src_key_padding_mask=~token_valid)
                flat = state.flatten(2).transpose(1, 2)
                context, _ = m.cross(m.cross_norm(flat), tokens, tokens, key_padding_mask=~token_valid, need_weights=False)
            state = state + context.transpose(1, 2).reshape_as(state)
            predicted = torch.where(spatial_valid[:, -1], history[:, -1], 0.)
        outputs = []
        previous_xy, previous_time = path.new_zeros(b, 2), times.new_zeros(b)
        for k in range(5):
            dt = (times[:, k] - previous_time).clamp_min(0.)
            action = torch.cat((path[:, k] / 32., (path[:, k] - previous_xy) / 5.,
                                times[:, k, None] / 6., dt[:, None] / 2., speed[:, None] / 15.,
                                target_speed[:, None] / 15.), 1)
            values = (state, predicted, detail, action, dt, valid[:, k])
            if torch.is_grad_enabled() and self.activation_checkpoint:
                state, predicted = checkpoint(self.step, *values, use_reentrant=False, preserve_rng_state=False)
            else:
                state, predicted = self.step(*values)
            # No temporal detach: all five rollout states participate in BPTT.
            outputs.append(predicted)
            previous_xy = torch.where(valid[:, k, None], path[:, k], previous_xy)
            previous_time = torch.where(valid[:, k], times[:, k], previous_time)
        return torch.stack(outputs + [outputs[-1]], 1)

    def forward(self, history, transforms, history_valid, prior_delta, prior_valid, speed):
        with torch.autocast(device_type=history.device.type, enabled=False):
            # Detach previous predictions/observations; keep gradients within this rollout.
            aligned, spatial = align_history(history.detach().float(), transforms, history_valid)
            valid = cumulative_valid(prior_valid.bool()) & history_valid.all(-1)[:, None]
            valid = valid.clone()
            valid[:, -1] = False
            ready = valid[:, :5].any(-1)
            if not bool(ready.any()):
                return None, valid, aligned
            delta = torch.where(valid[..., None], prior_delta.detach().float(), 0.)
            prediction = self.evolve(aligned[ready], spatial[ready], delta[ready], valid[ready], speed.float().reshape(-1)[ready])
            maps = prediction.new_zeros(len(history), 6, 64, 64, 64).index_copy(0, ready.nonzero().flatten(), prediction)
            if not bool(torch.isfinite(maps).all()):
                raise FloatingPointError('Nonfinite partially tuned world rollout')
        return maps, valid, aligned
