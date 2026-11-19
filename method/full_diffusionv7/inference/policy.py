"""V7 fine-tuned checkpoint with fresh perception, world and planning every 20Hz tick."""
from contextlib import nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv3.inference.policy import FullDiffusionV3Network
from method.full_diffusionv4.world import sha256
from method.full_diffusionv7.contract import ARCHITECTURE, CONTRACT, METHOD, TRAINING_FORMAT, validate_config
from method.full_diffusionv6.model import FutureAdaptedPlanner, align_future, build_world, generate_future, build_random_head_from_v3_metadata
from .history import LaggedPlanHistory, TickBEVHistory
from .clock import PlanningClock

ROOT = Path(__file__).resolve().parents[3]
ONLINE_CONTRACT = {
    'version': 'v7_fresh_world_plan20hz_lag5_v1',
    'simulation_interval_seconds': .05, 'planning_interval_seconds': .05,
    'world_refresh_interval_seconds': .05, 'observation_times_seconds': [-.5, -.25, 0.],
    'prior_age_seconds': .25, 'cache_age_at_planning_seconds': 0.,
    'future_map_times_seconds': [.5, 1., 1.5, 2., 2.5],
    'plan_history': 'exact integer tick k-5; never latest k-1',
    'observation_history': 'exact integer ticks k-10, k-5, k',
    'control': 'fresh current-tick decoded path and temporal delta; no cached control frames',
    'temporal_tail': 'invalid outside previous 3s horizon; never feed extrapolated tail to IRWM',
    'spatial_points': 10, 'temporal_points': 6, 'inference_steps': 25,
}


def advance_control_plan(path, delta, previous_pose, current_pose, age):
    """Rebase the cached plan into the actual current ego; no future-tail extrapolation."""
    if not 0 <= age < .25 + 1e-7:
        raise ValueError('Control cache must be refreshed at least every 0.25s')
    with torch.autocast(device_type=path.device.type, enabled=False):
        transform = torch.linalg.solve(current_pose.double(), previous_pose.double()).float()
        rebased_path = torch.einsum('bij,bpj->bpi', transform[:, :2, :2], path.float()) + transform[:, None, :2, 2]
        positions = torch.cat((delta.new_zeros(len(delta), 1, 2), delta.float().cumsum(1)), 1)
        query = torch.arange(1, 7, device=path.device).float()[None] * .5 + age
        valid = (query <= 3. + 1e-7).expand(len(path), -1)
        location = (query / .5).clamp(0, 6).expand(len(path), -1)
        left = location.floor().long().clamp(max=5)
        weight = location - left
        points = positions.gather(1, left[..., None].expand(-1, -1, 2)) * (1 - weight[..., None])
        points += positions.gather(1, (left + 1)[..., None].expand(-1, -1, 2)) * weight[..., None]
        points = torch.einsum('bij,bpj->bpi', transform[:, :2, :2], points) + transform[:, None, :2, 2]
        increments = torch.cat((points[:, :1], points[:, 1:] - points[:, :-1]), 1)
        increments = torch.where(valid[..., None], increments, 0.)
    return rebased_path, increments, valid


class FullDiffusionV7Network(FullDiffusionV3Network):
    def __init__(self, garage_net, checkpoint_path, device):
        nn.Module.__init__(self)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if (state.get('method') != METHOD or state.get('format') != TRAINING_FORMAT
                or state.get('training_contract') != CONTRACT
                or state['progress'].get('checkpoint_selection') != 'fixed_training_probe'):
            raise ValueError('Expected a V7 scratch checkpoint')
        self.run_config = deepcopy(state['config'])
        validate_config(self.run_config)
        config = self.run_config
        progress = state['progress']
        if state.get('experiment') != config['experiment']:
            raise ValueError('V7 checkpoint experiment and config differ')
        epoch = progress['epoch']
        batch = progress.get('step_in_epoch', 0)
        steps = state['dataset']['steps_per_epoch']
        if batch:
            selection_path = os.environ.get('WMMPC_V7_EVAL_SELECTION')
            if not selection_path:
                raise ValueError('An explicit evaluation selection is required for a training snapshot')
            selected = json.loads(Path(selection_path).read_text())
            allowed_paths = {Path(selected['checkpoint']),
                Path(selected['run_root']) / 'checkpoint/selected.pt',
                Path(selected['pilot_root']) / 'checkpoint/selected.pt'}
            if (selected.get('checkpoint_kind') != 'training_snapshot'
                    or Path(checkpoint_path).resolve() not in allowed_paths
                    or not 0 <= epoch < config['training']['epochs']
                    or not 0 < batch < steps
                    or steps != selected['steps_per_epoch']
                    or (epoch, batch, progress['updates']) != (selected['epoch'], selected['step_in_epoch'], selected['updates'])
                    or progress['updates'] != epoch * steps + batch
                    or sha256(Path(checkpoint_path)) != selected['checkpoint_sha256']):
                raise ValueError('V7 checkpoint does not match the selected training snapshot')
        elif (not 1 <= epoch <= config['training']['epochs']
                or progress['updates'] != epoch * steps
                or progress['training_probe']['epoch'] != epoch):
            raise ValueError('Expected a V7 epoch selected by the fixed training probe')
        if (config.get('architecture') != ARCHITECTURE
                or config.get('controller') != 'automot_temporal_pid_v1'):
            raise ValueError('Unexpected V7 architecture or controller')
        self.checkpoint_inference_steps = config['diffusion']['inference_steps']
        requested = int(os.environ.get('WMMPC_DIFFUSION_INFERENCE_STEPS', self.checkpoint_inference_steps))
        if requested != 25 or self.checkpoint_inference_steps != 25:
            raise ValueError('This evaluation fixes both rolling and planning DDIM to 25 steps')
        if sha256(ROOT / config['v3']['checkpoint']) != config['v3']['sha256']:
            raise ValueError('The original V3 normalization/model provenance changed')
        base, self.v3_config, _ = build_random_head_from_v3_metadata(ROOT / config['v3']['checkpoint'])
        if (self.v3_config['labels'] != config['labels']
                or self.v3_config['model']['history_frames'] != 2
                or self.v3_config['observation_interval_seconds'] != .25):
            raise ValueError('V3 and V7 input/target contracts differ')
        self.planner = FutureAdaptedPlanner(base, config['future_sampler'], train_base=True)
        if not config['training']['planner_tuning']['enabled'] or state.get('planner_base') is None:
            raise ValueError('The trained planner_base is required; adapter-only loading is invalid')
        self.planner.base.load_state_dict(state['planner_base'], strict=True)
        self.planner.adapter.load_state_dict(state['adapter'], strict=True)
        self.perception = FrozenPlannerPerception.from_net(garage_net)
        self.world = build_world(config, ROOT)
        if (state['world']['sha256'] != self.world.provenance['sha256']
                or state['world']['teacher_sha256'] != self.world.provenance['teacher_sha256']
                or state.get('world_tuned_state') is not None):
            raise ValueError('This V7 checkpoint requires its unchanged frozen BCDE world and teacher')
        self.diffusion = SpatialTemporalVPFullDiffusion(self.planner.codec, **{
            key: value for key, value in self.v3_config['diffusion'].items() if key != 'inference_steps'})
        self.history = TickBEVHistory()
        self.online_contract = ONLINE_CONTRACT.copy()
        self.rolling = LaggedPlanHistory()
        self.clock = PlanningClock()
        self.generator = torch.Generator(device=device).manual_seed(config['seed'])
        self.last_target_speed = self.last_temporal_delta = self.last_behavior_probabilities = None
        self.last_future_valid = None
        self.last_world_active = False
        self.last_prior_age = 0.
        self.last_prior_valid = [False] * 6
        self.last_timings = {}
        self.planning_updates = self.control_updates = 0
        self._scene, self._timestamp, self._pose = 'route', None, None
        self.last_tick = self.last_prior_tick = self.last_future_tick = None
        self.last_history_ticks = [None, None, None]
        self.last_future_disabled_reason = "warmup"
        self.last_tick_gap = 0
        self.last_planning_update = False
        self.load_report = {'format': state['format'], 'progress': state['progress'],
                            'planner_base_tensors': len(state['planner_base']),
                            'adapter_tensors': len(state['adapter']), 'world_frozen': True,
                            'online_contract': self.online_contract}
        if (garage_net.config.speed_kp, garage_net.config.speed_ki,
                garage_net.config.speed_kd, garage_net.config.speed_n) != (1.75, 1., 2., 20):
            raise ValueError('V7 preserves the V3 AutoMoT longitudinal PID settings')
        self.requires_grad_(False).to(device).eval()

    def set_observation(self, pose, timestamp, scene_id=None):
        scene = self._scene if scene_id is None else str(scene_id)
        tick = PlanningClock.tick(float(timestamp))
        previous_tick = PlanningClock.tick(self._timestamp) if self._timestamp is not None else None
        reset = scene != self._scene or (previous_tick is not None and tick <= previous_tick)
        if reset:
            self.history.entries.clear()
            self.rolling.reset()
            self.clock.reset()
            previous_tick = None
        self.last_tick_gap = max(0, tick - previous_tick - 1) if previous_tick is not None else 0
        self._scene, self.last_tick = scene, tick
        self.history.set_observation(pose, tick * .05)
        self._pose = torch.as_tensor(self.history.pose, dtype=torch.float64)
        self._timestamp = tick * .05

    def _future(self, raw, transforms, valid, prior, prior_valid, ages, speed, ego):
        available = prior_valid.any(-1)
        if bool((available & ((ages - .25).abs() > 1e-6)).any()):
            raise ValueError('Online prior must be exactly 0.25s old')
        if bool(prior_valid[:, -1].any()) or bool((available & ~prior_valid[:, :5].all(-1)).any()):
            raise ValueError('The prior must cover five future points and mask the sixth tail')
        future = generate_future(self.world, self.planner, raw, transforms, valid,
                                 prior, prior_valid, ages, speed, ego)
        if future is None:
            return None
        identity = torch.eye(3, device=raw.device).expand(len(raw), -1, -1)
        return align_future(future, identity, torch.zeros(len(raw), device=raw.device))

    @torch.no_grad()
    def forward(self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next=None):
        if self._pose is None or self._timestamp is None or len(rgb) != 1:
            raise RuntimeError('V7 inference needs a batch1 observation pose and simulation timestamp')
        if not self.clock.due(self._timestamp):
            raise RuntimeError('Each observation tick may be decoded only once')
        poses = self._pose.to(rgb.device)[None]
        self.last_planning_update = True
        started = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if rgb.is_cuda else None
        def mark(i):
            if events is not None:
                events[i].record()
        mark(0)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if rgb.is_cuda else nullcontext()
        with context:
            batch = {'rgb': rgb, 'lidar': lidar_bev, 'target_point': target_point,
                     'target_point_next': target_point_next, 'speed': ego_vel, 'command': command}
            bev, memory, ego = self.perception(batch)
            mark(1)
            history, transforms, history_valid = self.history.get_inputs(bev)
            prior, prior_valid, ages = self.rolling.prior([self._scene], poses, [self._timestamp])
            self.last_history_ticks = self.history.last_ticks + [self.last_tick]
            self.last_prior_tick = self.rolling.last_selected_ticks[0]
            if self.last_prior_tick is not None and self.last_prior_tick != self.last_tick - 5:
                raise AssertionError('Prior plan must be exactly five simulation ticks old')
            for selected, lag in zip(self.history.last_ticks, (10, 5)):
                if selected is not None and selected != self.last_tick - lag:
                    raise AssertionError('BEV history must be exactly ten/five simulation ticks old')
            raw = torch.cat((history, bev[:, None]), 1)
            world_transforms = torch.cat((transforms, torch.eye(3, device=rgb.device)[None, None]), 1)
            frame_valid = torch.cat((history_valid, history_valid.new_ones(1, 1)), 1)
            future = self._future(raw, world_transforms, frame_valid, prior, prior_valid, ages, ego_vel, ego)
            mark(2)
            if future is not None:
                if bool(future['age'].abs().any()):
                    raise AssertionError('Future conditions must be generated at the current tick')
                expected = torch.tensor([.5, 1., 1.5, 2., 2.5], device=rgb.device)
                torch.testing.assert_close(future['map_time'][0], expected, atol=0, rtol=0)
            condition = self.planner.condition(bev, memory, ego, target_point, history, transforms,
                                               history_valid, ego_vel, command, future)
            mark(3)
            sampled = self.diffusion.sample(
                lambda x, t: self.planner(x, t, condition)['predictions'][-1],
                1, rgb.device, self.generator, 25)
            path, delta = self.planner.codec.decode(sampled)
            mark(4)
            boxes = self.perception.net.head(bev) if self.config.detect_boxes else None
        if not bool(torch.isfinite(path).all() & torch.isfinite(delta).all()):
            raise FloatingPointError('V7 produced nonfinite control targets')
        # Commit only after successful decoding. No fallback to k-1 is permitted.
        self.rolling.update([self._scene], poses, [self._timestamp], delta.float())
        self.last_behavior_probabilities = condition['behavior_logits'].float().softmax(-1).detach()
        self.last_future_valid = future['valid'].detach() if future is not None else None
        self.last_world_active = future is not None and bool(future['valid'].any())
        self.last_future_tick = self.last_tick if future is not None else None
        if self.last_prior_tick is None:
            self.last_future_disabled_reason = 'missing_exact_k_minus_5_plan'
        elif not bool(history_valid.all()):
            self.last_future_disabled_reason = 'missing_exact_bev_history'
        elif not self.last_world_active:
            self.last_future_disabled_reason = 'no_valid_future_tokens'
        else:
            self.last_future_disabled_reason = None
        self.last_prior_age, self.last_prior_valid = float(ages[0]), prior_valid[0].tolist()
        self.clock.mark(self._timestamp)
        self.planning_updates += 1
        self.last_temporal_delta = delta.detach().float()
        speed = self.planner.codec.target_speed(delta)
        if not bool(torch.isfinite(speed).all()):
            raise FloatingPointError('V7 produced a nonfinite target speed')
        self.last_target_speed = float(speed[0])
        self.control_updates += 1
        if self.clock.age(self._timestamp) != 0. or self.planning_updates != self.control_updates:
            raise AssertionError('Every control update requires a fresh current-tick plan')
        if events is not None:
            events[-1].synchronize()
        self.last_timings = {'wall_seconds': time.perf_counter() - started}
        if events is not None:
            for name, i, j in [('perception', 0, 1), ('world', 1, 2), ('condition', 2, 3), ('denoising', 3, 4)]:
                self.last_timings[name + '_gpu_ms'] = events[i].elapsed_time(events[j])
        bins = torch.tensor(self.config.target_speeds, device=speed.device, dtype=speed.dtype)
        compatibility = speed.clamp(0, max(self.config.target_speeds))
        upper = torch.searchsorted(bins, compatibility).clamp(1, len(bins) - 1)
        weight = (compatibility - bins[upper - 1]) / (bins[upper] - bins[upper - 1])
        probabilities = torch.zeros(len(speed), len(bins), device=speed.device, dtype=speed.dtype)
        probabilities.scatter_(1, (upper - 1)[:, None], (1 - weight)[:, None])
        probabilities.scatter_(1, upper[:, None], weight[:, None])
        return None, probabilities.clamp_min(1e-12).log(), path.float(), None, None, None, boxes, None, None, None
