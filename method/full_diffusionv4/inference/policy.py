"""Causal V4 policy: one world rollout per frame, six points in each branch."""
from contextlib import nullcontext
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from method.full_diffusion2.inference.policy import TemporalBEVHistory
from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3.inference.policy import FullDiffusionV3Network
from method.full_diffusionv4.bootstrap import load_head
from method.full_diffusionv4.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv4.geometry import motion_features, spatial_six_to_controller
from method.full_diffusionv4.rolling import RollingPlanStore
from method.full_diffusionv4.world import TimedWorldModel

ROOT = Path(__file__).resolve().parents[3]


class FullDiffusionV4Network(FullDiffusionV3Network):
    def __init__(self, garage_net, checkpoint_path, device, allow_untrained=False, allow_synthetic_world=False):
        nn.Module.__init__(self)
        self.head, self.run_config, checkpoint = load_head(checkpoint_path, device, allow_untrained)
        config = self.run_config
        self.checkpoint_inference_steps = config['diffusion']['inference_steps']
        steps = int(os.environ.get('WMMPC_DIFFUSION_INFERENCE_STEPS', self.checkpoint_inference_steps))
        if not 1 <= steps <= config['diffusion']['train_steps']:
            raise ValueError('Invalid V4 inference step override')
        config['diffusion']['inference_steps'] = steps
        self.perception = FrozenPlannerPerception.from_net(garage_net)
        world_path = Path(os.environ.get('WMMPC_WORLD_CHECKPOINT', config['world']['checkpoint']))
        world_path = ROOT / world_path
        teacher_path = ROOT / config['backbone_directory'] / 'model_0030_1.pth'
        world_class, world_options = TimedWorldModel, {}
        if config['world'].get('tuning'):
            from method.full_diffusionv4.world_tuning import PartialTimedWorld
            world_class = PartialTimedWorld
            world_options = {'tuning': config['world']['tuning'], 'activation_checkpoint': False}
        self.world = world_class(world_path, teacher_path, config['world']['checkpoint_sha256'],
                                 allow_synthetic=allow_synthetic_world, **world_options)
        if checkpoint['world']['sha256'] != self.world.provenance['sha256']:
            raise ValueError('Planner was trained with a different world checkpoint')
        if checkpoint.get('world_tuned_state') is not None:
            from method.full_diffusionv4.world_tuning import load_tuned_state
            load_tuned_state(self.world, checkpoint['world_tuned_state'], config['world']['tuning'])
            self.world.provenance['joint_tuning'] = config['world']['tuning']
        elif config['world'].get('tuning') and not checkpoint.get('requires_training', False):
            raise ValueError('Jointly trained planner is missing its tuned world parameters')
        self.world.requires_grad_(False)
        self.diffusion = SpatialTemporalVPFullDiffusion(self.head.codec, **{
            key: value for key, value in config['diffusion'].items() if key != 'inference_steps'})
        self.history = TemporalBEVHistory(2, config['observation_interval_seconds'])
        self.rolling = RollingPlanStore(config['world']['max_prior_age_seconds'])
        self.generator = torch.Generator(device=device).manual_seed(config['seed'])
        self.last_target_speed = None
        self.last_temporal_delta = None
        self.last_behavior_probabilities = None
        self.last_future_valid = None
        self.last_world_active = False
        self._scene = 'route'
        self._timestamp = None
        self._pose = None
        if (garage_net.config.speed_kp, garage_net.config.speed_ki,
                garage_net.config.speed_kd, garage_net.config.speed_n) != (1.75, 1.0, 2.0, 20):
            raise ValueError('V4 preserves the V3 AutoMoT temporal PID settings')
        self.to(device)
        self.eval()

    def set_observation(self, pose, timestamp, scene_id=None):
        scene = self._scene if scene_id is None else str(scene_id)
        if scene != self._scene or (self._timestamp is not None and timestamp <= self._timestamp):
            self.history.entries.clear()
            self.rolling.reset()
        self._scene = scene
        self.history.set_observation(pose, timestamp)
        self._pose = torch.as_tensor(self.history.pose, dtype=torch.float64)
        self._timestamp = float(timestamp)

    @torch.no_grad()
    def forward(self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next=None):
        if self._pose is None or self._timestamp is None or len(rgb) != 1:
            raise RuntimeError('V4 online inference needs a batch1 observation pose and timestamp')
        batch = {'rgb': rgb, 'lidar': lidar_bev, 'target_point': target_point,
                 'target_point_next': target_point_next, 'speed': ego_vel, 'command': command}
        context = (torch.autocast('cuda', dtype=torch.bfloat16)
                   if rgb.is_cuda and self.run_config['precision'] == 'bf16' else nullcontext())
        poses = self._pose.to(rgb.device)[None]
        with context:
            bev, memory, ego = self.perception(batch)
            history, transforms, history_valid = self.history.get_inputs(bev)
            prior, prior_valid, ages = self.rolling.prior([self._scene], poses, [self._timestamp])
            raw = torch.cat((history, bev[:, None]), 1)
            identity = torch.eye(3, device=rgb.device).expand(1, 1, 3, 3)
            world_transforms = torch.cat((transforms, identity), 1)
            frame_valid = torch.cat((history_valid, history_valid.new_ones(1, 1)), 1)
            maps, future_valid, aligned = self.world(raw, world_transforms, frame_valid,
                                                      prior, prior_valid, ego_vel)
            motion = motion_features(aligned, frame_valid, ego_vel, prior, ages)
            condition = self.head.prepare_condition(bev, memory, ego, target_point, history,
                                                    transforms, history_valid, ego_vel, command,
                                                    maps, prior, future_valid, motion)
            sample = self.diffusion.sample(
                lambda noisy, time: self.head(noisy, time, condition)['predictions'][-1],
                1, rgb.device, self.generator, self.run_config['diffusion']['inference_steps'])
            path, delta = self.head.codec.decode(sample)
            speed = self.head.codec.target_speed(delta)
            if not bool(torch.isfinite(path).all() & torch.isfinite(delta).all() & torch.isfinite(speed).all()):
                raise FloatingPointError('V4 produced nonfinite control targets')
            boxes = self.perception.net.head(bev) if self.config.detect_boxes else None
        self.rolling.update([self._scene], poses, [self._timestamp], delta)
        self.last_target_speed = float(speed[0])
        self.last_temporal_delta = delta.float().detach()
        self.last_behavior_probabilities = condition['behavior_logits'].float().softmax(-1).detach()
        self.last_future_valid = future_valid.detach()
        self.last_world_active = maps is not None
        bins = torch.tensor(self.config.target_speeds, device=speed.device, dtype=speed.dtype)
        compatibility_speed = speed.clamp(0, max(self.config.target_speeds))
        upper = torch.searchsorted(bins, compatibility_speed).clamp(1, len(bins) - 1)
        weight = (compatibility_speed - bins[upper - 1]) / (bins[upper] - bins[upper - 1])
        probabilities = torch.zeros(len(speed), len(bins), device=speed.device, dtype=speed.dtype)
        probabilities.scatter_(1, (upper - 1)[:, None], (1 - weight)[:, None])
        probabilities.scatter_(1, upper[:, None], weight[:, None])
        return None, probabilities.clamp_min(1e-12).log(), path.float(), None, None, None, boxes, None, None, None

    def control_pid_direct(self, checkpoints, predicted_class_speed, current_speed):
        # Only the controller adapter resamples to the legacy lattice; the model
        # predicts six spatial points throughout its state, losses and output.
        points = (checkpoints.detach().cpu().float() if torch.is_tensor(checkpoints)
                  else torch.as_tensor(np.asarray(checkpoints), dtype=torch.float32))
        points = spatial_six_to_controller(points).numpy()
        return super().control_pid_direct(points, predicted_class_speed, current_speed)
