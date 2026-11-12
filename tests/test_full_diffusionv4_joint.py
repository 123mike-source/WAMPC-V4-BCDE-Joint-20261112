"""Partial tuning, recursive gradients, and consumed stream resume contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from method.full_diffusionv4.batched_rolling import BatchedRollingPlanStore
from method.full_diffusionv4.geometry import align_history
from method.full_diffusionv4.joint_data import DistributedRouteStreams
from method.full_diffusionv4.rolling import RollingPlanStore
from method.full_diffusionv4.world import sha256
from method.full_diffusionv4.world_tuning import PartialTimedWorld, TUNING, load_tuned_state
from method.irwm_bev_t_large import ARCHITECTURE
from method.irwm_bev_t_large.model import ActionBEVWorldModel

torch.set_num_threads(1)
SMALL = {'hidden': 16, 'history': 3, 'path_nodes': 6, 'skip_channels': 8, 'coarse_hidden': 32,
         'history_residuals': 1, 'transition_residuals': 2, 'coarse_residuals': 1, 'decoder_residuals': 1}


def pose(x=0., y=0., angle=0.):
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, -s, x], [s, c, y], [0., 0., 1.]], dtype=torch.float64)


class WorldTuningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(901)
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        teacher = root / 'teacher.pt'
        teacher.write_bytes(b'synthetic test teacher')
        model = ActionBEVWorldModel(**SMALL).eval()
        model.decode.output.weight.data.normal_(std=1e-3)
        path = root / 'world.pt'
        torch.save({'model': model.state_dict(), 'config': {'method': 'irwm_bev_t_large',
                    'architecture': ARCHITECTURE, 'model': SMALL},
                    'dataset': {'teacher_sha256': sha256(teacher),
                                'query_times_seconds': [.5, 1., 1.5, 2., 2.5, 3.]},
                    'synthetic_interface_test_only': True}, path)
        self.world = PartialTimedWorld(path, teacher, allow_synthetic=True, activation_checkpoint=False)
        self.history = torch.randn(1, 3, 64, 64, 64)
        self.transforms = torch.eye(3).expand(1, 3, 3, 3).clone()
        self.history_valid = torch.ones(1, 3, dtype=torch.bool)
        self.delta = torch.tensor([[[.5, .1]]]).expand(1, 6, 2).clone()
        self.valid = torch.tensor([[True] * 5 + [False]])
        self.speed = torch.ones(1)

    def tearDown(self):
        self.temp.cleanup()

    def run_world(self):
        return self.world(self.history, self.transforms, self.history_valid, self.delta, self.valid, self.speed)[0]

    def test_same_five_maps_as_original_fp32_recurrence(self):
        with torch.no_grad():
            aligned, mask = align_history(self.history, self.transforms, self.history_valid)
            delta = torch.where(self.valid[..., None], self.delta, 0.)
            path, times = delta.cumsum(1), self.world.model.times[None]
            original = self.world.model._evolve(aligned, mask, path, times, self.valid, path, times,
                                                 self.valid, self.speed, delta[:, 1].norm(dim=-1) / .5)
            tuned = self.run_world()
        torch.testing.assert_close(tuned, original, atol=0, rtol=0)
        self.world.train()
        self.assertFalse(self.world.training)
        self.assertFalse(self.world.model.training)

    def test_last_horizon_backpropagates_through_earlier_states_only_selected_parameters(self):
        states = []
        step = self.world.step
        def capture(*args):
            result = step(*args)
            result[0].retain_grad()
            result[1].retain_grad()
            states.append(result)
            return result
        self.world.step = capture
        output = self.run_world()
        output[:, 4].square().mean().backward()
        self.assertEqual(len(states), 5)
        for i, (state, prediction) in enumerate(states):
            if i < 4:
                self.assertGreater(float(state.grad.abs().sum()), 0.)
            self.assertGreater(float(prediction.grad.abs().sum()), 0.)
        for name, parameter in self.world.named_parameters():
            self.assertEqual(parameter.requires_grad, name in self.world.tuned_names)
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            else:
                self.assertIsNone(parameter.grad, name)
        for name in ('model.action.0.weight', 'model.transition.1.net.2.weight', 'model.decode.output.weight'):
            self.assertGreater(float(dict(self.world.named_parameters())[name].grad.abs().sum()), 0.)

    def test_checkpoint_and_direct_gradients_match(self):
        first = self.run_world()
        first[:, :5].square().mean().backward()
        grads = {n: p.grad.clone() for n, p in self.world.named_parameters() if p.requires_grad}
        self.world.zero_grad(set_to_none=True)
        self.world.activation_checkpoint = True
        second = self.run_world()
        second[:, :5].square().mean().backward()
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        for name, parameter in self.world.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, grads[name], atol=1e-7, rtol=1e-5)

    def test_tuned_state_roundtrip_and_frozen_weights_stay_identical(self):
        original = {n: p.detach().clone() for n, p in self.world.named_parameters()}
        self.run_world()[:, 4].square().mean().backward()
        torch.optim.AdamW([p for p in self.world.parameters() if p.requires_grad], lr=1e-3).step()
        exported = self.world.export_tuned()
        self.assertEqual(set(exported), self.world.tuned_names)
        for name, parameter in self.world.named_parameters():
            if name not in exported:
                torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            load_tuned_state(self.world, {}, TUNING)
        bad = deepcopy(exported)
        bad[next(iter(bad))] = torch.zeros(1)
        with self.assertRaises(ValueError):
            load_tuned_state(self.world, bad, TUNING)
        for n, p in self.world.named_parameters():
            if n in exported:
                p.data.zero_()
        load_tuned_state(self.world, exported, TUNING)
        for n, p in self.world.named_parameters():
            if n in exported:
                torch.testing.assert_close(p, exported[n], atol=0, rtol=0)


class StreamTests(unittest.TestCase):
    def test_rank_pools_are_disjoint_and_prefetch_resume_uses_consumed_cursor(self):
        samples = np.asarray([(r, f) for r in range(20) for f in range(5)])
        streams = [DistributedRouteStreams(samples, 2, rank, 2, 901) for rank in range(2)]
        pools = [set(r for stream in s.initial() for r in stream['pool']) for s in streams]
        self.assertFalse(pools[0] & pools[1])
        self.assertEqual(pools[0] | pools[1], set(range(20)))
        sampler = streams[0]
        iterator = iter(sampler)
        first, second, third = next(iterator), next(iterator), next(iterator)
        sampler.commit()
        state = sampler.state_dict()
        restored = DistributedRouteStreams(samples, 2, 0, 2, 901)
        restored.load_state_dict(state)
        resumed = iter(restored)
        self.assertEqual(next(resumed), second)
        self.assertEqual(next(resumed), third)
        for a, b in zip(first, second):
            self.assertEqual(samples[a, 0], samples[b, 0])
            self.assertEqual(samples[b, 1], samples[a, 1] + 1)

    def test_batched_pose_interpolation_matches_existing_store_and_resets(self):
        original, batched = RollingPlanStore(), BatchedRollingPlanStore()
        start = torch.stack([pose(), pose(3., 1., .2), pose(1., 2., -.4)])
        current = torch.stack([pose(.3, .1, .1), pose(3.5, 1.1, .3), pose(1., 2., -.4)])
        delta = torch.randn(3, 6, 2)
        scenes = ['a', 'b', 'c']
        for store in (original, batched):
            store.update(scenes, start, torch.tensor([1., 1., 1.]), delta)
        a = original.prior(scenes, current, torch.tensor([1.25, 1.5, .5]))
        b = batched.prior(scenes, current, torch.tensor([1.25, 1.5, .5]))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
        self.assertNotIn('c', batched.entries)


if __name__ == '__main__':
    unittest.main()
