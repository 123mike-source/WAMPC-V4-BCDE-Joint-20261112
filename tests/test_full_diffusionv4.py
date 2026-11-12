"""Numerical contracts for six-point planning, rolling time and fixed-frame BEVs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from method.full_diffusionv4.bootstrap import initialize_from_v3
from method.full_diffusionv4.diffusion import SpatialTemporalCodec, SpatialTemporalVPFullDiffusion
from method.full_diffusionv4.future import FutureNeighborhoodSampler, FutureReadout
from method.full_diffusionv4.geometry import (SPATIAL_SOURCE_INDICES, align_history, motion_features,
                                             spatial_six_to_controller, spatial_ten_to_six)
from method.full_diffusionv4.model import SpatialTemporalFullDiffusionHead
from method.full_diffusionv4.rolling import RollingPlanStore, advance_plan
from method.full_diffusionv4.world import TimedWorldModel, sha256, timed_model_class


torch.set_num_threads(1)


def pose(x=0., y=0., angle=0.):
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, -s, x], [s, c, y], [0., 0., 1.]], dtype=torch.float64)


def feature_grid():
    axis = torch.arange(64).float() + .5 - 32
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    result = torch.zeros(64, 64, 64)
    result[0], result[1] = xx, yy
    return result


def condition_args(batch=1):
    return (torch.randn(batch, 64, 64, 64), torch.randn(batch, 5, 256), torch.randn(batch, 1, 256),
            torch.zeros(batch, 2), torch.randn(batch, 2, 64, 64, 64),
            torch.eye(3).expand(batch, 2, 3, 3).clone(), torch.ones(batch, 2, dtype=torch.bool),
            torch.ones(batch), torch.tensor([[0., 0., 0., 1., 0., 0.]]).expand(batch, -1))


class GeometryTests(unittest.TestCase):
    def test_six_points_preserve_coverage_and_controller_lattice(self):
        points = torch.stack((torch.arange(10).float() + 2.5, torch.zeros(10)), -1)
        six = spatial_ten_to_six(points)
        torch.testing.assert_close(six[:, 0], torch.tensor(SPATIAL_SOURCE_INDICES) + 2.5)
        torch.testing.assert_close(spatial_six_to_controller(six), points)
        self.assertEqual(six.shape, (6, 2))

    def test_five_centisecond_advance_does_not_remove_half_second_node(self):
        delta = torch.stack((torch.arange(1, 7).float(), torch.zeros(6)), -1)
        shifted, valid = advance_plan(delta, pose(), pose(.1), .05)
        self.assertAlmostEqual(float(shifted[0, 0]), 1.1, places=6)
        self.assertEqual(valid.tolist(), [True] * 5 + [False])
        self.assertEqual(shifted[-1].tolist(), [0., 0.])

    def test_rebase_uses_actual_translation_and_rotation(self):
        delta = torch.tensor([[1., 0.]]).expand(6, -1)
        shifted, valid = advance_plan(delta, pose(), pose(.5, angle=np.pi / 2), .25)
        torch.testing.assert_close(shifted[valid], torch.tensor([[0., -1.]]).expand(5, -1), atol=1e-6, rtol=0)

    def test_stopped_actions_and_scene_boundaries(self):
        store = RollingPlanStore()
        store.update(['first'], pose()[None], [1.], torch.zeros(1, 6, 2))
        delta, valid, _ = store.prior(['second', 'first'], pose()[None].expand(2, -1, -1), [1.05, 1.05])
        self.assertFalse(bool(valid[0].any()))
        self.assertTrue(bool(valid[1, :5].all()))
        self.assertFalse(bool(delta.any()))
        _, valid, _ = store.prior(['first'], pose()[None], [0.5])
        self.assertFalse(bool(valid.any()))
        self.assertNotIn('first', store.entries)

    def test_history_alignment_has_correct_axes_and_boundary_mask(self):
        feature = feature_grid()[None, None]
        aligned, valid = align_history(feature, pose(3.)[None, None].float(), torch.ones(1, 1, dtype=torch.bool))
        target = feature.clone()
        target[:, :, 0] -= 3
        selected = valid.expand_as(aligned)
        torch.testing.assert_close(aligned[selected], target[selected], atol=1e-5, rtol=0)
        self.assertFalse(bool(valid.all()))

    def test_motion_descriptor_uses_observations_and_prior_only(self):
        result = motion_features(torch.zeros(2, 3, 64, 64, 64), torch.ones(2, 3, dtype=torch.bool),
                                 torch.ones(2), torch.zeros(2, 6, 2), torch.tensor([.05, .25]))
        self.assertEqual(result.shape, (2, 20))
        self.assertTrue(bool(torch.isfinite(result).all()))


class FutureTests(unittest.TestCase):
    def test_fixed_frame_sampling_centres_budget_radii_and_values(self):
        sampler = FutureNeighborhoodSampler().eval()
        captured = []
        hook = sampler.value.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
        maps = feature_grid()[None, None].expand(1, 6, -1, -1, -1)
        delta = torch.tensor([[[1., 0.]]]).expand(1, 6, -1)
        output = sampler(maps, delta, torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, 256), torch.zeros(1, 20))
        hook.remove()
        centres = [0, 9, 18, 34, 59]
        self.assertEqual(output['tokens'].shape, (1, 95, 256))
        torch.testing.assert_close(output['positions'][0, centres], torch.stack((torch.arange(1, 6).float(), torch.zeros(5)), -1))
        torch.testing.assert_close(captured[0][0, centres, :2], output['positions'][0, centres], atol=2e-6, rtol=0)
        torch.testing.assert_close(output['window_radii'][0, 0], torch.tensor([3., 1.5]))
        torch.testing.assert_close(output['window_radii'][0, -1], torch.tensor([6., 3.]))

    def test_no_valid_future_is_finite_and_exact_bypass(self):
        sampler = FutureNeighborhoodSampler().eval()
        output = sampler(torch.randn(1, 6, 64, 64, 64), torch.zeros(1, 6, 2),
                         torch.zeros(1, 6, dtype=torch.bool), torch.zeros(1, 256), torch.zeros(1, 20))
        readout = FutureReadout()
        readout.gate.data.fill_(1.)
        query = torch.randn(1, 1, 256, requires_grad=True)
        result = readout(query, torch.zeros(1, 6, 2), readout.prepare(output))
        torch.testing.assert_close(result, query, atol=0, rtol=0)
        result.sum().backward()
        self.assertTrue(bool(torch.isfinite(query.grad).all()))

    def test_anchor_stays_exact_under_learned_offsets(self):
        sampler = FutureNeighborhoodSampler().eval()
        sampler.offset.bias.data.fill_(100.)
        delta = torch.tensor([[[1., 0.]]]).expand(1, 6, -1)
        output = sampler(torch.zeros(1, 6, 64, 64, 64), delta, torch.ones(1, 6, dtype=torch.bool),
                         torch.zeros(1, 256), torch.zeros(1, 20))
        torch.testing.assert_close(output['positions'][0, [0, 9, 18, 34, 59]], delta.cumsum(1)[0, :5])
        torch.testing.assert_close(output['window_radii'][0, -1], torch.tensor([9., 4.5]))


class PlannerTests(unittest.TestCase):
    def test_measured_moments_survive_checkpoint_precision_and_reject_other_labels(self):
        from method.full_diffusionv4.data import validate_normalization_artifact
        config = json.loads((Path(__file__).parents[1] / 'method/full_diffusionv4/config.json').read_text())
        moments = {'mean': np.full((12, 2), 0.123456789123).tolist(),
                   'std': np.full((12, 2), 1.23456789123).tolist()}
        head = SpatialTemporalFullDiffusionHead(normalization=moments)
        stored = {'mean': head.codec.mean.tolist(), 'std': head.codec.std.tolist()}
        self.assertNotEqual(moments, stored)
        artifact = {'method': 'full_diffusionv4', 'label_config': config['labels'],
                    'data_root': config['data_root'], 'normalization': moments}
        validate_normalization_artifact(artifact, config, stored)
        stored['mean'][0][0] += .01
        with self.assertRaisesRegex(ValueError, 'actual six-point'):
            validate_normalization_artifact(artifact, config, stored)
        artifact['label_config'] = {**config['labels'], 'temporal_interval_seconds': .25}
        with self.assertRaisesRegex(ValueError, 'labels or data root'):
            validate_normalization_artifact(artifact, config, moments)

    def test_twelve_slot_codec_and_invalid_nan_label_gradient(self):
        codec = SpatialTemporalCodec()
        path, delta = torch.randn(2, 6, 2), torch.randn(2, 6, 2)
        encoded = codec.encode(path, delta)
        self.assertEqual(encoded.shape, (2, 12, 2))
        decoded = codec.decode(encoded)
        torch.testing.assert_close(decoded[0], path)
        torch.testing.assert_close(decoded[1], delta)
        prediction = torch.ones(1, 6, 2, requires_grad=True)
        target = torch.ones_like(prediction)
        target[:, -1] = float('nan')
        mask = torch.tensor([[True] * 5 + [False]])
        codec.masked_l1(prediction, target, mask).backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

    def test_ddim_uses_twenty_five_calls_with_twelve_slots(self):
        diffusion = SpatialTemporalVPFullDiffusion()
        calls = []
        result = diffusion.sample(lambda x, t: calls.append(x.shape) or torch.zeros_like(x), 1, torch.device('cpu'),
                                  torch.Generator().manual_seed(901), 25)
        self.assertEqual(len(calls), 25)
        self.assertEqual(result.shape, (1, 12, 2))

    def test_independent_branches_and_live_future_gradient(self):
        torch.manual_seed(901)
        head = SpatialTemporalFullDiffusionHead(dropout=0).eval()
        maps = torch.randn(1, 6, 64, 64, 64)
        delta = torch.tensor([[[.5, 0.]]]).expand(1, 6, -1)
        condition = head.prepare_condition(*condition_args(), maps, delta,
                                            torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, 20))
        path = delta.cumsum(1)
        noisy = head.codec.encode(path, delta)
        first = head(noisy, torch.tensor([.5]), condition)
        altered = noisy.clone()
        altered[:, :6] += .01
        second = head(altered, torch.tensor([.5]), condition)
        self.assertEqual(first['predictions'].shape, (2, 1, 12, 2))
        torch.testing.assert_close(first['predictions'][:, :, 6:], second['predictions'][:, :, 6:], atol=0, rtol=0)
        first['predictions'].sum().backward()
        gradients = [layer.future.gate.grad for layer in list(head.layers) + list(head.temporal_layers)]
        self.assertTrue(any(value is not None and float(value.abs()) > 0 for value in gradients))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients if value is not None))

    def test_future_sampling_is_outside_denoising_loop(self):
        head = SpatialTemporalFullDiffusionHead(layers=1, dropout=0).eval()
        calls = []
        hook = head.future_sampler.register_forward_hook(lambda *_: calls.append(1))
        delta = torch.tensor([[[.5, 0.]]]).expand(1, 6, -1)
        condition = head.prepare_condition(*condition_args(), torch.zeros(1, 6, 64, 64, 64), delta,
                                            torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, 20))
        with torch.no_grad():
            SpatialTemporalVPFullDiffusion(head.codec).sample(
                lambda x, t: head(x, t, condition)['predictions'][-1], 1, torch.device('cpu'),
                torch.Generator().manual_seed(901), 25)
        hook.remove()
        self.assertEqual(len(calls), 1)

    def test_v3_bootstrap_preserves_temporal_weights_and_remaps_spatial_heads(self):
        from method.full_diffusionv3.model import SpatialTemporalFullDiffusionHead as V3Head
        source = V3Head(dropout=0)
        config = json.loads((Path(__file__).parents[1] / 'method/full_diffusionv4/config.json').read_text())
        config['model']['dropout'] = 0
        migrated, audit = initialize_from_v3({'config': {'method': 'full_diffusionv3',
                                                        'architecture': 'independent_dual_query64'},
                                             'model': source.state_dict()}, config)
        torch.testing.assert_close(migrated.temporal_query_encoder[0].weight, source.temporal_query_encoder[0].weight, atol=0, rtol=0)
        torch.testing.assert_close(migrated.slot_embedding.weight[6:], source.slot_embedding.weight[10:], atol=0, rtol=0)
        self.assertEqual(migrated.layers[0].task_decoder[-1].out_features, 12)
        self.assertEqual(migrated.temporal_layers[0].task_decoder[-1].out_features, 12)
        self.assertTrue(audit['transformed'])
        self.assertEqual(audit['normalization_source'], 'v3_bootstrap_affine')

    def test_route_streams_visit_each_frame_once_in_time_order(self):
        from method.full_diffusionv4.data import RouteStreamBatchSampler
        samples = np.array([[2, 4], [1, 6], [2, 2], [1, 3], [3, 8], [3, 1]])
        batches = list(RouteStreamBatchSampler(samples, 2))
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(len(samples))))
        seen = {}
        for batch in batches:
            self.assertEqual(len(set(int(samples[i, 0]) for i in batch)), len(batch))
            for index in batch:
                route, frame = map(int, samples[index])
                self.assertGreater(frame, seen.get(route, -1))
                seen[route] = frame


class WorldTests(unittest.TestCase):
    def test_teacher_preprocessing_and_coordinate_contract_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / 'teacher.pth'
            teacher.write_bytes(b'test-teacher')
            teacher_config = root / 'config.json'
            teacher_config.write_text('{"test_preprocessing": 1}')
            contract = {'action_shape': [6, 2], 'feature_shape': [6, 64, 64, 64],
                        'times_seconds': [.5, 1., 1.5, 2., 2.5, 3.],
                        'history_times_seconds': [-.5, -.25, 0.],
                        'feature_frame': 'fixed input observation ego frame',
                        'axes': 'width=x forward; height=y right', 'extent_metres': [-32, 32]}
            state = {'config': {'method': 'irwm_bev_t', 'model': {'hidden': 96, 'history': 3, 'path_nodes': 6}},
                     'model': timed_model_class()().state_dict(), 'synthetic_interface_test_only': True,
                     'dataset': {'teacher_sha256': sha256(teacher),
                                 'teacher_config_sha256': sha256(teacher_config),
                                 'query_times_seconds': contract['times_seconds'], 'action_contract': contract}}
            checkpoint = root / 'world.pt'
            torch.save(state, checkpoint)
            world = TimedWorldModel(checkpoint, teacher, allow_synthetic=True)
            self.assertEqual(world.provenance['teacher_config_sha256'], sha256(teacher_config))
            teacher_config.write_text('{"test_preprocessing": 2}')
            with self.assertRaisesRegex(ValueError, 'different frozen perception configs'):
                TimedWorldModel(checkpoint, teacher, allow_synthetic=True)
            state['dataset']['teacher_config_sha256'] = sha256(teacher_config)
            contract['axes'] = 'swapped axes'
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, 'fixed-frame BEV/action contract'):
                TimedWorldModel(checkpoint, teacher, allow_synthetic=True)

    def test_strict_timed_checkpoint_freeze_masks_and_original_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / 'teacher.pth'
            teacher.write_bytes(b'synthetic-teacher-identity-for-unit-test')
            torch.manual_seed(901)
            model = timed_model_class()()
            model.decode[-1].weight.data.normal_(std=1e-4)
            checkpoint = root / 'synthetic_world.pt'
            state = {'config': {'method': 'irwm_bev_t', 'model': {'hidden': 96, 'history': 3, 'path_nodes': 6}},
                     'dataset': {'teacher_sha256': sha256(teacher), 'query_times_seconds': [.5, 1., 1.5, 2., 2.5, 3.]},
                     'model': model.state_dict(), 'synthetic_interface_test_only': True}
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, 'Synthetic interface-test'):
                TimedWorldModel(checkpoint, teacher)
            world = TimedWorldModel(checkpoint, teacher, allow_synthetic=True)
            world.train()
            self.assertFalse(world.training)
            self.assertFalse(any(parameter.requires_grad for parameter in world.parameters()))
            history = torch.randn(1, 3, 64, 64, 64)
            transforms = torch.eye(3).expand(1, 3, 3, 3)
            valid = torch.ones(1, 3, dtype=torch.bool)
            delta = torch.tensor([[[.5, 0.]]]).expand(1, 6, -1).clone()
            active = torch.ones(1, 6, dtype=torch.bool)
            predicted, _, aligned = world(history, transforms, valid, delta, active, torch.ones(1))
            _, pixels = align_history(history, transforms, valid)
            with torch.no_grad():
                expected = model.eval()(aligned, pixels, delta, torch.ones(1))
            torch.testing.assert_close(predicted, expected, atol=0, rtol=0)
            active[:, -1] = False
            first = world(history, transforms, valid, delta, active, torch.ones(1))[0]
            delta[:, -1] = 1e5
            second = world(history, transforms, valid, delta, active, torch.ones(1))[0]
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            teacher.write_bytes(b'different-teacher')
            with self.assertRaisesRegex(ValueError, 'different frozen perception'):
                TimedWorldModel(checkpoint, teacher, allow_synthetic=True)


if __name__ == '__main__':
    unittest.main()
