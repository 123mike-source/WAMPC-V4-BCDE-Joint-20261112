"""Time alignment, cache coverage, causal inputs, real-head gradients and validation."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv3.model import SpatialTemporalFullDiffusionHead
from method.full_diffusionv4.joint_data import DistributedRouteStreams
from method.full_diffusionv5.contract import CONTRACT, INDEX_FORMAT, validate_config, validate_manifest
from method.full_diffusionv5.data import FRAME_KEYS, V5Dataset
from method.full_diffusionv5.model import FutureAdaptedPlanner
from method.full_diffusionv5.prepare import anchor_indices, missing_features, reusable_context
from method.full_diffusionv5.rolling import QuarterSecondPlanStore
from method.full_diffusionv5.train import FreshWorldPlanner, masked_ade, validate

torch.set_num_threads(1)


def config():
    return json.loads((Path(__file__).resolve().parents[1] /
                       'method/full_diffusionv5/config_train4hz.json').read_text())


def pose(x=0., y=0., angle=0.):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, x], [s, c, y], [0., 0., 1.]], dtype=torch.float64)


def batch():
    path = torch.stack((torch.arange(1, 11).float(), torch.zeros(10)), -1)[None]
    delta = torch.tensor([1., 0.]).expand(1, 6, 2).clone()
    return {'raw_bev': torch.zeros(1, 3, 64, 64, 64), 'memory': torch.randn(1, 3, 256),
            'ego': torch.randn(1, 1, 256), 'history_transforms': torch.eye(3).expand(1, 3, 3, 3).clone(),
            'history_mask': torch.ones(1, 3, dtype=torch.bool), 'speed': torch.tensor([2.]),
            'command': torch.tensor([[1., 0., 0., 0., 0., 0.]]), 'target_point': torch.tensor([[8., 0.]]),
            'path': path, 'spatial_mask': torch.ones(1, 10, dtype=torch.bool),
            'temporal_delta': delta, 'temporal_mask': torch.ones(1, 6, dtype=torch.bool),
            'future_positions': delta.cumsum(1), 'behavior_targets': torch.zeros(1, 3, dtype=torch.long),
            'behavior_mask': torch.ones(1, 3, dtype=torch.bool), 'current_pose': pose()[None]}


class ObservationWorld(nn.Module):
    """A cheap substitute for BEV prediction, so these tests need no GPU allocation."""
    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.calls = 0

    def forward(self, history, transforms, history_mask, prior, valid, speed):
        self.calls += 1
        self.last_prior = prior
        if not bool(valid.any()):
            return None, valid, history
        return history[:, -1:].expand(-1, 6, -1, -1, -1) * self.frozen, valid, history


def make_policy():
    torch.manual_seed(901)
    base = SpatialTemporalFullDiffusionHead(layers=1, d_ffn=64, dropout=0.,
                                          normalization={'std': [[1., 1.]] * 16})
    planner = FutureAdaptedPlanner(base, config()['future_sampler'])
    diffusion = SpatialTemporalVPFullDiffusion(base.codec, train_steps=1000)
    return FreshWorldPlanner(planner, ObservationWorld(), diffusion, config()).train()


class ContractAndDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / 'index'
        self.directory.mkdir()
        (self.directory / 'labels').mkdir()
        for folder in ('features', 'context'):
            (self.directory / folder).mkdir()
            (self.directory / folder / 'complete.json').write_text(json.dumps({'dataset_sha256': 'test-index'}))
        self.manifest = {'method': INDEX_FORMAT, 'training_contract': CONTRACT, 'identity': 'test-index',
                         'teacher_cache': str(self.root / 'teacher'), 'label_config': config()['labels']}
        (self.directory / 'manifest.json').write_text(json.dumps(self.manifest))
        (self.directory / 'routes.json').write_text(json.dumps(['route_a']))
        for split in ('train', 'val'):
            np.save(self.directory / f'{split}.npy', np.array([[0, 0, 2], [0, 1, 3]], np.int32))
        sample = batch()
        fields = {k: sample[k][0].numpy()[None].repeat(2, axis=0) for k in FRAME_KEYS}
        fields.update(frame=np.array([2, 3]), history_frames=np.array([[0, 1, 2], [1, 2, 3]]))
        np.savez(self.directory / 'labels/00000.npz', **fields)
        torch.save({'dataset_sha256': 'test-index', 'memory': sample['memory'].repeat(2, 1, 1),
                    'ego': sample['ego'].repeat(2, 1, 1)}, self.directory / 'context/00000.pt')
        (self.root / 'teacher/route_a').mkdir(parents=True)
        for f in range(4):
            values = torch.full((64, 64, 64), float(f), dtype=torch.bfloat16)
            np.save(self.root / 'teacher/route_a' / f'{f:04}.npy', values.view(torch.uint16).numpy())

    def tearDown(self):
        self.temp.cleanup()

    def test_contract_rejects_old_triplets_and_inconsistent_sampling(self):
        settings = config()
        validate_config(settings)
        validate_manifest(self.manifest, settings)
        for key in ('training_contract',):
            wrong = deepcopy(settings)
            del wrong[key]
            with self.assertRaises(ValueError):
                validate_config(wrong)
        wrong = deepcopy(settings)
        wrong['training']['rolling_inference_steps'] = 1
        with self.assertRaises(ValueError):
            validate_config(wrong)
        with self.assertRaises(ValueError):
            validate_manifest({'method': 'full_diffusionv5_index_v1'})

    def test_single_route_chain_includes_both_parities(self):
        dataset = V5Dataset(self.directory)
        np.testing.assert_array_equal(dataset.chains(), [[0, 2], [0, 3]])
        a, b = dataset[0], dataset[1]
        self.assertEqual(a['chain_id'], b['chain_id'])
        self.assertEqual(b['timestamp'] - a['timestamp'], .25)
        self.assertEqual(b['raw_bev'].shape, (3, 64, 64, 64))
        self.assertEqual([float(x.mean()) for x in b['raw_bev']], [1., 2., 3.])
        self.assertEqual(b['current_pose'].shape, (3, 3))

    def test_missing_features_do_not_remove_anchors(self):
        before = np.load(self.directory / 'train.npy').copy()
        (self.root / 'teacher/route_a/0001.npy').unlink()
        missing, required = missing_features(self.directory, self.manifest, [0])
        self.assertEqual(required, 4)
        self.assertEqual(missing, [(0, 1, 1)])
        np.testing.assert_array_equal(np.load(self.directory / 'train.npy'), before)

    def test_cache_identity_is_required(self):
        (self.directory / 'features/complete.json').write_text(json.dumps({'dataset_sha256': 'old'}))
        with self.assertRaises(ValueError):
            V5Dataset(self.directory)

    def test_anchor_indices_keep_last_two_eligible_anchors(self):
        found = anchor_indices(np.arange(40), skip_first=10)
        np.testing.assert_array_equal(found, np.arange(10, 28))
        frames = np.r_[np.arange(20), np.arange(21, 50)]
        found = frames[anchor_indices(frames, skip_first=0)]
        self.assertFalse(bool(((found >= 8) & (found <= 22)).any()))

    def test_context_reuse_matches_raw_frame_numbers(self):
        old = reusable_context((self.directory, 'test-index'), 0)
        np.testing.assert_array_equal(old['frames'], [2, 3])
        self.assertEqual(len(old['memory']), 2)
        with self.assertRaises(ValueError):
            reusable_context((self.directory, 'wrong-index'), 0)

    def test_validation_uses_one_fresh_rollout_per_anchor_and_same_noise(self):
        policy = make_policy()
        settings = config()
        settings['index_directory'] = 'index'
        settings['training']['batch_per_gpu'] = 1
        def cpu_batch(values, device):
            return {k: v.float() if torch.is_tensor(v) and v.dtype == torch.bfloat16 else v
                    for k, v in values.items()}
        with patch('method.full_diffusionv5.train.ROOT', self.root), patch('method.full_diffusionv5.train.move', cpu_batch):
            metrics = validate(policy, settings, torch.device('cpu'))
        self.assertEqual(policy.world.calls, 2)
        self.assertEqual(metrics['frames_all'], 2)
        self.assertGreater(metrics['frames_ready'], 0)
        self.assertEqual(metrics['gain_vs_v3_future_ade_m'], 0.)
        self.assertEqual(metrics['inference_steps'], 25)


class RollingTests(unittest.TestCase):
    def test_quarter_second_interpolation_rebases_to_actual_rotated_pose(self):
        store = QuarterSecondPlanStore()
        delta = torch.tensor([1., 0.]).expand(1, 6, 2).clone().requires_grad_()
        store.update(['route'], pose()[None], [1.], delta)
        prior, valid, age = store.prior(['route'], pose(.5, 0., math.pi / 2)[None], [1.25])
        expected = torch.stack((torch.zeros(5), -torch.arange(1, 6).float()), -1)
        torch.testing.assert_close(prior[0, :5].cumsum(0), expected)
        self.assertEqual(valid.tolist(), [[True] * 5 + [False]])
        self.assertEqual(prior[0, 5].tolist(), [0., 0.])
        self.assertEqual(age.tolist(), [.25])
        self.assertFalse(prior.requires_grad)

    def test_gaps_rewinds_and_wrong_cadence_are_cold_starts(self):
        for now in (1., .75, 1.05, 1.5):
            store = QuarterSecondPlanStore()
            store.update(['route'], pose()[None], [1.], torch.ones(1, 6, 2))
            prior, valid, age = store.prior(['route'], pose()[None], [now])
            self.assertFalse(bool(valid.any()))
            self.assertEqual(float(age.sum()), 0.)
        store.update(['route'], pose()[None], [1.5], torch.ones(1, 6, 2))
        self.assertTrue(bool(store.prior(['route'], pose()[None], [1.75])[1].any()))

    def test_resume_and_batch_reordering_preserve_route_state(self):
        store = QuarterSecondPlanStore()
        store.update(['a', 'b'], pose()[None].repeat(2, 1, 1), [1., 2.], torch.ones(2, 6, 2))
        restored = QuarterSecondPlanStore()
        restored.load_state_dict(store.state_dict())
        for x, y in zip(store.prior(['b', 'a'], pose()[None].repeat(2, 1, 1), [2.25, 1.25]),
                        restored.prior(['b', 'a'], pose()[None].repeat(2, 1, 1), [2.25, 1.25])):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            store.prior(['a', 'a'], pose()[None].repeat(2, 1, 1), [1.25, 1.25])

    def test_ddp_route_pools_are_disjoint_and_continue_at_4hz(self):
        samples = np.array([(route, frame) for route in range(8) for frame in range(6)])
        samplers = [DistributedRouteStreams(samples, 2, rank, 2) for rank in range(2)]
        iterators = [iter(s) for s in samplers]
        previous = {}
        for _ in range(4):
            batches = [next(i) for i in iterators]
            routes = [int(samples[index, 0]) for b in batches for index in b]
            self.assertEqual(len(set(routes)), 4)
            for b in batches:
                for index in b:
                    route, frame = map(int, samples[index])
                    if route in previous:
                        self.assertEqual(frame - previous[route], 1)
                    previous[route] = frame
            for s in samplers:
                s.commit()


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.policy = make_policy()
        self.batch = batch()
        self.prior = torch.tensor([1., 0.]).expand(1, 6, 2).clone()
        self.prior[:, -1] = 0
        self.valid = torch.tensor([[True] * 5 + [False]])
        self.ages = torch.tensor([.25])

    def test_fresh_future_times_and_gate_zero_match_v3_exactly(self):
        future = self.policy.future(self.batch, self.prior, self.valid, self.ages)
        self.assertEqual(future['age'].tolist(), [0.])
        self.assertEqual(future['map_time'].tolist(), [[.5, 1., 1.5, 2., 2.5]])
        self.assertEqual(future['tokens'].shape[1], 95)
        condition = self.policy.condition(self.batch, future)
        noisy, time = torch.randn(1, 16, 2), torch.tensor([.7])
        actual = self.policy.planner(noisy, time, condition)['predictions']
        base = self.policy.planner.base(noisy, time, condition)['predictions']
        torch.testing.assert_close(actual, base, rtol=0, atol=0)

    def test_world_and_sampler_inputs_do_not_depend_on_future_labels(self):
        prior = self.prior.clone().requires_grad_()
        first = self.policy.future(self.batch, prior, self.valid, self.ages)
        changed = deepcopy(self.batch)
        for key in ('path', 'temporal_delta', 'future_positions'):
            changed[key] = changed[key] + 999.
        second = self.policy.future(changed, prior, self.valid, self.ages)
        self.assertFalse(self.policy.world.last_prior.requires_grad)
        torch.testing.assert_close(first['tokens'], second['tokens'], rtol=0, atol=0)

    def test_bad_prior_age_and_fabricated_tail_are_rejected(self):
        with self.assertRaises(ValueError):
            self.policy.future(self.batch, self.prior, self.valid, torch.tensor([.5]))
        with self.assertRaises(ValueError):
            self.policy.future(self.batch, self.prior, torch.ones_like(self.valid), self.ages)

    def test_train_backward_and_25_step_prior_keep_base_and_world_frozen(self):
        for branch in ('spatial', 'temporal'):
            self.policy.planner.adapter[branch][0].gate.data.fill_(.1)
        # A covered low-noise candidate isolates gradient connectivity from the
        # readout's intentional rejection of far-away noisy trajectories.
        with patch.object(self.policy.diffusion, 'corrupt',
                          side_effect=lambda clean, steps: (clean, torch.full((len(clean),), .1))):
            result = self.policy(self.batch, self.prior, self.valid, self.ages,
                                 torch.Generator().manual_seed(901))
        self.assertTrue(bool(torch.isfinite(result['loss'])))
        result['loss'].backward()
        for name, p in self.policy.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
            else:
                self.assertIsNone(p.grad, name)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.policy.parameters() if p.requires_grad), 0.)
        self.assertFalse(result['sampled_delta'].requires_grad)
        self.assertEqual(result['sampled_delta'].shape, (1, 6, 2))
        self.assertEqual(float(result['valid_prior_points']), 5.)
        self.assertFalse(self.policy.planner.base.training)
        self.assertFalse(self.policy.world.training)

    def test_cold_start_has_zero_loss_but_produces_a_detached_bootstrap_plan(self):
        result = self.policy(self.batch, torch.zeros_like(self.prior), torch.zeros_like(self.valid),
                             torch.zeros_like(self.ages), torch.Generator().manual_seed(901))
        self.assertEqual(float(result['loss']), 0.)
        result['loss'].backward()
        self.assertTrue(all(p.grad is not None for p in self.policy.parameters() if p.requires_grad))
        self.assertFalse(result['sampled_delta'].requires_grad)
        self.assertTrue(bool(torch.isfinite(result['sampled_delta']).all()))

    def test_ade_ignores_masked_future_targets(self):
        value = masked_ade(torch.tensor([[[3., 4.], [999., 999.]]]), torch.zeros(1, 2, 2),
                           torch.tensor([[True, False]]))
        self.assertEqual(float(value), 5.)


if __name__ == '__main__':
    unittest.main()
