"""BCDE numerical checks: real model geometry, gradients, loss masks, and recovery."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from method.irwm_bev_t_large import ARCHITECTURE
from method.irwm_bev_t_large.model import ActionBEVWorldModel, LegacyModel
from method.irwm_bev_t_large.loss import error_metrics, loss_components
from method.irwm_bev_t_large.train import update, save_checkpoint, load_checkpoint, check_config
from method.irwm_bev_t_large.data import prepare, SequenceDataset, collate
from method.irwm_bev_t_large.runtime import configure_worker_tmp
from method.full_diffusionv4.world import TimedWorldModel, sha256, timed_model_class

torch.set_num_threads(1)
SMALL = {'hidden': 16, 'history': 3, 'path_nodes': 6, 'skip_channels': 8, 'coarse_hidden': 32,
         'history_residuals': 1, 'transition_residuals': 2, 'coarse_residuals': 1, 'decoder_residuals': 1}


def config():
    return json.loads((Path(__file__).parents[1] / 'method/irwm_bev_t_large/config.json').read_text())


def inputs():
    return {'history': torch.randn(1, 3, 64, 64, 64),
            'history_valid': torch.ones(1, 3, 1, 64, 64, dtype=torch.bool),
            'temporal_delta': torch.tensor([[[.5, 0.]]]).expand(1, 6, -1).clone(),
            'speed': torch.ones(1)}


def supervision(current=None):
    current = torch.ones(1, 64, 64, 64) if current is None else current
    return {'target': current[:, None].expand(-1, 6, -1, -1, -1).clone(),
            'spatial_valid': torch.ones(1, 6, 1, 64, 64, dtype=torch.bool),
            'query_valid': torch.ones(1, 6, dtype=torch.bool),
            'corridor': torch.ones(1, 1, 64, 64),
            'current': current, 'current_valid': torch.ones(1, 1, 64, 64, dtype=torch.bool)}


class ModelTests(unittest.TestCase):
    def test_default_192_pyramid_six_steps_and_deep_gradients(self):
        torch.manual_seed(901)
        model = ActionBEVWorldModel()
        self.assertEqual(sum(p.numel() for p in LegacyModel().parameters()), 1916320)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 6578944)
        observed = {}
        hooks = []
        for name, layer in [('high', model.history_encoder.high), ('middle', model.history_encoder.middle),
                            ('coarse', model.history_encoder.coarse), ('decode_low', model.decode.low)]:
            def hook(_, args, output, key=name):
                observed.setdefault(key, []).append(tuple(output.shape))
            hooks.append(layer.register_forward_hook(hook))
        batch = inputs()
        with torch.no_grad():
            copied = model(**batch)
        torch.testing.assert_close(copied, batch['history'][:, -1, None].expand_as(copied), atol=0, rtol=0)
        self.assertEqual(observed['high'], [(1, 64, 64, 64)])
        self.assertEqual(observed['middle'], [(1, 192, 32, 32)])
        self.assertEqual(observed['coarse'], [(1, 384, 16, 16)])
        self.assertEqual(observed['decode_low'], [(1, 64, 32, 32)] * 6)
        for hook in hooks:
            hook.remove()
        model.decode.output.weight.data.normal_(std=1e-3)
        prediction = model(**batch)
        self.assertEqual(prediction.shape, (1, 6, 64, 64, 64))
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        loss = loss_components(prediction, **supervision(batch['history'][:, -1]))['loss']
        loss.backward()
        for parameter in [model.history_encoder.high[0].weight, model.history_encoder.middle[-1].net[2].weight,
                          model.history_encoder.coarse[0].weight, model.transition[-1].net[2].weight,
                          model.decode.low[0].weight, model.action[0].weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_actions_change_prediction_and_invalid_tail_is_ignored(self):
        torch.manual_seed(901)
        model = ActionBEVWorldModel(**SMALL).eval()
        model.decode.output.weight.data.normal_(std=1e-3)
        batch = inputs()
        with torch.no_grad():
            first = model(**batch)
            altered = {**batch, 'temporal_delta': batch['temporal_delta'] + torch.tensor([0., .5])}
            second = model(**altered)
            self.assertGreater(float((first - second).abs().max()), 0.)
            path = batch['temporal_delta'].cumsum(1)
            valid = torch.tensor([[True] * 5 + [False]])
            times = model.times[None]
            common = (batch['history'], batch['history_valid'])
            first = model._evolve(*common, path, times, valid, path, times, valid, batch['speed'], batch['speed'])
            path[:, -1] = float('nan')
            second = model._evolve(*common, path, times, valid, path, times, valid, batch['speed'], batch['speed'])
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            torch.testing.assert_close(first[:, -1], first[:, -2], atol=0, rtol=0)

    def test_width_constraints(self):
        with self.assertRaises(ValueError):
            ActionBEVWorldModel(hidden=190)


class LossTests(unittest.TestCase):
    def test_perfect_static_prediction_has_zero_all_objectives(self):
        values = supervision()
        result = loss_components(values['target'].clone(), **values)
        self.assertAlmostEqual(float(result['loss']), 0., places=6)
        self.assertEqual(float(result['dynamic_nmse']), 0.)
        self.assertEqual(float(result['temporal_nmse']), 0.)

    def test_dynamic_region_error_has_greater_dynamic_penalty(self):
        values = supervision()
        values['target'][:, :, :, 8:10, 8:10] += 1
        dynamic = values['target'].clone()
        background = values['target'].clone()
        dynamic[:, 0, :, 8:10, 8:10] += 1
        background[:, 0, :, 20:22, 20:22] += 1
        a = error_metrics(dynamic, **values)
        b = error_metrics(background, **values)
        torch.testing.assert_close(a['nmse'], b['nmse'])
        self.assertGreater(float(a['dynamic_nmse'][0, 0]), float(b['dynamic_nmse'][0, 0]))

    def test_temporal_change_detects_copy_current_baseline(self):
        values = supervision(torch.zeros(1, 64, 64, 64))
        values['target'] += torch.arange(1, 7)[None, :, None, None, None]
        perfect = error_metrics(values['target'].clone(), **values)
        copied = error_metrics(torch.zeros_like(values['target']), **values)
        self.assertEqual(float(perfect['temporal_nmse'].sum()), 0.)
        self.assertTrue(bool((copied['temporal_nmse'] > 0).all()))

    def test_multiscale_supervision_separates_low_and_high_frequency_error(self):
        values = supervision()
        checker = ((torch.arange(64)[:, None] + torch.arange(64)[None]) % 2 * 2 - 1).float()
        high = values['target'] + .1 * checker
        low = values['target'] + .1
        a = error_metrics(high, **values)
        b = error_metrics(low, **values)
        torch.testing.assert_close(a['nmse'], b['nmse'], atol=1e-7, rtol=1e-5)
        self.assertLess(float(a['multiscale_nmse'].max()), 1e-10)
        self.assertGreater(float(b['multiscale_nmse'].min()), .009)

    def test_masked_nan_targets_never_leak_into_losses_or_gradients(self):
        values = supervision()
        values['spatial_valid'][:, :, :, :4] = False
        values['current_valid'][:, :, :4] = False
        values['query_valid'][:, -1] = False
        values['target'][:, :, :, :4] = float('nan')
        values['target'][:, -1] = float('nan')
        values['current'][:, :, :4] = float('nan')
        prediction = torch.ones_like(values['target'], requires_grad=True)
        result = loss_components(prediction, **values)
        self.assertTrue(all(bool(torch.isfinite(value)) for value in result.values()))
        result['loss'].backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertFalse(bool(prediction.grad[:, -1].any()))
        values['spatial_valid'].zero_()
        self.assertEqual(float(loss_components(prediction, **values)['loss']), 0.)

    def test_auxiliary_weights_are_live_in_training_objective(self):
        values = supervision()
        values['target'][:, :, :, 8:10, 8:10] += torch.arange(1, 7)[None, :, None, None, None]
        prediction = torch.ones_like(values['target'], requires_grad=True)
        base = loss_components(prediction, **values, settings={'dynamic_nmse': 0., 'temporal_nmse': 0., 'multiscale_nmse': 0.})
        full = loss_components(prediction, **values)
        self.assertGreater(float(full['loss']), float(base['loss']))
        grad_base = torch.autograd.grad(base['loss'], prediction, retain_graph=True)[0]
        grad_full = torch.autograd.grad(full['loss'], prediction)[0]
        self.assertGreater(float((grad_full - grad_base).abs().sum()), 0.)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_worker_tmp()

    def test_cached_dataset_loads_in_spawn_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / 'index'
            (index / 'metadata').mkdir(parents=True)
            manifest = {'teacher_sha256': 'a' * 64, 'teacher_config_sha256': 'b' * 64,
                        'query_times_seconds': [.5, 1., 1.5, 2., 2.5, 3.]}
            (index / 'manifest.json').write_text(json.dumps(manifest))
            (index / 'routes.json').write_text(json.dumps(['synthetic_route']))
            np.save(index / 'train.npy', np.array([[0, 2]]))
            np.savez(index / 'metadata/00000.npz', frames=np.arange(15),
                     poses=np.c_[np.arange(15) * .25, np.zeros((15, 2))], speed=np.ones(15))
            cache = root / 'cache' / ('a' * 16 + '_' + 'b' * 12 + '_bf16_v1')
            (cache / 'synthetic_route').mkdir(parents=True)
            (cache / 'complete.json').write_text(json.dumps({'missing_frames': 0, 'teacher_sha256': 'a' * 64}))
            for frame in [0, 1, 2, 4, 6, 8, 10, 12, 14]:
                np.save(cache / 'synthetic_route' / f'{frame:04}.npy', np.full((64, 64, 64), 0x3f80, dtype=np.uint16))
            dataset = SequenceDataset(index, 'train', root / 'cache')
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1,
                                                  multiprocessing_context='spawn', collate_fn=collate, timeout=60)
            batches = list(loader)
            self.assertEqual(len(batches), 1)
            batch = batches[0]
            self.assertEqual(len(batch['sensors'][0]), 9)
            self.assertTrue(bool((batch['sensors'][0][0]['bev'] == 1).all()))
            torch.testing.assert_close(batch['temporal_delta'][0], torch.tensor([[.5, 0.]]).expand(6, -1))
            del batch, batches, loader, dataset
            gc.collect()

    def test_future_teacher_changes_only_supervision(self):
        features = torch.randn(1, 9, 64, 64, 64)
        delta = torch.tensor([[[.5, 0.]]]).expand(1, 6, -1)
        path = delta.cumsum(1)
        batch = {'sensors': None, 'source_in_anchor': torch.zeros(1, 9, 3),
                 'temporal_delta': delta, 'speed': torch.ones(1), 'path': path,
                 'path_valid': torch.ones(1, 6, dtype=torch.bool), 'query_xy': path,
                 'query_valid': torch.ones(1, 6, dtype=torch.bool), 'route_id': torch.zeros(1, dtype=torch.long)}
        first_inputs, first_truth, _ = prepare(batch, lambda _: features, torch.device('cpu'))
        features = features.clone()
        features[:, 3:] += 10
        second_inputs, second_truth, _ = prepare(batch, lambda _: features, torch.device('cpu'))
        self.assertEqual(set(first_inputs), {'history', 'history_valid', 'temporal_delta', 'speed'})
        for key in first_inputs:
            torch.testing.assert_close(first_inputs[key], second_inputs[key], atol=0, rtol=0)
        self.assertGreater(float((first_truth['target'] - second_truth['target']).abs().mean()), 9.)

    def test_v4_loader_dispatch_strict_weights_and_synthetic_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / 'teacher.pth'
            teacher.write_bytes(b'synthetic-teacher-for-BCDE-test')
            cfg = config()
            cfg['model'] = SMALL
            model = ActionBEVWorldModel(**SMALL)
            checkpoint = root / 'world.pt'
            state = {'config': cfg, 'model': model.state_dict(),
                     'dataset': {'teacher_sha256': sha256(teacher), 'query_times_seconds': [.5, 1., 1.5, 2., 2.5, 3.]},
                     'synthetic_interface_test_only': True}
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, 'Synthetic'):
                TimedWorldModel(checkpoint, teacher)
            world = TimedWorldModel(checkpoint, teacher, allow_synthetic=True)
            self.assertIsInstance(world.model, ActionBEVWorldModel)
            self.assertFalse(any(p.requires_grad for p in world.parameters()))
            self.assertEqual(timed_model_class()().hidden, 96)
            with self.assertRaises(ValueError):
                timed_model_class('irwm_bev_t_large', 'unrecognized')
            batch = inputs()
            maps, valid, _ = world(batch['history'], torch.eye(3).expand(1, 3, 3, 3),
                                    torch.ones(1, 3, dtype=torch.bool), batch['temporal_delta'],
                                    torch.tensor([[True] * 5 + [False]]), batch['speed'])
            self.assertEqual(maps.shape, (1, 6, 64, 64, 64))
            self.assertFalse(bool(valid[0, -1]))

    def test_actual_update_checkpoint_resume_reproduces_next_update(self):
        torch.manual_seed(901)
        cfg = config()
        cfg.update(world_size=1, batch_per_gpu=32, workers_per_rank=0)
        cfg['model'] = SMALL
        check_config(cfg, 1)
        with self.assertRaises(ValueError):
            check_config(cfg, 2)
        model = ActionBEVWorldModel(**SMALL)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        batch = inputs()
        values = supervision(batch['history'][:, -1])
        values['target'] += .1
        prepared = [(batch, values, torch.zeros(1, dtype=torch.long))]
        device = torch.device('cpu')
        update(prepared, model, optimizer, scheduler, cfg, device)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = {'epoch': 0, 'next_batch': 1, 'updates': 1, 'best_nmse': None, 'validation_pending': False}
            save_checkpoint(root, model, optimizer, scheduler, progress, cfg, {}, device)
            update(prepared, model, optimizer, scheduler, cfg, device)
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            recovered = load_checkpoint(root / 'latest.pt', model, optimizer, scheduler, cfg, {}, device)
            self.assertEqual(recovered['updates'], 1)
            update(prepared, model, optimizer, scheduler, cfg, device)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
            changed = deepcopy(cfg)
            changed['loss']['temporal_nmse'] = .9
            with self.assertRaisesRegex(ValueError, 'Resume requires'):
                load_checkpoint(root / 'latest.pt', model, optimizer, scheduler, changed, {}, device)


if __name__ == '__main__':
    unittest.main()
