"""Run on the allocated node after the finite DDP/resume check."""
import json
from pathlib import Path
import torch
import numpy as np
from method.full_diffusion.data import read_measurement
from method.full_diffusion2.data import frame_pose
from method.full_diffusionv3.data import future_motion_targets

root = Path('/share/home/u21012/nixiran/WAMPC')
runtime = root / '.runtime/v4_joint_train_20261112'
output = root / 'results/full_diffusionv4_joint_20261112/ddp_smoke'
state = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
assert state['diagnostic_only'] and state['requires_training']
assert state['progress']['updates'] == 9
assert len(state['rank_states']) == 8
assert all(s['sampler']['consumed'] == 9 for s in state['rank_states'])
checksums = [s['parameter_checksum'] for s in state['rank_states']]
assert all(s == checksums[0] for s in checksums)
original = torch.load(root / state['config']['world']['checkpoint'], map_location='cpu', weights_only=False)
tuned = state['world_tuned_state']
assert all(torch.isfinite(p).all() for p in tuned.values())
changed = [n for n, p in tuned.items() if not torch.equal(p, original['model'][n.removeprefix('model.')])]
assert changed, 'No trained world parameter changed'
log = (runtime / 'final_check.log').read_text()
assert 'Grad strides do not match' not in log, 'Gradient layout still needs repair'
assert 'Traceback' not in log
manifest = json.loads((root / state['config']['index_directory'] / 'manifest.json').read_text())
assert manifest['identity'] == state['dataset']['prepared_identity']
for name in ('mean', 'std'):
    torch.testing.assert_close(state['model']['codec.' + name], torch.tensor(manifest['normalization'][name]), atol=0, rtol=0)
prepared = root / state['config']['index_directory']
routes = json.loads((prepared / 'routes.json').read_text())
settings = {k: v for k, v in state['config']['labels'].items() if k != 'spatial_source_indices'}
max_label_error, max_transform_error = 0., 0.
for split in ('train', 'val'):
    samples = np.load(prepared / (split + '.npy'))
    for index in np.linspace(0, len(samples) - 1, 4, dtype=int):
        route, slot = map(int, samples[index])
        with np.load(prepared / 'labels' / f'{route:05}.npz') as label:
            frame = int(label['frame'][slot])
            directory = Path(manifest['data_root']) / routes[route] / 'measurements'
            current = read_measurement(directory / f'{frame:04}.json.gz')
            future = {k: read_measurement(directory / f'{frame+k:04}.json.gz') for k in range(2, 13, 2)}
            direct = future_motion_targets(current, future, settings)
            for name in ('temporal_delta', 'future_positions', 'behavior_targets', 'temporal_mask', 'behavior_mask'):
                if direct[name].dtype.kind in 'biu':
                    np.testing.assert_array_equal(direct[name], label[name][slot])
                else:
                    error = float(np.max(np.abs(direct[name] - label[name][slot])))
                    max_label_error = max(max_label_error, error)
                    np.testing.assert_allclose(direct[name], label[name][slot], atol=2e-4, rtol=0)
            for h, old_frame in enumerate(label['history_frames'][slot]):
                observed = read_measurement(directory / f'{int(old_frame):04}.json.gz')
                direct_transform = np.linalg.solve(frame_pose(current), frame_pose(observed))
                max_transform_error = max(max_transform_error, float(np.max(np.abs(direct_transform - label['history_transforms'][slot, h]))))
                np.testing.assert_allclose(direct_transform, label['history_transforms'][slot, h], atol=2e-4, rtol=0)
result = {'status': 'PASS', 'updates': 9, 'resumed_from_updates': [5, 6, 8], 'ddp_ranks': 8,
          'consumed_cursor': [s['sampler']['consumed'] for s in state['rank_states']],
          'identical_rank_parameter_checksums': True, 'gradient_layout_warning': False,
          'world_tensors': len(tuned), 'world_trainable_parameters': sum(p.numel() for p in tuned.values()),
          'changed_world_tensors': len(changed), 'prepared_identity': manifest['identity'],
          'raw_measurement_checked_samples': 8, 'max_label_error_m': max_label_error,
          'max_observed_transform_error': max_transform_error,
          'diagnostic_weights_discarded_for_fresh_formal_run': True}
(runtime / 'audit_start.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
