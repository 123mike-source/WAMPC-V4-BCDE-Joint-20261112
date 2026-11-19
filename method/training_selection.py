"""Immutable BridgeDrive keyframe selection shared by PDM-Lite training loaders.

Selection applies to current training anchors only. History, future targets
and cache slots retain their original identities. The all-training policy
merges held-out candidates into training and disables validation.
"""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = ROOT / 'configs/training_data.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def keep_keyframe(speed, angles, previous_speed, previous_angles, rng):
    """Compare with the previous raw frame, drawing RNG only for unchanged rows."""
    angles = np.asarray(angles)
    if not np.isfinite(speed) or not np.isfinite(angles).all():
        raise ValueError('Nonfinite expert speed or geometric path angle')
    changed = previous_speed is None or abs(float(speed) - float(previous_speed)) > .1
    if previous_angles is not None:
        difference = angles - previous_angles
        wrapped = np.arctan2(np.sin(difference), np.cos(difference))
        changed = changed or bool(np.any(np.abs(wrapped) > np.deg2rad(.5)))
    return changed or rng.random() < .14


@lru_cache(maxsize=4)
def _load_policy(policy_path, payload):
    policy = json.loads(payload)
    if not policy.get('enabled', False):
        return None
    directory = (ROOT / policy['selection_directory']).resolve()
    manifest_path = directory / 'manifest.json'
    if sha256(manifest_path) != policy['manifest_sha256']:
        raise ValueError('Training selection manifest changed; restore the pinned selection')
    manifest = json.loads(manifest_path.read_text())
    if manifest['format'] != 'pdm_bridgedrive_keyframes_v1':
        raise ValueError('Unsupported training selection format')
    if Path(manifest['data_root']).resolve() != Path(policy['data_root']).resolve():
        raise ValueError('Training selection data root differs from the active policy')
    for name, digest in manifest['files_sha256'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Training selection file changed: {name}')
    routes = json.loads((directory / 'routes.json').read_text())
    train_routes = set(json.loads((directory / 'training_routes.json').read_text()))
    with np.load(io.BytesIO((directory / 'selected_frames.npz').read_bytes()), allow_pickle=False) as archive:
        selected = {name: archive[f'r{rid:05}'].copy()
                    for rid, name in enumerate(routes) if name in train_routes}
    return policy, manifest, selected, hashlib.sha256(payload).hexdigest()


def validation_enabled():
    if not POLICY_FILE.is_file():
        return True
    policy = json.loads(POLICY_FILE.read_bytes())
    return not (policy.get('enabled') and policy.get('split_mode') == 'all_training')


def record_no_validation(progress):
    metrics = {'enabled': False, 'checkpoint_selection': 'final_epoch'}
    progress.update(validation_pending=False, validation=metrics, checkpoint_selection='final_epoch')
    return metrics


def final_checkpoint(output, progress, epochs):
    """Publish final weights only after the complete no-validation training budget."""
    if validation_enabled() or progress['epoch'] < epochs or progress.get('validation_pending', False):
        return
    output = Path(output)
    temporary = output / 'final.partial'
    shutil.copyfile(output / 'latest.pt', temporary)
    temporary.replace(output / 'final.pt')


def load_samples(directory, split):
    """The all-training policy also merges old held-out rows for legacy indexes."""
    directory = Path(directory)
    rows = np.load(directory / f'{split}.npy', mmap_mode='r')
    if validation_enabled():
        return rows
    if split != 'train':
        return rows[:0]
    validation = np.load(directory / 'val.npy', mmap_mode='r')
    if not len(validation):
        return rows
    combined = np.concatenate((rows, validation))
    return np.unique(combined, axis=0)


def context_identities(manifest, routes):
    """Reuse frozen parent contexts only with a pinned manifest and route mapping."""
    identities = [manifest['identity']] * len(routes)
    parent = manifest.get('context_parent')
    if parent is not None:
        directory = Path(parent['directory'])
        if sha256(directory / 'manifest.json') != parent['manifest_sha256']:
            raise ValueError('Parent context index manifest changed')
        original = json.loads((directory / 'manifest.json').read_text())
        parent_routes = json.loads((directory / 'routes.json').read_text())
        count = parent['route_count']
        if original['identity'] != parent['identity'] or len(parent_routes) != count or routes[:count] != parent_routes:
            raise ValueError('Parent context route identities differ')
        identities[:count] = [parent['identity']] * count
    return identities


def filter_samples(samples, routes, data_root, split='train', frame_column=1,
                   slots_directory=None, slots_key='frames'):
    """Return an order-preserving row view; metadata slots are never renumbered.

    slots_directory maps column 1 through a per-route NPZ for indexes storing
    metadata/label slots rather than raw frame IDs. A selected frame is never
    resampled, so applying the policy twice is idempotent.
    """
    if split != 'train':
        return (samples if validation_enabled() else samples[:0]), None
    if not POLICY_FILE.is_file():
        raise FileNotFoundError(f'Required training data policy is missing: {POLICY_FILE}')
    payload = POLICY_FILE.read_bytes()
    policy = json.loads(payload)
    if not policy.get('enabled', False):
        return samples, None
    if Path(data_root).resolve() != Path(policy['data_root']).resolve():
        raise ValueError(f'Training data root differs from the required selection: {data_root}')
    _, manifest, selected, identity = _load_policy(str(POLICY_FILE), payload)
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError('Training rows must be a two-dimensional array')
    mask = np.zeros(len(samples), dtype=bool)
    order = np.argsort(samples[:, 0], kind='stable')
    ordered_routes = samples[order, 0]
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_routes)) + 1, len(order)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start == end:
            continue
        indices = order[start:end]
        rid = int(samples[indices[0], 0])
        if not 0 <= rid < len(routes):
            raise ValueError(f'Invalid route ID {rid}')
        allowed = selected.get(routes[rid])
        if allowed is None:
            continue
        frames = samples[indices, frame_column]
        if slots_directory is not None:
            with np.load(Path(slots_directory) / f'{rid:05}.npz', allow_pickle=False) as archive:
                lookup = archive[slots_key]
                if lookup.ndim != 1 or np.any(frames < 0) or np.any(frames >= len(lookup)):
                    raise ValueError(f'Invalid metadata slots in route {rid}')
                frames = lookup[frames]
        mask[indices] = np.isin(frames, allowed, assume_unique=False)
    result = samples[mask]
    if len(samples) and not len(result):
        raise ValueError('No training rows survived the required BridgeDrive selection')
    report = {'policy_sha256': identity, 'selection_identity': manifest['identity'],
              'selection_directory': str((ROOT / policy['selection_directory']).resolve()),
              'rows_before': len(samples), 'rows_after': len(result),
              'rows_sha256': hashlib.sha256(result.tobytes()).hexdigest(),
              'split': split, 'validation_policy': 'unchanged' if validation_enabled() else 'disabled',
              'support_frames': 'original history and future frames retained'}
    return result, report
