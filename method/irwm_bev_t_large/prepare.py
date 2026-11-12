"""Portable reconstruction of the original quality-filtered, family-isolated index."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_reference(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def reconstruct_split(routes, reference, official, seed):
    test_triggers = []
    for r in ET.parse(official).getroot().findall('route'):
        for s in r.findall('./scenarios/scenario'):
            t = s.find('trigger_point')
            if t is not None:
                test_triggers.append((r.attrib['town'], s.attrib['type'], float(t.attrib['x']), float(t.attrib['y'])))
    if not test_triggers:
        raise ValueError('Official route scenario triggers are missing')
    template_root = ROOT / 'third_party/carla_garage/data'
    template_dirs = {town: list(template_root.glob(f'*_{town}')) for town in ('Town12', 'Town13')}
    verified, test_families, unknown = set(), set(), []
    for name in routes:
        fam = reference.family(name)
        scenario, town, ident, _ = fam.split(':')
        weather = re.match(r'Town\d+_Rep\d+_.+_(\d+)_route', name.split('/')[1]).group(1)
        candidates = [d / scenario / f'{ident}_{weather}.xml' for d in template_dirs.get(town, [])]
        template = next((x for x in candidates if x.exists()), None)
        if template is None:
            unknown.append(name)
            continue
        rs = ET.parse(template).getroot().find('route')
        verified.add(fam)
        for s in rs.findall('./scenarios/scenario'):
            t = s.find('trigger_point')
            if t is None:
                continue
            x, y = float(t.attrib['x']), float(t.attrib['y'])
            if any(tt == town and st == s.attrib['type'] and (x - tx)**2 + (y - ty)**2 < 25
                   for tt, st, tx, ty in test_triggers):
                test_families.add(fam)
    eligible = [r for r in routes if reference.family(r) not in test_families]
    by_scenario = defaultdict(list)
    for r in eligible:
        if reference.family(r) in verified:
            by_scenario[r.split('/')[0]].append(r)
    key = lambda r: hashlib.sha256(f'{seed}:{r}'.encode()).hexdigest()
    selected, selected_families = [], set()
    for scenario, names in sorted(by_scenario.items()):
        chosen = min(names, key=key)
        selected.append(chosen)
        selected_families.add(reference.family(chosen))
    for r in sorted((r for r in eligible if reference.family(r) in verified), key=key):
        if len(selected) >= 44:
            break
        if reference.family(r) not in selected_families:
            selected.append(r)
            selected_families.add(reference.family(r))
    if len(selected) != 44:
        raise ValueError(f'Expected 44 validation routes, got {len(selected)}')
    val_names = set(selected)
    train_names = {r for r in eligible if reference.family(r) not in selected_families}
    if {reference.family(r) for r in train_names} & selected_families:
        raise ValueError('Train/validation family overlap')
    split = {'seed': seed, 'validation_routes': sorted(selected),
             'validation_families': sorted(selected_families),
             'excluded_validation_variants': sorted(set(eligible) - train_names - val_names),
             'excluded_official_test_families': sorted(test_families),
             'official_overlap_rule': 'same town/scenario and trigger within 5 m; exclude whole family',
             'unmapped_source_templates': unknown,
             'validation_scenarios': dict(Counter(r.split('/')[0] for r in selected))}
    return sorted(train_names | val_names), train_names, val_names, split


def metadata_task(task):
    rid, name, data_root, output, skip, stride = task
    destination = Path(output) / 'metadata' / f'{rid:05}.npz'
    if not destination.exists():
        # Execute the original metadata extraction and anchor rules without editing them.
        reference = load_reference('original_irwm_prepare', ROOT / 'method/irwm_bev/prepare.py')
        reference.route_metadata(task)
    with np.load(destination) as arrays:
        frames = arrays['frames']
    original = np.zeros(len(frames), dtype=bool)
    if len(frames) >= 7:
        continuous = (np.diff(np.lib.stride_tricks.sliding_window_view(frames, 7), axis=1) == 1).all(1)
        original[2:len(frames)-4] = continuous
    original &= (frames >= skip) & (frames % stride == 0)
    good = np.zeros(len(frames), dtype=bool)
    if len(frames) >= 15:
        windows = np.lib.stride_tricks.sliding_window_view(frames, 15)
        good[2:len(frames)-12] = (np.diff(windows, axis=1) == 1).all(1)
    anchors = np.flatnonzero(original & good)
    positions = (anchors[:, None] + np.array([-2, -1, 0, 2, 4, 6, 8, 10, 12])).reshape(-1)
    required = np.unique(frames[positions])
    return rid, anchors, len(frames), int(original.sum()), required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--seed', type=int, default=901)
    parser.add_argument('--stride', type=int, default=2)
    args = parser.parse_args()
    out, data_root = args.output.resolve(), args.data_root.resolve()
    out.relative_to(ROOT / 'data')
    if (out / 'manifest.json').exists():
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['data_root'] != str(data_root) or manifest['seed'] != args.seed:
            raise ValueError('Existing index has different inputs')
        for name, key in [('split.json', 'split_sha256'), ('train.npy', 'train_index_sha256'), ('val.npy', 'val_index_sha256'), ('required_frames.npy', 'required_frames_sha256')]:
            if sha(out / name) != manifest[key]:
                raise ValueError('Existing index identity changed: ' + name)
        print(json.dumps({'event': 'reuse_index', **manifest}), flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metadata').mkdir(exist_ok=True)
    from method.full_diffusion.garage import load_config, BACKBONE
    config = load_config()
    reference = load_reference('original_irwm_prepare', ROOT / 'method/irwm_bev/prepare.py')
    audit = load_reference('original_garage_quality', ROOT / 'tools/audit_full_diffusion.py')
    source = ROOT / 'data/full_diffusion/pdm_lite_seed901'
    routes, counts = [], Counter()
    start = time.monotonic()
    if (source / 'routes.json').exists():
        routes = json.loads((source / 'routes.json').read_text())
        source_kind = 'reuse_existing_full_diffusion_routes'
    else:
        source_kind = 'reconstruct_full_diffusion_quality_and_repetition_filter'
        for scenario in sorted(data_root.iterdir()):
            if not scenario.is_dir() or scenario.name.startswith('.'):
                continue
            for route in sorted(scenario.iterdir()):
                if not route.is_dir() or route.name.startswith('FAILED_'):
                    continue
                counts['candidate_routes'] += 1
                repetition = re.search(r'_Rep(\d+)', route.name)
                if repetition is None or int(repetition.group(1)) >= config.num_repetitions:
                    counts['excluded_repetition'] += 1
                    continue
                results = route / 'results.json.gz'
                if not results.exists() or not audit.quality_passes(read_json(results)):
                    counts['excluded_quality'] += 1
                    continue
                routes.append(route.relative_to(data_root).as_posix())
            print(json.dumps({'event': 'route_inventory', 'scenario': scenario.name, 'accepted_routes': len(routes)}), flush=True)
    if not routes or any(not (data_root / r / 'measurements').is_dir() for r in routes):
        raise ValueError('Source routes are empty or unavailable under the selected dataset')
    write_json(out / 'source_routes.json', routes)
    official = ROOT / 'third_party/carla_garage/leaderboard/data/bench2drive220.xml'
    kept, train_names, val_names, split = reconstruct_split(routes, reference, official, args.seed)
    for filename, value in [('routes.json', kept), ('split.json', split)]:
        if (out / filename).exists() and json.loads((out / filename).read_text()) != value:
            raise ValueError('Partial index has different route ordering or split')
        write_json(out / filename, value)
    print(json.dumps({'event': 'split', 'source_routes': len(routes), 'kept_routes': len(kept),
                     'train_routes': len(train_names), 'val_routes': len(val_names),
                     'excluded_test_families': len(split['excluded_official_test_families']),
                     'unmapped_templates': len(split['unmapped_source_templates'])}), flush=True)
    tasks = [(i, r, str(data_root), str(out), config.skip_first, args.stride) for i, r in enumerate(kept)]
    samples = {'train': [], 'val': []}
    old_counts = Counter()
    required, frame_count = [], 0
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        for count, (rid, anchors, n, old, frames) in enumerate(pool.map(metadata_task, tasks, chunksize=1), 1):
            part = 'val' if kept[rid] in val_names else 'train'
            if part == 'val' and not len(anchors):
                raise ValueError('A validation route has no full-horizon samples: ' + kept[rid])
            samples[part].extend((rid, int(a)) for a in anchors)
            old_counts[part] += old
            frame_count += n
            required.extend((rid, int(f)) for f in frames)
            if count % 100 == 0:
                print(json.dumps({'event': 'metadata', 'indexed_routes': count, 'total_routes': len(kept),
                                 'raw_frames': frame_count, 'seconds': time.monotonic() - start}), flush=True)
    filtering = {}
    for part in ('train', 'val'):
        array = np.asarray(samples[part], dtype=np.int32).reshape(-1, 2)
        np.save(out / f'{part}.npy', array)
        filtering[part] = {'old_samples': old_counts[part], 'samples': len(array),
                           'removed_incomplete_3s': old_counts[part] - len(array),
                           'active_routes': len(np.unique(array[:, 0]))}
    if filtering['val']['active_routes'] != 44:
        raise ValueError('Validation must retain all 44 routes')
    np.save(out / 'required_frames.npy', np.asarray(required, dtype=np.int32).reshape(-1, 2))
    from .legacy import module
    contract = module('interface').CONTRACT
    manifest = {'version': 2, 'method': 'irwm_bev_t', 'data_root': str(data_root), 'seed': args.seed,
                'routes': len(kept), 'train_routes': len(train_names), 'validation_routes': 44,
                'train_samples': len(samples['train']), 'validation_samples': len(samples['val']),
                'active_train_routes': filtering['train']['active_routes'], 'raw_frames': frame_count,
                'sample_stride': args.stride, 'skip_first': config.skip_first, 'num_repetitions': config.num_repetitions,
                'frame_dt_s': .25, 'history_frames': 3, 'future_frames': 12,
                'query_times_seconds': [.5, 1., 1.5, 2., 2.5, 3.], 'action_contract': contract,
                'filtering': filtering, 'teacher_sha256': sha(BACKBONE / 'model_0030_1.pth'),
                'teacher_config_sha256': sha(BACKBONE / 'config.json'),
                'source_routes_sha256': sha(out / 'source_routes.json'), 'official_xml_sha256': sha(official),
                'split_sha256': sha(out / 'split.json'), 'train_index_sha256': sha(out / 'train.npy'),
                'val_index_sha256': sha(out / 'val.npy'), 'required_frames': len(required),
                'required_frames_sha256': sha(out / 'required_frames.npy'),
                'selection': 'route macro average masked multi-step normalized feature MSE',
                'feature_contract': 'tfpp-64x64x64-bf16-v1; x forward = width; y right = height; [-32,32] m',
                'temporal_contract': 'six exact future observations at 0.5 to 3.0 s; full horizon only; no feature interpolation',
                'source_kind': source_kind, 'source_filter_counts': dict(counts),
                'source_scripts_sha256': {name: sha(ROOT / name) for name in [
                    'method/full_diffusion/data.py', 'tools/audit_full_diffusion.py',
                    'method/irwm_bev/prepare.py', 'method/irwm_bev_t/prepare.py']},
                'split_reconstructed': source_kind.startswith('reconstruct'),
                'original_cluster_index_unavailable': True,
                'cache_coverage': 'union of three history and six exact future observations for every retained anchor'}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'event': 'index_ready', **manifest, 'seconds': time.monotonic() - start}), flush=True)


if __name__ == '__main__':
    main()
