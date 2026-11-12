"""Replay exact fixed-stream route visits, in route-sized chunks without Torch."""
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit():
    config = json.loads((ROOT / 'runs/tongji_bcde_joint_seed901_20261112/config.json').read_text())
    completed = json.loads((ROOT / 'runs/tongji_bcde_joint_seed901_20261112/completed.json').read_text())
    samples = np.load(ROOT / 'artifacts/dataset_index/prepared/train.npy', allow_pickle=False)
    settings = config['training']
    global_batch = settings['batch_per_gpu'] * settings['world_size']
    # DistributedRouteStreams groups routes in ascending ID, slots in ascending order.
    groups = {}
    for index, (route, slot) in enumerate(samples):
        groups.setdefault(int(route), []).append((int(slot), index))
    scenes = [np.asarray([index for _, index in sorted(groups[route])]) for route in sorted(groups)]
    order = list(range(len(scenes)))
    random.Random(config['seed']).shuffle(order)
    visits = np.zeros(len(samples), dtype=np.int64)
    max_first_pass = 0
    for global_slot in range(global_batch):
        pool = order[global_slot::global_batch]
        max_first_pass = max(max_first_pass, sum(len(scenes[route]) for route in pool))
        remaining, route_cursor, cycle = completed['updates'], 0, 0
        while remaining:
            if route_cursor == len(pool):
                cycle += 1
                random.Random(config['seed'] + 100003 * cycle + global_slot).shuffle(pool)
                route_cursor = 0
            sequence = scenes[pool[route_cursor]]
            count = min(remaining, len(sequence))
            visits[sequence[:count]] += 1
            remaining -= count
            route_cursor += 1
    report = {'training_samples': len(samples), 'global_streams': global_batch,
              'consumed_updates': completed['updates'], 'draws': int(visits.sum()),
              'unique_visited': int(np.count_nonzero(visits)),
              'min_visits': int(visits.min()), 'max_visits': int(visits.max()),
              'mean_visits': float(visits.mean()), 'max_stream_first_pass_updates': max_first_pass,
              'visit_histogram': dict(sorted(Counter(map(int, visits)).items())),
              'interpretation': 'all retained train anchors; epochs are update budgets, not exact one-pass epochs'}
    assert report['draws'] == completed['updates'] * global_batch
    return report


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
