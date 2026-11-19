"""Checkpoint selection on fixed, balanced training anchors using full DDIM."""
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from method.full_diffusionv4.bootstrap import ROOT
from method.full_diffusionv4.joint_train import move, rank_world
from method.full_diffusionv4.world import sha256 as hash_file
from method.full_diffusionv6.data import collate_with_predecessors
from method.full_diffusionv6.rolling import QuarterSecondPlanStore

from .speed import GROUPS, speed_group


def atomic_json(path, payload):
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    partial.replace(path)


class TrainingProbe:
    def __init__(self, dataset, config, output, device, training_prior):
        self.config, self.device, self.output = config, device, output
        self.training_prior = training_prior
        self.settings = config['training_probe']
        self.stop_mps = config['labels']['stop_speed_mps']
        self.transition_mps = config['loss']['speed_supervision']['transition_mps']
        rank, size = rank_world()
        payload = [self.select_rows(dataset) if rank == 0 else None]
        if size > 1:
            dist.broadcast_object_list(payload, src=0, device=device)
        rows = payload[0]
        if len({row[1] for row in rows}) != len(rows):
            raise ValueError('The fixed probe requires a distinct route per anchor')
        self.panel_sha256 = sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
        if rank == 0:
            info = dict(source='filtered_training_anchors', held_out=False, groups=GROUPS,
                        index_identity=dataset.manifest['identity'], sha256=self.panel_sha256,
                        rows=rows, settings=self.settings)
            if (output / 'training_probe.json').exists():
                if json.loads((output / 'training_probe.json').read_text())['sha256'] != self.panel_sha256:
                    raise ValueError('Training probe differs from original run')
            else:
                atomic_json(output / 'training_probe.json', info)
            print(f'V7_PROBE_PANEL {len(rows)} distinct training routes sha256={self.panel_sha256}', flush=True)
        local = rows[rank::size]
        self.batches = []
        n = self.settings['batch_per_rank']
        for start in range(0, len(local), n):
            shard = local[start:start + n]
            examples = []
            for index, route, slot, frame, _ in shard:
                item = dataset[index]
                if item['prior_observation'] is None:
                    item['prior_observation'] = dataset.observation(route, slot - 1, frame - 1)
                examples.append(item)
            batch, support = collate_with_predecessors(examples)
            self.batches.append((batch, support, torch.tensor([r[-1] for r in shard])))

    def select_rows(self, dataset):
        rng = np.random.default_rng(self.settings['seed'])
        goal = self.settings['samples_per_group']
        dt = self.config['labels']['temporal_interval_seconds']
        groups = [[] for _ in GROUPS]
        selected_routes = set()
        for index in rng.permutation(len(dataset)):
            route, slot, frame = map(int, dataset.samples[index])
            if route in selected_routes:
                continue
            m = dataset.metadata(route)
            if slot == 0 or int(m['frame'][slot - 1]) != frame - 1 or not m['temporal_mask'][slot, 1]:
                continue
            if not m['spatial_mask'][slot].any():
                continue
            current = float(m['speed'][slot].item())
            target = float(np.linalg.norm(m['temporal_delta'][slot, 1]) / dt)
            group = int(speed_group(torch.tensor(target), torch.tensor(current), self.stop_mps, self.transition_mps))
            if len(groups[group]) >= goal:
                continue
            if not (dataset.teacher_root / dataset.routes[route] / f'{frame - 3:04}.npy').is_file():
                continue
            groups[group].append([int(index), route, slot, frame, group])
            selected_routes.add(route)
            if all(len(g) == goal for g in groups):
                # Interleave categories, then shuffle once before partitioning across ranks.
                rows = [row for group in groups for row in group]
                return [rows[i] for i in rng.permutation(len(rows))]
        raise ValueError(f'Insufficient training probe anchors by group: {[len(g) for g in groups]}')

    @torch.inference_mode()
    def measure(self, policy):
        rank, size = rank_world()
        d = self.device
        generator = torch.Generator(device=d).manual_seed(self.settings['seed'] + rank + 1)
        was_training = policy.training
        policy.eval()
        sums = torch.zeros((len(GROUPS), 5), device=d, dtype=torch.float64)
        try:
            for batch, support, group in self.batches:
                batch = move(batch, d)
                group = group.to(d)
                prior, valid, age, filled = self.training_prior(QuarterSecondPlanStore(), policy, batch,
                                                                 support, generator)
                if filled != len(group) or not bool(valid.any(-1).all()):
                    raise ValueError('Probe requires the exact predecessor at every training anchor')
                with torch.autocast(device_type=d.type, dtype=torch.bfloat16, enabled=d.type == 'cuda'):
                    condition = policy.condition(batch, policy.future(batch, prior, valid, age))
                    path, delta = policy.sample(condition, len(group), d, generator,
                                                self.config['diffusion']['inference_steps'])
                codec = policy.planner.codec
                predicted = codec.target_speed(delta)
                target = codec.target_speed(batch['temporal_delta'])
                ade = (path.float() - batch['path'].float()).norm(dim=-1)
                mask = batch['spatial_mask'].bool()
                errors = torch.stack(((predicted - target).abs(), predicted - target,
                                      torch.where(mask, ade, 0.).sum(-1) / mask.sum(-1).clamp_min(1),
                                      ((predicted >= self.stop_mps) & (target < self.stop_mps)).float(),
                                      torch.ones_like(target)), dim=-1).double()
                sums.index_add_(0, group, errors)
        finally:
            policy.train(was_training)
        if size > 1:
            dist.all_reduce(sums)
        if not bool(torch.isfinite(sums).all()) or not bool((sums[:, -1] == self.settings['samples_per_group']).all()):
            raise ValueError('Missing or nonfinite training probe metrics')
        groups = {}
        for name, (error, bias, ade, false_move, count) in zip(GROUPS, sums.tolist()):
            groups[name] = dict(count=int(count), speed_mae_mps=error / count, speed_bias_mps=bias / count,
                                spatial_ade_m=ade / count, stopped_expert_moving_predictions=int(false_move))
        return dict(speed_macro_mae_mps=float((sums[:, 0] / sums[:, -1]).mean()),
                    spatial_ade_m=float(sums[:, 2].sum() / sums[:, -1].sum()), groups=groups)

    def reference(self, policy, load_state, planner_training_state):
        path = ROOT / self.settings['reference_checkpoint']
        if hash_file(path) != self.settings['reference_sha256']:
            raise ValueError('Probe reference checkpoint checksum differs')
        stored = self.output / 'probe_v6_final.json'
        rank, _ = rank_world()
        if stored.is_file():
            result = json.loads(stored.read_text())
            if (result['panel_sha256'] != self.panel_sha256
                    or result['checkpoint_sha256'] != self.settings['reference_sha256']):
                raise ValueError('Probe V6 reference identity differs')
            return result
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state['format'] != 'full_diffusionv6_train4hz_scratch_v1' or state['progress']['epoch'] != 10
                or state['config']['world'] != self.config['world'] or state['config']['v3'] != self.config['v3']):
            raise ValueError('Wrong V6 reference checkpoint')
        original = {part: {name: t.detach().cpu().clone() for name, t in tensors.items()}
                    for part, tensors in planner_training_state(policy.planner).items()}
        try:
            load_state(policy.planner, state)
            result = self.measure(policy)
        finally:
            load_state(policy.planner, original)
        result.update(panel_sha256=self.panel_sha256, checkpoint_sha256=self.settings['reference_sha256'])
        if rank == 0:
            atomic_json(stored, result)
        return result

    def consider(self, policy, config, metadata, progress, reference, planner_state, checkpoint_format, contract):
        result = self.measure(policy)
        rank, _ = rank_world()
        if rank != 0:
            return
        ade_limit = reference['spatial_ade_m'] * self.settings['max_spatial_ade_ratio']
        ade, speed = result['spatial_ade_m'], result['speed_macro_mae_mps']
        passed = ade <= ade_limit
        # Prefer candidates within the V6 spatial guard, then rank them by speed error.
        key = [not passed, speed if passed else speed + .5 * ade, ade, progress['epoch']]
        result.update(epoch=progress['epoch'], updates=progress['updates'], panel_sha256=self.panel_sha256,
                      spatial_guard_m=ade_limit, passed_spatial_guard=passed, selection_key=key)
        atomic_json(self.output / f"probe_epoch_{progress['epoch']:02}.json", result)
        print('V7_TRAINING_PROBE ' + json.dumps(result, allow_nan=False), flush=True)
        best_path = self.output / 'best.json'
        if best_path.is_file() and json.loads(best_path.read_text())['selection_key'] < key:
            return
        best_tmp = self.output / 'best.pt.partial'
        payload = dict(format=checkpoint_format, training_contract=contract, method=config['method'],
                       experiment=config['experiment'], config=config, world_tuned_state=None,
                       world=policy.world.provenance, v3=config['v3'], dataset=metadata,
                       progress={**progress, 'checkpoint_selection': 'fixed_training_probe', 'training_probe': result},
                       gates=policy.planner.gates(), **planner_state(policy.planner))
        torch.save(payload, best_tmp)
        result['checkpoint_sha256'] = hash_file(best_tmp)
        best_tmp.replace(self.output / 'best.pt')
        atomic_json(best_path, result)


def verify_best(output, epochs):
    output = Path(output)
    results = [json.loads((output / f'probe_epoch_{i:02}.json').read_text()) for i in range(1, epochs + 1)]
    selected = json.loads((output / 'best.json').read_text())
    best = min(results, key=lambda r: r['selection_key'])
    if (any(not math.isfinite(r['speed_macro_mae_mps']) for r in results)
            or any(r['panel_sha256'] != selected['panel_sha256'] for r in results)
            or any(selected[k] != v for k, v in best.items())
            or hash_file(output / 'best.pt') != selected['checkpoint_sha256']):
        raise ValueError('Best weight does not match the completed probe history')
    return selected
