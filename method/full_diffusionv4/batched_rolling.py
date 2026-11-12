"""The same seconds/pose rolling contract, with one batched CUDA transform."""
import torch

from .geometry import FUTURE_TIMES, cumulative_valid
from .rolling import Plan, RollingPlanStore


class BatchedRollingPlanStore(RollingPlanStore):
    def prior(self, scenes, poses, timestamps):
        stamps = timestamps.detach().cpu().tolist() if torch.is_tensor(timestamps) else list(timestamps)
        b, device = len(scenes), poses.device
        delta = torch.zeros(b, 6, 2, device=device)
        valid = torch.zeros(b, 6, dtype=torch.bool, device=device)
        ages = torch.zeros(b, device=device)
        selected, old, elapsed = [], [], []
        for i, scene in enumerate(scenes):
            previous = self.entries.get(str(scene))
            if previous is None:
                continue
            age = float(stamps[i]) - previous.timestamp
            if age <= 0 or age > self.max_gap + 1e-8:
                self.reset(scene)
            else:
                selected.append(i); old.append(previous); elapsed.append(age)
        if not selected:
            return delta, valid, ages
        index = torch.tensor(selected, device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            previous_delta = torch.stack([p.delta.to(device) for p in old]).double()
            previous_pose = torch.stack([p.pose.to(device) for p in old]).double()
            age = torch.tensor(elapsed, device=device, dtype=torch.float64)
            query = torch.tensor(FUTURE_TIMES, device=device, dtype=torch.float64)[None] + age[:, None]
            available = query <= 3. + 1e-8
            positions = torch.cat((previous_delta.new_zeros(len(old), 1, 2), previous_delta.cumsum(1)), 1)
            location = (query / .5).clamp(0, 6)
            left = location.floor().long().clamp(max=5)
            right, weight = left + 1, location - left
            points = (positions.gather(1, left[..., None].expand(-1, -1, 2)) * (1 - weight[..., None])
                      + positions.gather(1, right[..., None].expand(-1, -1, 2)) * weight[..., None])
            transform = torch.linalg.solve(poses[index].double(), previous_pose)
            points = torch.einsum('bij,bkj->bki', transform,
                                  torch.cat((points, torch.ones_like(points[..., :1])), -1))[..., :2]
            increments = torch.cat((points[:, :1], points[:, 1:] - points[:, :-1]), 1)
            increments = torch.where(available[..., None], increments, 0.)
            delta[index] = increments.float()
            valid[index] = cumulative_valid(available)
            ages[index] = age.float()
        return delta, valid, ages

    def update(self, scenes, poses, timestamps, delta):
        if delta.shape != (len(scenes), 6, 2) or not bool(torch.isfinite(delta).all()):
            raise ValueError('Only finite six-interval predictions can enter rolling state')
        stamps = timestamps.detach().cpu().tolist() if torch.is_tensor(timestamps) else list(timestamps)
        saved_delta, saved_pose = delta.detach().float().clone(), poses.detach().double().clone()
        for i, scene in enumerate(scenes):
            self.entries[str(scene)] = Plan(saved_delta[i], saved_pose[i], float(stamps[i]))
