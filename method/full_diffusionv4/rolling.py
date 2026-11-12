"""Detached plans advanced by elapsed seconds, then rebased using real odometry."""
from dataclasses import dataclass
import torch

from .geometry import FUTURE_TIMES, cumulative_valid


@dataclass
class Plan:
    delta: torch.Tensor
    pose: torch.Tensor
    timestamp: float


def advance_plan(delta, source_pose, current_pose, elapsed):
    """Interpolate only inside the old 3s plan; never invent a missing tail."""
    if delta.shape != (6, 2) or source_pose.shape != (3, 3) or current_pose.shape != (3, 3):
        raise ValueError('Expected a six-interval plan and two SE(2) transforms')
    if not 0 <= elapsed < 3 or not bool(torch.isfinite(delta).all()):
        return torch.zeros_like(delta), torch.zeros(6, device=delta.device, dtype=torch.bool)
    with torch.autocast(device_type=delta.device.type, enabled=False):
        times = torch.tensor(FUTURE_TIMES, device=delta.device, dtype=torch.float64)
        query = times + elapsed
        valid = query <= 3.0 + 1e-8
        old = torch.cat((delta.new_zeros(1, 2), delta.float().cumsum(0)), 0).double()
        location = (query / .5).clamp(0, 6)
        left = location.floor().long().clamp(max=5)
        right = left + 1
        weight = location - left
        xy = old[left] * (1 - weight[:, None]) + old[right] * weight[:, None]
        transform = torch.linalg.solve(current_pose.double(), source_pose.double())
        xy = (torch.cat((xy, torch.ones_like(xy[:, :1])), -1) @ transform.T)[:, :2]
        increments = torch.cat((xy[:1], xy[1:] - xy[:-1]), 0)
        increments = torch.where(valid[:, None], increments, 0.)
    return increments.float(), cumulative_valid(valid)


class RollingPlanStore:
    """One state per scene/route, safe for batch reordering and route boundaries."""
    def __init__(self, max_gap=0.5):
        self.max_gap = float(max_gap)
        self.entries = {}

    def reset(self, scene=None):
        if scene is None:
            self.entries.clear()
        else:
            self.entries.pop(str(scene), None)

    def prior(self, scenes, poses, timestamps):
        b, device = len(scenes), poses.device
        deltas = torch.zeros(b, 6, 2, device=device)
        valid = torch.zeros(b, 6, dtype=torch.bool, device=device)
        ages = torch.zeros(b, device=device)
        for i, scene in enumerate(scenes):
            old = self.entries.get(str(scene))
            if old is None:
                continue
            elapsed = float(timestamps[i]) - old.timestamp
            if elapsed <= 0 or elapsed > self.max_gap + 1e-8:
                self.reset(scene)
                continue
            deltas[i], valid[i] = advance_plan(old.delta.to(device), old.pose.to(device), poses[i], elapsed)
            ages[i] = elapsed
        return deltas, valid, ages

    def update(self, scenes, poses, timestamps, delta):
        if delta.shape != (len(scenes), 6, 2) or not bool(torch.isfinite(delta).all()):
            raise ValueError('Only finite decoded metric temporal predictions can enter rolling state')
        for i, scene in enumerate(scenes):
            self.entries[str(scene)] = Plan(delta[i].detach().clone(), poses[i].detach().double().clone(),
                                             float(timestamps[i]))

    def state_dict(self):
        return {key: {'delta': value.delta.cpu(), 'pose': value.pose.cpu(), 'timestamp': value.timestamp}
                for key, value in self.entries.items()}

    def load_state_dict(self, state):
        self.entries = {key: Plan(**value) for key, value in state.items()}
