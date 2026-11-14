"""A route's previous prediction must be exactly one recorded frame old."""
import math

import torch

from method.full_diffusionv4.batched_rolling import BatchedRollingPlanStore


class QuarterSecondPlanStore(BatchedRollingPlanStore):
    def __init__(self):
        super().__init__(max_gap=.25)

    def prior(self, scenes, poses, timestamps):
        if len(set(map(str, scenes))) != len(scenes):
            raise ValueError('A batch must contain distinct routes to preserve causality')
        stamps = timestamps.detach().cpu().tolist() if torch.is_tensor(timestamps) else list(timestamps)
        for scene, now in zip(scenes, stamps):
            previous = self.entries.get(str(scene))
            if previous is not None and not math.isclose(float(now) - previous.timestamp, .25, abs_tol=1e-6, rel_tol=0):
                # A gap, rewind, new route cycle or mismatched cadence is a cold start.
                self.reset(scene)
        return super().prior(scenes, poses, stamps)
