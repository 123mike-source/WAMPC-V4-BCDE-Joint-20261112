"""Two CPU ranks exercise warm/cold DDP paths; no dataset or GPU is used."""
import os
import runpy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    helper = runpy.run_path('tests/test_full_diffusionv5_train4hz.py')
    policy = helper['make_policy']()
    model = DDP(policy, broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    data = helper['batch']()
    prior = torch.tensor([1., 0.]).expand(1, 6, 2).clone()
    prior[:, -1] = 0
    generator = torch.Generator().manual_seed(901 + rank)
    for step in range(2):
        warm = step > 0 or rank > 0
        valid = torch.tensor([[warm] * 5 + [False]])
        ages = torch.tensor([.25 if warm else 0.])
        optimizer.zero_grad(set_to_none=True)
        result = model(data, prior, valid, ages, generator)
        result['loss'].backward()
        optimizer.step()
        checksum = torch.stack([sum(p.detach().double().sum() for p in policy.parameters() if p.requires_grad)])
        copies = [torch.empty_like(checksum) for _ in range(dist.get_world_size())]
        dist.all_gather(copies, checksum)
        assert all(torch.equal(checksum, v) for v in copies), 'DDP parameters diverged'
        assert all(p.grad is None for p in policy.planner.base.parameters())
        assert all(p.grad is None for p in policy.world.parameters())
    print(f'V5_TRAIN4HZ_DDP_OK rank={rank} iterations=2', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
