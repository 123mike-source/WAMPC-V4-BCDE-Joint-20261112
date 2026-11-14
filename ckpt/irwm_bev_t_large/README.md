# IRWM-BEV-T-large (BCDE) training checkpoints

These checkpoints are stored with Git LFS. Run `git lfs pull` after cloning before using them.

| File | Training state | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `best.pt` | Best validation NMSE, epoch 21 | 178237189 | `b83eba4a33bb90031c43fea668087346a7e5a6af8aada53a8c5094f6f0703c2a` |
| `latest.pt` | Final completed epoch 24, 166944 updates; resumable state | 178262631 | `7d8b90746e976d1b8ffc2821d3082ab306aa35c71b2fc5873471efb960e8c49b` |

The files came from `/share/home/u21012/nixiran/WAMPC/ckps/irwm_bev_t_large/` on Tongji Gao HPC. `best.pt` is the published `release/best.pt`; its original validation report is in `best.manifest.json`. `latest.pt` came from `tongji_bcde_seed901_20261111/latest.pt`, and `completed.json` records the finished training state. SHA-256 values above were calculated on the source files and checked after transfer.

The V4 instructions expect the best world-model checkpoint at `ckps/irwm_bev_t_large/release/best.pt`. From the repository root, place the LFS-fetched file there with:

```bash
mkdir -p ckps/irwm_bev_t_large/release
cp ckpt/irwm_bev_t_large/best.pt ckps/irwm_bev_t_large/release/best.pt
```

`ckps/` remains ignored as a local runtime directory. The V4 planner checkpoint still needs its own `world_tuned_state` alongside this base BCDE checkpoint.
