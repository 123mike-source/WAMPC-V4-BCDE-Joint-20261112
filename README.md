# WAMPC：V5 4Hz 连续训练与 BCDE 世界模型

**最新更新：2026-11-14。** 当前版本实现 V5 训练：保留 IRWM 原有三帧观测间隔，
每条路线按 4Hz 连续滚动，每个锚点重新生成未来 BEV，并将轨迹邻域特征注入空间和时间双分支。

**当前状态：训练代码与 CPU 检查已完成，正式训练尚未启动。**
本版尚未完成真实 BCDE 权重的 GPU 训练冒烟，闭环刷新频率与调度待确定。
仓库同时保存 2026-11-12 已完成的 V4 联合训练源码、配置、日志和结果。

## 版本入口

| 版本 | 入口 | 状态 |
|---|---|---|
| **V5，当前** | [训练说明](method/full_diffusionv5/TRAINING.md)、[配置](method/full_diffusionv5/config_train4hz.json)、[训练代码](method/full_diffusionv5/train.py) | 4Hz 连续训练实现；17 项 CPU 测试和两进程 CPU DDP 检查通过 |
| V4，历史训练 | [原说明及复现步骤](docs/README.v4-20261112.md)、[实际运行记录](runs/tongji_bcde_joint_seed901_20261112/) | 8×L40，10 epochs，2,180 次更新，best 为 epoch 5 |

本次 V5 参考 [neasonrum/WMMPC 的固定提交](https://github.com/neasonrum/WMMPC/tree/0c8a65142739515d2c9796d7358dd35b67d150da)，
并修改训练时序与数据读取。来源及文件哈希见 [V5 来源记录](provenance/v5_training_source.json)
与 [第三方声明](THIRD_PARTY_NOTICES.md)。

## V5 时间与模型约定

| 项目 | 当前训练设置 |
|---|---|
| 原始观测频率 | 4Hz，相邻观测间隔 0.25s |
| 单次历史输入 | `t−0.5、t−0.25、t` 三张 TF++ 教师 BEV |
| 路线采样 | 每条路线一条连续 4Hz 链，每项一个当前锚点 |
| 滚动先验 | 前 0.25s 的模型时间分支预测，detach 后时间插值并按实际观测位姿重定位 |
| 轨迹覆盖 | 旧计划覆盖不到当前 `t+3s`，第六段先验无效，不外推尾部 |
| 世界模型刷新 | 每个训练锚点重新生成；缓存年龄恒为 0 |
| 未来邻域条件 | 当前未来 `0.5、1、1.5、2、2.5s` 五张 BEV，共 95 个 token |
| 双分支 | 参考 V5 的冻结 V3：10 个几何空间点、6 个 0.5s 时间位移 |
| 参数训练 | 冻结 V3 与预训练 BCDE，训练未来采样器和零门控适配层 |
| 规划损失 | `L_spatial + 2 × L_temporal`，六段时间权重 `[3,3,2,2,1,1]` |
| 滚动与验证去噪 | 均为 25 步 DDIM |

例如 `t=1.25s`：使用 `[0.75,1.00,1.25]s` 观测与 `P_1.00`，生成/采样
`[1.75,2.25,2.75,3.25,3.75]s` 的未来 BEV。下一项 `t=1.50s` 使用 `P_1.25` 重新滚动。
路线切换、帧缺口或时间回退时冷启动，由冻结 V3 生成第一条先验。
真实未来轨迹、速度与位姿仅用于监督，不进入世界模型与未来条件。

训练使用单锚点损失；原参考实现的奇偶两条 2Hz 链和缓存年龄 `0/0.25/0.5s` 三时刻损失已移除。
验证采用同样的 4Hz 滚动方式，V5 与 V3 共享初始噪声，分别记录全部帧和世界条件可用帧的路线宏平均 ADE。

## 数据与缓存

沿用旧 seed901 索引的有效训练/验证路线划分，重新枚举所有满足三帧连续历史、完整 3s 未来的 4Hz 锚点。

| 统计 | 训练 | 验证 |
|---|---:|---:|
| 有效路线 | 4,796 | 44 |
| 4Hz 锚点 | 442,768 | 3,784 |
| 需补提取教师 BEV | 2,350 | 16 |
| 可复用 memory/ego | 222,607 | 1,906 |
| 需提取 memory/ego | 220,161 | 1,878 |

以上为原服务器元数据审计结果，覆盖过滤后的源划分。2,366 个缺失教师帧的原始文件均存在；
未对全量传感器文件解码。详见 [数据审计](docs/v5_train4hz_20261114/data_support.json)。

缺失缓存不删除训练样本。`prepare features` 复用同教师的 BEV，只补缺帧；
`prepare contexts` 按原始帧号复用 V4 memory/ego，提取未覆盖帧。
缓存完成标记绑定新索引 identity，旧三锚点索引不能直接用于新训练。

## 运行 V5

环境、子模块、权重和固定源索引的安装与恢复步骤见 [环境及数据准备](docs/README.v4-20261112.md#环境与复现)。
四项基础权重/教师配置继续使用 [权重依赖记录](provenance/weight_dependencies.json) 中的 SHA256。
仓库中的训练配置保留原服务器路径；迁移时将配置复制到 `configs/local_v5_train4hz.json`，
设置本机 `data_root`、索引目录、`training.world_size` 和批量。
如没有旧 V4 上下文缓存，将 `context_source_index` 设为 `null`，从原始观测提取所需上下文。

在已有 Slurm/LSF 计算分配内、完成源索引恢复和权重准备后：

```bash
source scripts/activate.sh
v5_config=configs/local_v5_train4hz.json
python -m method.full_diffusionv5.prepare labels --config "$v5_config" --workers 8
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv5.prepare features \
  --config "$v5_config" --workers 4 --batch-size 16
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv5.prepare contexts \
  --config "$v5_config" --workers 4 --batch-size 16
torchrun --standalone --nproc_per_node=8 -m method.full_diffusionv5.train \
  --config "$v5_config" --output ckps/full_diffusionv5/local_train4hz_fresh
```

进程数必须匹配配置 `training.world_size`。原服务器的一体化入口是
[`tools/train_v5_train4hz_tongji.sh`](tools/train_v5_train4hz_tongji.sh)，依赖其已有 `.runtime` 环境脚本。
默认配方为 8 卡、每卡 32 项、5 个样本预算 epoch；每 epoch 1,730 次更新。
流跨 epoch 连续，epoch 不保证每条路线恰好一遍。`--resume` 仅接受同配置、同索引和同卡数的断点。

新权重格式为 `full_diffusionv5_train4hz_v1`，索引格式为 `full_diffusionv5_index_train4hz_v1`。
闭环适配需等在线时序确定后实现；原参考的 2Hz 缓存加载器不接受新权重格式。

## 已完成检查

```bash
python scripts/verify_snapshot.py
source scripts/activate.sh
python -m unittest discover -s tests -p 'test_full_diffusionv5_train4hz.py' -v
torchrun --standalone --nproc_per_node=2 tools/smoke_v5_train4hz_ddp.py
```

- 17 项 CPU 测试通过：历史选择、时间插值、真实位姿对齐、尾部掩码、路线隔离、
  断点状态、无未来标签输入、零门控等价、梯度与验证时序。
- 两个 CPU/Gloo rank 各执行两次 DDP 更新，冷/热条件混合、参数同步和冻结检查通过。
- 检查使用实际 V3/V5 规划头代码及廉价世界替身；真实 BCDE 的 GPU/NCCL、吞吐与显存尚待验证。

完整 [修改报告](docs/v5_train4hz_20261114/REPORT.md)、[检查记录](docs/v5_train4hz_20261114/verification.json)、
[CPU 日志](docs/v5_train4hz_20261114/unit.log)、[DDP 日志](docs/v5_train4hz_20261114/ddp.log) 已归档。
这些是实现检查记录；本版尚无正式训练曲线或闭环成绩。

## V4 历史结果

2026-11-12 的 V4 使用六个空间点和六个时间位移，部分解冻 BCDE 联合训练。
epoch 5 的离线 future ADE 为 1.39727m，epoch 10 为 1.45502m。
详情见 [V4 原说明](docs/README.v4-20261112.md)、[验证表](results/validation.csv) 与原始运行日志。
历史训练来源见 [V4 来源记录](provenance/source_snapshot.json)；旧发布清单对应提交
`2463503ecf8238780b34722b7e3b8c7bda5c5291`，另存于
[`publication_manifest.v4_20261112.json`](provenance/publication_manifest.v4_20261112.json)。
