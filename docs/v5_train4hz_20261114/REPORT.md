# V5 训练修改记录：4Hz 连续链与每帧新鲜世界

日期：2026-11-14（北京时间）。

已将训练修改部署到 `/share/home/u21012/nixiran/WAMPC`。
参考版本为 `neasonrum/WMMPC@0c8a65142739515d2c9796d7358dd35b67d150da`。
此次范围是训练、训练索引、缓存准备和对应检查。没有部署 V5 的闭环 agent。
没有启动正式训练、全量缓存提取或 CARLA，也没有申请或释放 GPU。

## 1. 已实现的时间逻辑

| 项目 | 新训练实现 |
|---|---|
| 路线流 | 每条路线一条链，依次读取相邻 4Hz 锚点，不拆奇偶帧 |
| 每个 item | 一个当前时刻 `t` |
| 观测 | 三张原始教师 BEV：`t−0.5、t−0.25、t` |
| 滚动先验 | 同路线 `t−0.25` 的模型时间分支预测，25 步 DDIM，detach 后存入状态 |
| 时间前移 | 对旧累计位置按时间插值，再用真实已观测位姿做 SE(2) 对齐 |
| 先验末尾 | 当前 `t+3.0s` 超出旧计划覆盖，先验第六段 mask=False，不补造尾部 |
| 世界滚动 | 当前观测与前移先验在每个锚点重新生成未来图 |
| 未来条件 | 当前 `t+[0.5,1,1.5,2,2.5]s` 五张图，95 个轨迹邻域 token |
| 缓存年龄 | 恒为 0，不再训练旧缓存年龄 0.25/0.5 的附加损失 |
| 冷启动 | 缺少上一帧计划、帧缺口、时间回退或路线重启时，使用冻结 V3 生成先验 |

例如当前 `t=1.25s`：观测为 `[0.75,1.00,1.25]s`，先验源为 `P_1.00`，
五张未来 BEV 对应 `[1.75,2.25,2.75,3.25,3.75]s`。
下一项 `t=1.50s` 使用 `P_1.25` 并重新运行世界模型。

因此，训练中不再出现原三锚点 item 在 `t+0.5s` 继续读旧缓存的问题。
未来 GT 仅产生监督，不作为世界动作、邻域位置或运动条件。

## 2. 模型与优化

共享模型函数与参考 V5 的 AST 一致，仅更新模块说明。
保留参考 V5 的空间/时间双分支和冻结 V3 的归一化及输出形状：
10 个几何空间点、6 个时间点。此次没有重新改点数。

- V3 基座冻结，使用 `ckps/full_diffusionv3/release/epoch_10.pt`。
- BCDE 世界模型冻结，使用 `ckps/irwm_bev_t_large/release/best.pt`，
  SHA256 `b83eba4a33bb90031c43fea668087346a7e5a6af8aada53a8c5094f6f0703c2a`。
- 训练未来采样器以及两分支的零门控适配层。
- 每个当前锚点计算一次空间 L1 与时间位移 L1，权重保持 1:2。
- 保留时间段权重 `[3,3,2,2,1,1]`。
- 训练滚动先验与离线验证均使用 25 步 DDIM。
- 流式采样保持各 DDP 进程路线池互斥；断点同时保存已消费采样游标、滚动计划和 RNG。

配置默认继承参考的 8 GPU × 每卡 32 项、5 个样本预算 epoch。
当前训练集大小对应每 epoch 1,730 次更新。样本预算 epoch 不保证所有路线恰好遍历一次。
实际卡数和 batch 应在获得计算资源后选定独立配置。

## 3. 数据及缓存审计

原始数据路径为 `/share/home/u21012/data/bench2drive-PDMlite`。
沿用旧 seed901 索引的有效路线划分，不把结果描述成整个 PDMlite 目录的全量训练。
枚举所有历史连续且有完整 3s 未来的 4Hz 锚点，缓存缺口不删除样本。

| 统计 | 训练 | 验证 |
|---|---:|---:|
| 有效路线 | 4,796 | 44 |
| 新 4Hz 锚点 | 442,768 | 3,784 |
| 所需观测 BEV 帧并集 | 452,360 | 3,872 |
| 缺失教师 BEV | 2,350 | 16 |
| 可复用 memory/ego | 222,607 | 1,906 |
| 需提取 memory/ego | 220,161 | 1,878 |

全部 2,366 个缺失教师帧的 RGB、LiDAR、measurement 原始文件均已逐一检查存在。
审计使用原索引帧元数据、缓存目录与缺失帧文件存在性，未对全量原始传感器文件解码。
memory/ego 源缓存的教师哈希、索引 identity 和路线编号已核对一致。

`prepare features` 在旧教师目录中只补缺帧，并验证所需旧缓存的 NPY 头。
`prepare contexts` 按原始帧号复用旧 memory/ego，补齐未覆盖的当前帧。
新索引位于 `data/full_diffusionv5/pdm_seed901_train4hz_fresh_v1`，尚未生成。

## 4. 验证结果与范围

服务器隔离环境：PyTorch `2.5.0+cu121`。本次使用 CPU 执行检查。

1. Python AST 与配置 JSON 检查通过。
2. 17 项测试全部通过，最终执行时间 8.919s。覆盖：
   三帧历史、奇偶帧同链、末尾锚点、缓存缺口、缓存 identity、帧号复用、
   0.25s 时间插值、真实 SE(2) 位姿对齐、无效尾部、帧缺口、状态恢复、
   跨 rank 路线隔离、无未来标签条件泄漏、零门控与 V3 精确一致、
   有覆盖候选的反向传播、冷启动与同噪声验证。
3. 两个 CPU/Gloo rank 各运行两次 DDP 更新通过，包括一个 rank 冷启动、
   另一个 rank 有世界条件的组合；优化后参数校验一致，冻结参数无梯度。
4. 启动脚本 `bash -n` 通过。

神经网络逻辑测试使用实际 V3/V5 规划头代码以及廉价观测世界替身。
尚未在 GPU 上加载真实 BCDE 权重跑训练冒烟，未验证 NCCL、吞吐或显存占用。
闭环时钟与刷新频率留待后续确定。

## 5. 文件与运行入口

服务器新增：

- `method/full_diffusionv5/train.py`：当前帧损失、模型滚动、同频验证、断点。
- `method/full_diffusionv5/data.py`：单锚点、单路线链、三帧观测。
- `method/full_diffusionv5/rolling.py`：严格 0.25s 先验与缺口冷启动。
- `method/full_diffusionv5/prepare.py`：标签、教师特征与上下文三阶段准备。
- `method/full_diffusionv5/config_train4hz.json`：新训练配置。
- `method/full_diffusionv5/contract.py`：时间与版本约束。
- `method/full_diffusionv5/model.py`、`__init__.py`、`TRAINING.md`。
- `tools/train_v5_train4hz_tongji.sh`：已有分配内的数据准备与训练入口。
- `tests/test_full_diffusionv5_train4hz.py`、`tools/smoke_v5_train4hz_ddp.py`。

在已获得且卡数匹配配置的分配中运行：

```bash
cd /share/home/u21012/nixiran/WAMPC
bash tools/train_v5_train4hz_tongji.sh method/full_diffusionv5/config_train4hz.json all
```

可将 `all` 改为 `labels`、`features`、`contexts`、`train` 分阶段运行。
脚本不申请、取消或释放计算分配。默认输出到
`ckps/full_diffusionv5/tongji_train4hz_fresh_seed901`，已有输出需要显式 `RESUME=1`。

新 checkpoint 格式为 `full_diffusionv5_train4hz_v1`，独立于旧 2Hz 缓存方案。
旧闭环加载器会拒绝此格式，后续确定在线调度后再适配。

## 归档证据

本目录的 `manifest.json` 保存本轮部署文件哈希与参考 commit，
`training_changes.diff` 保存相对参考 V5 的修改，`data_support.json` 保存数据审计。
`verification.json`、`unit.log` 与 `ddp.log` 保存实际 CPU 检查输出。
完整源码在仓库的 `method/full_diffusionv5/`、`tools/` 与 `tests/`。
来源与发布父提交见 `provenance/v5_training_source.json`。
