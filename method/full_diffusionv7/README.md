# WAMPC V7

V7 在 PDM-Lite 上以 4Hz 锚点训练，输出 10 个空间路径点和 6 段 0.5 秒时间位移。TF++ 感知和 [BCDE 世界模型](../irwm_bev_t_large/)均冻结。CARLA 闭环以 20Hz 更新，使用 `k−10、k−5、k` 的观测、`k−5` 的规划先验和 25 步 DDIM。

- 训练入口为 [`train.py`](train.py)。[Baseline](config_train4hz_baseline.json) 和 [uniform](config_train4hz_uniform.json) 的 `max_noise_fraction` 分别为 `0.5` 和 `0.0`。各配置中的路径指向同济 HPC 的索引、TF++、V3 和 BCDE 权重。训练 probe 也使用配置指定的 V6 参考权重。
- CARLA 闭环的模型入口为 [`inference/policy.py`](inference/policy.py)，评测 agent 为 [`full_diffusionv7_fast_agent.py`](../../evaluation/full_diffusionv7_fast_agent.py)。评测时将 `--agent-config` 指向已有的 V7 `.pt` 文件。

训练源码来自同济 HPC 的 WAMPC 工作树 `d98a90b16619d585d6db400d3cfd924c35b225b6`，闭环源码来自 `v7_speed_uniform_pretest35_20261119_r01` 评测快照。
