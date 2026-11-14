# 来源与第三方声明

本仓库是一次研究训练的代码与记录快照。原有第三方文件保留其版权头与许可证；此文件不为其他作者的代码重新授权。

| 来源 | 固定身份 / 使用方式 | 许可证记录 |
|---|---|---|
| [neasonrum/WAMPC](https://github.com/neasonrum/WAMPC) | `d98a90b16619d585d6db400d3cfd924c35b225b6` 基础上的实际工作区源码 | 导出时根目录未发现 LICENSE；[原 README](docs/README.upstream.md) 保留 |
| [neasonrum/WMMPC](https://github.com/neasonrum/WMMPC) | `0c8a65142739515d2c9796d7358dd35b67d150da`；参考 V5 的冻结 V3 与零门控未来读取，本版修改为 4Hz 连续训练和每锚点新鲜世界 | 保留来源说明；固定文件身份见 [V5 来源记录](provenance/v5_training_source.json) |
| [CVCI2_Res_V11_PlanA](https://github.com/123mike-source/CVCI2_Res_V11_PlanA) | `be6b7842adb8f28d2f05068d8f28206e2b3581ac`；参考世界模型部分解冻、递归滚动和采样训练策略，按 BCDE 结构适配 | 未整体复制其仓库；引用文件见正式配置 `reference` 字段 |
| [BridgeDrive](https://github.com/ShuLiu-ETHZ/BridgeDrive) | `85aa089a321a7b7cbc42aa58b3be508eb223d9e2`；原样保留 `blocks.py` 与 `conditional_unet1d.py`，通过 `WMMPC_BRIDGE_MODULES` 加载 | [MIT](vendor/bridgedrive/LICENSE)，文件保存在 `vendor/bridgedrive/modules/` |
| [CARLA Garage](https://github.com/autonomousvision/carla_garage) | git submodule，`48e6a6dbcb133220333bd65e242fbc2b492bb15b`；TF++ 感知与数据预处理 | [原许可证副本](docs/licenses/carla_garage.LICENSE)，子模块保留原许可 |
| [CARLA](https://github.com/carla-simulator/carla) 0.9.15 navigation agents | WAMPC 已有的导航源文件副本；逐文件哈希见 [source.json](third_party/carla_navigation_0_9_15/source.json) | [MIT](third_party/carla_navigation_0_9_15/LICENSE) |

训练数据使用 Bench2Drive PDM-lite。`artifacts/dataset_index/` 保存本次划分、anchor 索引与紧凑数值元数据，原始图像、点云和大型特征缓存未上传。数据的获取与使用遵循数据集原始条款。

完整实际文件来源、字节数与 SHA256 见 [`provenance/source_snapshot.json`](provenance/source_snapshot.json)。仓库保存的服务器绝对路径用于追溯运行，不代表这些路径在其他机器上可直接使用。
