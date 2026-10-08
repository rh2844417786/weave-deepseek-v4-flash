# Weave / DeepSeek-V4-Flash 实验项目

本项目使用原版 DeepSeek-V4-Flash 混合精度权重。交付包含 M16/N64 分组 tile、真实 CUDA-event 校准与 `scripts/run_full_experiment.py` 正式两组入口。最新服务器记录d4275c0：同步单点诊断四rank通过、容器退出0，宿主驱动因root-owned JSON的0600权限屏蔽继承ACL而退出1，完整校准尚未恢复。当前修正原子JSON发布权限为0664，新`-2`作业需验证宿主读取和导出链路。此前9228519报告core通过、校准四rank各完成285点后CUDA故障；本次同步重放未复现该故障，完整扫描仍待恢复。GPQA 官方数据已下载，并用锁定的 DeepSeek tokenizer/编码器完成本地输入长度审计。历史ced332b报告在e59e064上完成修正后全43层checkpoint、原生EP4、Weave eager和Weave graph功能验收，8个生成token一致。原始工作目录证据待归档，正式速度结果仍未产生。

## 入口

- [分组 tile、实测校准与正式 GPQA 统一入口](docs/FORMAL_GPQA_EXPERIMENT.md)
- [校准 CUDA 故障：单点定位与285点恢复](docs/CALIBRATION_FAULT_RECOVERY.md)

- [当前服务器镜像、版本与工作目录契约](docs/RUNTIME_0517.md)
- [服务器仅访问 GitHub：数据与依赖离线准备](docs/RUNTIME_0517.md#服务器网络与离线准备)
- [真实 43 层 checkpoint、WMMA 与 SGLang graph 验收入口](docs/REAL_CHECKPOINT_GRAPH_ACCEPTANCE.md)
- [最新模型验收：四阶段通过、8 token 一致（服务器生成摘要）](reports/model_acceptance-0c508db6a19e.json)
- [最新完整作业：core通过、校准285点后CUDA故障](reports/full_experiment-047ed4696f52.json)
- [单点诊断：四rank通过，驱动被JSON权限阻塞](reports/calibration_diagnostic-047ed4696f52-1_operation_record.json)
- [前一轮 a934293 验收记录（旧 logprob 门导致停止）](reports/model_acceptance.json)
- [CUDA 功能核心、动态调度与 SGLang adapter](docs/CUDA_FUNCTIONAL_CORE.md)
- [历史 SGLang 0.5.16 接入核对](docs/SGLANG_V0516_INTEGRATION.md)
- [EP 路由参考当前编码计划](docs/superpowers/plans/2026-09-30-deepseek-routing-reference.md)
- [DeepSeek-V4-Flash 迁移与当前决策](docs/DEEPSEEK_V4_FLASH_MIGRATION.md)
- [需求与架构设计](docs/superpowers/specs/2026-09-29-weave-qwen35-design.md)
- [详细编码计划](docs/superpowers/plans/2026-09-29-weave-qwen35.md)
- [GPT-6 Sol 交接](docs/HANDOFF_GPT6_SOL.md)
- [GPQA 数据、长度与复核方法](data/gpqa/README.md)
- [DeepSeek GPQA 输入长度审计](data/gpqa/deepseek_length_audit.json)

当前目标：原版 DeepSeek-V4-Flash 混合精度权重，在 GPU 4–7 上以 EP=4 比较原生 SGLang 与完整 Weave。两组采用相同的专家划分和 attention/dense 并行布局；原生运行时对 MXFP4 专家 EP 路径的支持须先验证。SGLang 的 TP 参数可与 EP 参数同时使用，不能把 `--tp-size 4` 等同于 MoE 只使用 TP。GPQA Main、batch=1、Think High、最多生成 32K（32768）tokens，每组服务 ready 后预热与测量共最多20分钟。旧 Qwen 设计、计划和交接文件保留为历史记录，不能直接作为 DeepSeek 实现合同。

已有代码包括：只读预检、DeepSeek GPQA token ID 准备、decode 计数、EP 路由去重/合并 CPU 参考，以及原生 EP4 Docker 启动入口。现已加入 CUDA IPC persistent 功能核心、GPU 端 c/K 选择和连续 stealing、raw MXFP4 专家加载器，以及源码核对后的 SGLang 0.5.17 eager adapter。默认使用当前服务器已有的 `lmsysorg/sglang:v0.5.17`，工作目录为 `<repo>/work`。后端名为 `weave_functional`：历史 H100 CUDA 编译、两轮四卡小张量功能与 core graph replay 已通过并核对原始报告；修正后的 BF16 WMMA、真实 checkpoint 与 batch=1 SGLang graph 在 e59e064 的服务器摘要中报告通过。完整原始 rank/coverage 报告待回传；本轮分组 Tensor Core tile、实测校准和正式性能入口已实现，当前源码的 H100 验收与测量待执行。证据见 `reports/core_acceptance.json`，状态见 `reports/implementation_status.json`。

当前服务验收中，logprob 和 top-k 候选集合仅保存诊断，不作为通过/失败门限；实际生成 token 一致性、checkpoint、43 层执行与 graph replay 按验收入口检查。

`tests/gpu/check_checkpoint.py` 覆盖全部 43 层 scalar/WMMA/tiled 差分、实际窃取设备计数和 core graph；另有分组 K=1/2/4/8 检查。`python3 scripts/run_full_experiment.py` 在 GPUQ 分配内串起新源码 gate、校准、原生 EP4、Weave eager/graph 验收以及正式两组 GPQA。当前 CUDA 源码 digest 已变化，e59e064 的通过摘要不能替代本轮 gate；实测 profile 及 GPQA 速度对照仍待服务器生成。

旧 Qwen 输入长度审计的本地复核（不适用于 DeepSeek）：

```bash
python3 scripts/fetch_gpqa.py --verify-only
.venv-data/bin/python scripts/audit_gpqa_lengths.py
```

联网开发机上复核旧 Qwen 审计的命令（仅为历史审计；服务器使用上面的 DeepSeek 离线准备入口）：

```bash
python3 scripts/fetch_gpqa.py
python3 -m pip install -r requirements-data.txt
python3 scripts/audit_gpqa_lengths.py --tokenizer-dir /mnt/public_data/Qwen/Qwen3.5-35B-A3B
```

DeepSeek Think High 输入审计覆盖 GPQA Main 448 题：中位数 191、最长 2550 tokens，无 8K prefill 样本。用 `python3 scripts/prepare_gpqa.py --model-dir <DeepSeek模型目录>` 生成两组共用的 token ID 清单；清单位于 Git 忽略目录。上述数字是 CPU 编码结果，不是 GPU 生成或速度结果。原始数据与生成文本不加入 Git；同步来源和下载脚本。[私有 GitHub 仓库](https://github.com/rh2844417786/weave-deepseek-v4-flash) 已创建，DeepSeek 实际 SGLang EP 启动配置尚待核验。
