# Weave / DeepSeek-V4-Flash 实验项目

本项目使用原版 DeepSeek-V4-Flash 混合精度权重，routed专家保留MXFP4/E8M0。历史服务器提交2910864已回传完整374点校准、native/Weave eager/graph验收和两组各1200秒的GPQA预算对照。共同完成4题上，SGLang为107.64 tokens/s，当时Weave为11.31 tokens/s，约0.1051倍；全448题未完成，当轮Weave另有6次cache_flush_error。原始42文件摘要已核对。

最新C32 tiled/eager固定计划诊断已在46a2e1d回传。本轮开始P0/P1计算组织优化：热路径缓存、probe编译/计时对照、真实B32 trace采集，以及相同M16算术的staged132/264/528/direct回放。相关47项CPU检查和源码审查通过；新CUDA源码的编译、全部43层数值和性能证据待服务器执行，当前没有新的吞吐收益结论。

## 入口

- [P0/P1计算组织对照与分阶段运行](docs/P0_P1_COMPUTE_CONTROLS.md)
- [服务器DeepSeek V4 Flash执行Prompt](docs/SERVER_PROMPT_P0_P1_COMPUTE_CONTROLS.md)
- [P0/P1本地验证及服务器待办边界](reports/p0_p1_compute_controls_validation.json)
- [分组 tile、实测校准与正式 GPQA 统一入口](docs/FORMAL_GPQA_EXPERIMENT.md)
- [GPU编号动态读取调度器分配](docs/GPU_ALLOCATION.md)
- [缓存清理失败恢复与服务器复测](docs/CACHE_FLUSH_RECOVERY.md)
- [当前GPQA的设备端阶段诊断与服务器命令](docs/DECODE_EXECUTION_PROFILE.md)
- [已返回的预算GPQA性能结果](reports/server/full-experiment-9d9ee0d02724/comparison.json)
- [校准 CUDA 故障：单点定位与285点恢复](docs/CALIBRATION_FAULT_RECOVERY.md)
- [最新2048故障：362点恢复、异步诊断与本地盘导出](docs/CALIBRATION_2048_RECOVERY.md)

- [当前服务器镜像、版本与工作目录契约](docs/RUNTIME_0517.md)
- [服务器仅访问 GitHub：数据与依赖离线准备](docs/RUNTIME_0517.md#服务器网络与离线准备)
- [真实 43 层 checkpoint、WMMA 与 SGLang graph 验收入口](docs/REAL_CHECKPOINT_GRAPH_ACCEPTANCE.md)
- [最新模型验收：四阶段通过、8 token 一致（服务器生成摘要）](reports/model_acceptance-0c508db6a19e.json)
- [最新完整作业：374点、模型验收与预算GPQA](reports/server/full-experiment-9d9ee0d02724/experiment.json)
- [历史285点后CUDA故障](reports/full_experiment-047ed4696f52.json)
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

当前目标：原版 DeepSeek-V4-Flash 混合精度权重，在调度器分配的四张 H100 上以 EP=4 比较原生 SGLang 与完整 Weave。两组采用相同的专家划分和 attention/dense 并行布局；原生Marlin专家EP4已在服务器验收中执行。SGLang 的 TP 参数可与 EP 参数同时使用，不能把 `--tp-size 4` 等同于 MoE 只使用 TP。GPQA Main、batch=1、Think High、最多生成 32K（32768）tokens，每组服务 ready 后预热与测量共最多20分钟。旧 Qwen 设计、计划和交接文件保留为历史记录，不能直接作为 DeepSeek 实现合同。

已有代码包括：只读预检、DeepSeek GPQA token ID 准备、decode 计数、EP 路由去重/合并 CPU 参考，以及原生 EP4 Docker 启动入口。现已加入 CUDA IPC persistent 功能核心、GPU 端 c/K 选择和连续 stealing、raw MXFP4 专家加载器，以及源码核对后的 SGLang 0.5.17 eager adapter。默认使用当前服务器已有的 `lmsysorg/sglang:v0.5.17`，工作目录为 `<repo>/work`。后端名为 `weave_functional`：历史 H100 CUDA 编译、两轮四卡小张量功能与 core graph replay 已通过并核对原始报告；修正后的 BF16 WMMA、真实 checkpoint 与 batch=1 SGLang graph 在 e59e064 的服务器摘要中报告通过。9906ca7c源码的完整rank/coverage、实测bank及GPQA报告已回传；新增诊断源码需要重新执行源码绑定的H100 gates。证据见 `reports/core_acceptance.json`，状态见 `reports/implementation_status.json`。

当前服务验收中，logprob 和 top-k 候选集合仅保存诊断，不作为通过/失败门限；实际生成 token 一致性、checkpoint、43 层执行与 graph replay 按验收入口检查。

`tests/gpu/check_checkpoint.py` 覆盖全部 43 层 scalar/WMMA/tiled 差分、实际窃取设备计数和 core graph；另有分组 K=1/2/4/8 检查。`python3 scripts/run_full_experiment.py` 在 GPUQ 分配内串起新源码 gate、校准、原生 EP4、Weave eager/graph 验收以及正式两组 GPQA。新增诊断改变CUDA源码digest，旧profile和验收不能代替新gate；使用 --execution-profile 运行时，计时结果明确标为instrumented_diagnostic。

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

DeepSeek Think High 输入审计覆盖 GPQA Main 448 题：中位数 191、最长 2550 tokens，无 8K prefill 样本。用 `python3 scripts/prepare_gpqa.py --model-dir <DeepSeek模型目录>` 生成两组共用的 token ID 清单；清单位于 Git 忽略目录。上述数字是 CPU 编码结果，不是 GPU 生成或速度结果。原始数据与生成文本不加入 Git；同步来源和下载脚本。[GitHub 仓库](https://github.com/rh2844417786/weave-deepseek-v4-flash)已创建，DeepSeek 原生与自研后端的EP4实际启动记录已随服务器报告归档。
