# Weave / DeepSeek-V4-Flash 实验项目

## 当前目标：提高原生 SGLang 吞吐

2026-10-22 主线为 `native_marlin_peer_tail_v1`：保留原生 front 与 prefill，
将 decode down、局部合并和输出归约接入同一 resident worker 执行器，加入输出通道
chunk、有限初始 stealing 和新实测 DAG 模型。它是 MoE tail 的机制适配，不是完整
Weave 五阶段复现。源码、CPU 检查和离线服务器入口已交付；CUDA 编译、四卡机制、
完整模型 Graph 与新吞吐结果仍需服务器验证。

入口：[执行器及验证说明](docs/NATIVE_TAIL_EXECUTOR.md) ·
[统一服务器脚本](scripts/run_native_tail.py) · [服务器完整 Prompt](docs/SERVER_PROMPT_NATIVE_TAIL.md)。原有 [CTA measured-cost 方法](docs/NATIVE_COST_MODEL.md)
及其历史报告保留；本轮不继续增加旧 CTA 比例规则或要求重做 P0。

### 历史 P0 阶段记录

2026-10-20 起，主线改为优化原生 SGLang 的 DeepSeek-V4-Flash 吞吐。32并发用于首轮控制变量，后续可调整实际batch和并发。先执行原生 P0，记录真实 MoE 分支、routed/shared 时间区间、Marlin 启动资源和四rank差异，再决定第一项优化。

入口：[原生 P0 说明](docs/NATIVE_P0.md) · [服务器执行 Prompt](docs/SERVER_PROMPT_NATIVE_P0.md)。原生 P0 已回传 H100 采样证据；当前新增配对检查完成本地验证，实际配对GPU执行待回传。P0 不修改 CTA 配额，不输出正式加速比。

最新诊断：服务器 `172bd60` 已回传原生 P0 图与分阶段证据，但同题输出长度明显变短。下一步使用[同32题配对检查](docs/NATIVE_P0_PAIR.md)及[服务器执行 Prompt](docs/SERVER_PROMPT_NATIVE_P0_PAIR.md)，分别核对未采样与P0采样的生成行为。差异只记录，根因尚未确定。

固定前缀入口已修正prefill页预算：6059个原始token按256页对齐后需要9472，五组统一使用16384预算，并在请求前核对服务实际限制。旧27＋5拆批失败记录保留，新GPU重跑待回传。

当前跟进：[固定前缀诊断](docs/FIXED_PREFIX_P0.md) · [服务器执行 Prompt](docs/SERVER_PROMPT_FIXED_PREFIX_P0.md)。使用短的相同前缀对照、原生重复和P0事件/缓冲拆分，并补原生kernel trace。根因仍待服务器证据，有限数值差异只记录。

## 历史方法身份与状态

- **`hopper_fp8`：Hopper 算子候选**。独立专家计算 + NCCL slot SUM + finalizer；未执行论文persistent调度。已回传的1894.69 vs247.69 token/s及约80%专家处理占比属于这条候选路径，不能作为论文Weave性能结论。
- **`hopper_persistent`：论文机制集成的新路径**，标识`hopper_persistent_paper_v1`。在同一persistent grid内实现Hopper设备tile、路由驱动c/K选择、chunk流水和bubble stealing。服务器提交 `a848782` 已回传机制检查与实际 C32 graph 服务报告（`paper-reproduction-144dd5f6c7ca`、`gpqa-c32-833ac7472986`）。报告标记服务执行已验证；机制范围及完整论文复现仍应分别审阅，不由路径名推定。
- 不因路径名或开关宣称“完整Weave”。必须区分源码实现、kernel机制证据、43层服务证据与完整论文实验复现。历史报告不改写。

历史入口：[论文机制实现说明](docs/PAPER_REPRODUCTION.md) · [服务器执行Prompt](docs/SERVER_PROMPT_PAPER_REPRODUCTION.md)

## 历史阶段记录

本项目使用原版 DeepSeek-V4-Flash 混合精度权重，routed专家保留MXFP4/E8M0。历史服务器提交2910864已回传完整374点校准、native/Weave eager/graph验收和两组各1200秒的GPQA预算对照。共同完成4题上，SGLang为107.64 tokens/s，当时Weave为11.31 tokens/s，约0.1051倍；全448题未完成，当轮Weave另有6次cache_flush_error。原始42文件摘要已核对。

P0/P1的2408点真实trace回放已在09f905e回传：热路径缓存有效，staged直接tile映射相对132 CTA的配对时间比中位数约1.68；这是计算回放结果。随后714af26回传的两次C32作业均通过core/online GPU门，但在cached persistent生成token验收处停止，staged服务吞吐尚未测得。

独立 Hopper FP8 激活/MXFP4 候选已在 `dbe03bd` 回传完整 1376 点真实 B32 回放，数值差异采用 record-only。相对 BF16 的配对时间比几何均值为 eager 2.749、graph 2.768；这仍是本地专家计算时间。`e8498aa` 已回传三组实际 C32 eager 服务吞吐：Native 235.47、BF16 staged 75.68、Hopper FP8 166.40 tokens/s。Hopper 相对 BF16 提升到 2.20 倍，但首 token 等待仍长；`5629aaf` 已完成诊断：大批次 persistent prefill 约99.5%的模型区间落在 routed experts，最长停顿与 prefill 高度重叠。当前新增独立紧凑多tile Hopper prefill，实际GPU验证与性能待服务器回传。历史 `a60a250` 的旧阈值失败记录保留。

## 入口

- [GPQA Main：两组 C32、各20分钟对照](docs/GPQA_C32_COMPARISON.md)
- [本轮 GPQA C32 服务器执行 Prompt](docs/SERVER_PROMPT_GPQA_C32.md)

- [大行数 Hopper FP8 prefill 与三组 C32 对照](docs/HOPPER_PREFILL.md)
- [本轮 prefill 服务器执行 Prompt](docs/SERVER_PROMPT_HOPPER_PREFILL.md)

- [Prefill、首 token 与输出停顿诊断](docs/PREFILL_STALL_DIAGNOSTIC.md)
- [本轮诊断服务器执行 Prompt](docs/SERVER_PROMPT_PREFILL_STALL.md)

- [Hopper FP8 实际 EP4 decode 与 C32 对照](docs/HOPPER_C32_SERVICE.md)
- [本轮服务器执行 Prompt](docs/SERVER_PROMPT_HOPPER_C32.md)

- [Hopper专家计算候选与真实B32对照](docs/HOPPER_EXPERT_CANDIDATE.md)
- [Hopper候选服务器执行Prompt](docs/SERVER_PROMPT_HOPPER_CANDIDATE.md)
- [本轮候选本地验证与服务器待办](reports/hopper_expert_candidate_validation.json)
- [Hopper误差只记录策略与验证](reports/hopper_record_only_validation.json)
- [Staged-direct实际decode与C32服务对照](docs/STAGED_C32_SERVICE.md)
- [本轮服务器Claude Code + DeepSeek V4 Flash执行Prompt](docs/SERVER_PROMPT_STAGED_C32.md)
- [本轮本地验证记录](reports/staged_c32_service_validation.json)
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

当前目标：原版 DeepSeek-V4-Flash 混合精度权重，在调度器分配的四张 H100 上以 EP=4 比较原生 SGLang 与完整 Weave。两组采用相同的专家划分和 attention/dense 并行布局；原生Marlin专家EP4已在服务器验收中执行。SGLang 的 TP 参数可与 EP 参数同时使用，不能把 `--tp-size 4` 等同于 MoE 只使用 TP。当前GPQA Main使用32并发、Think High、最多生成32K（32768）tokens，每组正式测量1200秒，加载和预热另计。论文机制实现状态以上方说明为准。旧 Qwen 设计、计划和交接文件保留为历史记录，不能直接作为 DeepSeek 实现合同。

已有代码包括：只读预检、DeepSeek GPQA token ID 准备、decode 计数、EP 路由去重/合并 CPU 参考，以及原生 EP4 Docker 启动入口。现已加入 CUDA IPC persistent 功能核心、GPU 端 c/K 选择和连续 stealing、raw MXFP4 专家加载器，以及源码核对后的 SGLang 0.5.17 eager adapter。默认使用当前服务器已有的 `lmsysorg/sglang:v0.5.17`，工作目录为 `<repo>/work`。后端名为 `weave_functional`：历史 H100 CUDA 编译、两轮四卡小张量功能与 core graph replay 已通过并核对原始报告；修正后的 BF16 WMMA、真实 checkpoint 与 batch=1 SGLang graph 在 e59e064 的服务器摘要中报告通过。9906ca7c源码的完整rank/coverage、实测bank及GPQA报告已回传；新增诊断源码需要重新执行源码绑定的H100 gates。证据见 `reports/core_acceptance.json`，状态见 `reports/implementation_status.json`。

当前服务验收中，logprob 和 top-k 候选集合仅保存诊断，不作为通过/失败门限；旧入口的实际生成 token 一致性、checkpoint、43 层执行与 graph replay 按原验收入口检查。新 Hopper C32 套件的 token 内容差异也只记录，仍要求有效完整响应和实际路径覆盖。

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
