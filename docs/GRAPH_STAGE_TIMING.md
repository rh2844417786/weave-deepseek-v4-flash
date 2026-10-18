# CUDA Graph C32 分阶段计时

## 目的与已有证据

`reports/server/gpqa-c32-9f1635df2afe` 已证明两组实际使用 full decode CUDA Graph、没有 eager 回退，并各完成1200秒：Native 1894.69 token/s，Weave 247.69 token/s。Weave每rank9294次重放，其中9250次B32；全部43层设备计数一致。新诊断用于确定时间花在哪里；现有吞吐本身不能判定是哪一段主导。

## 计时边界

每个B32 Weave routed-expert调用捕获四个 `torch.cuda.Event(enable_timing=True, external=True)` 节点；事件在捕获前分配并初始化：

| 区间 | 报告字段 | 实际包含内容 |
|---|---|---|
| workspace.run 前→后 | expert_pipeline_ms | 输入复制、专家分组、量化与专家计算；不是纯GEMM |
| workspace.run 后→slot SUM后 | slot_sum_ms | 跨rank FP32 slot SUM和所在流等待，包括rank到达偏差；不是纯NVLink传输时间 |
| slot SUM后→finalizer后 | finalize_ms | 设备collective计数器更新、按顺序聚合、BF16输出和origin mask |
| 首边界→末边界 | routed_total_ms | 单层上述三段总区间 |
| graph.replay 前后外部事件 | whole_graph_ms | 一次实际模型graph重放的GPU时间线跨度，包含发射间隙 |
| whole_graph减43层routed合计 | residual_ms | 其余未细分工作，含注意力、共享专家、外层TP、路由/缩放及发射间隙等；不能全部归为注意力 |

当前B32 slot张量每层每rank为192×4096×4字节，即3MiB。它是参与SUM的张量大小，不是实际链路流量；本轮不据此宣称网络带宽或通信效率。

PyTorch的[CUDA Event说明](https://docs.pytorch.org/docs/stable/generated/torch.cuda.streams.Event.html)区分了外部event节点与图内依赖。兼容性由服务器原有Torch2.11/CUDA13镜像中的新GPU检查实测；不通过安装新版本绕过。

## 采样与扰动

显式 `--graph-stage-profile` 开启，默认关闭。每128次实际B32重放采样一次，保留最近64个样本；非B32调用保持原计算路径。两组都采集外层graph时间，只有Weave采集43层内部阶段。

B32图内的各层event节点会在每次重放执行，采样时才额外执行外层end-event同步并读取时间。**同步在整次重放之后，绝不插在三个阶段之间。** 读取完成后才允许下一次重放覆盖事件；检查事件在本次外层时间窗内、层顺序和时间闭合，防止读取旧时间戳。

这会扰动服务，所有结果标为 `measurement_mode: instrumented_diagnostic`、`formal_gpqa:false`。脚本保留诊断运行的观测TPS，但禁止输出正式Weave/Native加速比。正式性能基线仍是已有未插桩报告。Native没有内部43层event，不能把两组插桩后的TPS差值当作公平性能比较。

计时在请求预热结束后的空闲覆盖快照与结束快照之间按实际replay sequence筛选；环形缓冲只保留最近64个采样点，报告不是全部1200秒的每步平均。默认采样参数下，Native与Weave保留的末段时间跨度不同；不将它们当作同一时间区间的配对样本。

## 四rank汇总

`stage-profile-check.json`验证来源、配置、采样间隔、全部43层、正有限时间和同一重放序号的四rank覆盖，输出：

- 每rank、每层的mean/p50/p95/max；
- 每rank的三段合计、routed合计、整次graph时间和剩余时间；
- 对每个对齐采样点，选取whole_graph_ms最大的rank，再使用**同一个rank**的三段构成，汇总到 `critical_rank`；
- 同一步whole_graph的rank间极差。

不将不同rank的专家计算最大值、通信最大值、聚合最大值直接相加。不跨GPU比较绝对事件时间戳。时间闭合只允许浮点/事件精度的舍入差；所有专家输出数值差异仍为record_only，不设误差上限。

如何解读：专家处理占比高，下一步细分workspace内的准备与GEMM；slot SUM占比高，先检查rank间计算不均衡和等待，再讨论传输；finalize占比高，检查聚合与输出；剩余时间高则细分注意力、共享专家及外层TP。诊断数据返回前不预判结论。

## GPU检查与作业

沿用 `scripts/run_gpqa_c32.py` 两组各1200秒GPQA负载。显式图诊断模式先运行原四rank graph smoke，再在**独立容器和目录**运行 `tests/gpu/check_graph_stage_events.py`。

新增检查使用4张H100、H128/I128/E32、真实B32路由，分别捕获诊断runtime和未插桩原graph，执行4次变化输入的重放。验证外部事件新鲜性、计时边界、实际collective、原始slot元数据、有限输出，并记录对原graph/eager/FP8 oracle/BF16 oracle/BF16 reference的误差。两张图必须reset后再销毁NCCL；正常退出和完整四rank证据通过后才加载真实模型。两次检查的120秒清理期限分别计时。

公开产物：

- `stage-event-gate/stage-event-rank{0..3}.json`及清理/运行时记录；
- `stage-event-proof-check.json`；
- 两组 `graph-before.json`、`graph-after.json` 内嵌的 `stage_profile`；
- 两组 `stage-profile-check.json`；
- 总 `stage-diagnostic-summary.json`；
- `comparison.json`的ratio为空，原因标记 `instrumented_stage_diagnostic`。

只导出标量、层号和内部重放序号，不导出题目、token IDs、激活或路由。无新增外部依赖，已有模型只读、镜像与缓存复用，GPUQ卡号不硬编码。

执行见 [服务器Prompt](SERVER_PROMPT_GRAPH_STAGE_TIMING.md)。本机CPU模拟可验证计时顺序和汇总，不能提供真实GPU阶段耗时。
