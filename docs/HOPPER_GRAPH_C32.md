# Weave Hopper：C32 CUDA Graph 适配

## 本轮实现

Native SGLang 与 Weave 均使用 full decode CUDA Graph；为 batch 1–32 每个整数分别捕获图，并关闭 graph padding。服务最多同时处理 32 个请求，实际 decode batch 随请求完成及调度变化。两组 prefill graph 均关闭；Weave prefill 保留已优化的 Hopper FP8 实现。

新的 `HopperGraphDecodeRuntime` 复用原有 workspace、绑定权重、FP32 slot buffer、BF16 output buffer 和 Triton finalizer。每个 batch 先完成 eager warmup，再捕获专家计算、跨 rank slot SUM、设备计数器更新与最终聚合。重放时不依赖 Python forward 更新计数，不在 forward 中读取 GPU 值或分配输出 buffer。

当前适配范围为既有 SGLang 0.5.17 的固定源码、4 张 H100、TP4/EP4、expert TP1、attention DP1。新 `graph_hooks.py` 校验原生源码身份，并为 Native/Weave 使用同一 observer。运行时直接记录成功重放次数、原始 batch/图 bucket、捕获次数与 eager forward；启动参数本身不能证明重放。

## 运行与证据

入口沿用 `scripts/run_gpqa_c32.py`，增加 `--decode-graph full`。不传该选项仍使用原 eager 对照。固定 GPQA Main 448 题、32 并发、各 1200 秒、Think High、最多 32768 输出 token、自然 EOS；请求与吞吐口径见 [GPQA 对照说明](GPQA_C32_COMPARISON.md)。

执行顺序：

1. 校验已有完整 prefill 数值证明、镜像、checkpoint 和源码身份。
2. 在已有镜像内运行新的四卡 graph smoke。两个绑定权重 handle，各捕获 batch 32→1，共 64 张图；执行 38 次重放，覆盖所有 batch、变更输入与路由、跨 bucket/handle 切换、eager B65 prefill 后重放。
3. 顺序加载 Native 和 Weave，完成图捕获、预热、缓存清理后，各测量 1200 秒。
4. 汇总四 rank 的真实 graph 重放和 Weave 43 层 GPU 执行证据，再导出公开报告。

smoke 使用缩小的 H128/I128/E32 模型形状，检查 NCCL、finalizer、地址和设备计数。真实 H4096/I2048/E256 的 43 层由服务启动与正式窗口覆盖验证。新增 graph smoke 不等同于真实模型性能结果。

新证据包括：

- `graph-gate/hopper-graph-rank{0..3}.json` 与 `graph-proof-check.json`；
- 两组 `graph-before.json`、`graph-after.json`、`graph-check.json`；
- Weave 的 `coverage-before.json`、`coverage-after.json`、`coverage-check.json`；
- `comparison.json` 中的 `decode_graph: full` 与既有吞吐和数据耗尽标记。

正式窗口要求：每 rank 已捕获 1–32 全部 bucket，实际出现 B32 重放；raw batch 与 captured batch 相等；四 rank 重放分布一致；窗口内没有新增捕获、失败重放或 eager decode 回退。Weave 每层设备 decode 次数应等于重放数，设备 slot collective 次数应为重放数 ×43；prefill 次数及其 collective 单独统计。

数值误差与生成内容差异保持 `numerical_policy: record_only`。只因非有限输出、非法内存、编译/运行失败、无效路由、身份/覆盖缺失或清理失败停止，不新增精度门限。

## 边界与代价

- 原 CUDA 专家内核、原 eager decode/finalizer/prefill 文件保持不变；新增 graph runtime 单独记录源码 hash。
- 捕获 32 个 bucket 会增加启动时间和 graph 内存；若启动内存不足，保留失败证据，不能静默缩减 bucket 或退回 eager 后标成 graph。
- 既有实验记录保持原样。图适配的本地 CPU 检查不能证明服务器 CUDA 编译、NCCL 捕获/重放成功，也不能证明吞吐提高。
- 服务器只从 GitHub 拉取本仓库和固定数据；继续使用已有镜像、包、编译缓存及只读模型。无在线安装或镜像拉取。

具体执行见 [服务器 Graph Prompt](SERVER_PROMPT_HOPPER_GRAPH_C32.md)。
