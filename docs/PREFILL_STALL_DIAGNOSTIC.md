# C32 prefill、首 token 等待与输出停顿诊断

## 为什么做这一轮

`e8498aa` 的完整 C32 eager 对照中，Native、BF16 staged 和 Hopper FP8 分别为 235.47、75.68、166.40 decode tokens/s。Hopper 已比 BF16 staged 快约 2.20 倍，但平均 TTFT 仍为 42.69 秒，Native 为 0.857 秒。两者平均客户端输出 token 间隔分别为 137.57、134.92 ms。Hopper 还有约 67.79 秒的最长输出间隔。

这些统计提示首 token 前等待与长停顿值得优先调查，但不能单凭 TTFT 断定 GPU prefill 耗时。本轮增加观测，保留原来的计算、调度和负载。

## 本轮回答的问题

1. 请求进入 scheduler 后，在首次 forward 前排队多久？首次 forward 到 prefill 完成多久？
2. 每次 prefill 的完整模型 GPU stream 区间有多长？其中 43 层 routed experts 占多少？
3. 客户端最长输出间隔是否与 prefill GPU 区间重叠？有没有没有 prefill 重叠的长停顿？
4. 同一次 forward 的四个 rank 是否对齐？最慢 rank 与其他 rank 的时间差多大？
5. 抽样 decode 的整模型与 routed-expert 区间各有多长？

## 观测方式

三组仍是 Native、BF16 staged、Hopper FP8，统一 TP4/EP4、eager、C32、原 custom 数据和每组 600 秒窗口。数值误差与生成内容差异仍只记录。

共享的源码校验 hook 记录每次模型调用的主机起止时间。每次 prefill 和每 32 次 decode 中的一次，再记录完整模型及 43 层 routed expert 的 CUDA event 区间。事件预分配；forward 中不调用 synchronize、不读取 elapsed_time、不下载路由或激活。最多保留 16000 次 host 调用、256 次 GPU 采样，溢出明确报告，不静默覆盖。

模型与层级耗时直接用各自起止 CUDA event 测量；相对 anchor 的偏移只用于时间线对齐与近似窗口裁剪，避免长窗口中的短耗时相减精度损失。CUDA 13 的 elapsed-time API 返回 float 毫秒值，见 [NVIDIA 官方 API](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__EVENT.html)。

每个 rank 在空闲时校准 CUDA anchor 与 host perf_counter 的时间区间。客户端预热结束后才启用采集；测量结束、请求中止并清理缓存确认空闲后读取 event。四个 rank 的时长按同一次调用报告最大值、中位数和差值，不把并行 rank 耗时相加。

客户端只保留匿名请求序号、时间、输出计数、允许导出的服务器时间字段，以及最长的 64 个输出间隔。SGLang 的 scheduler 时间字段每个输出可见；完整 API 字段仅请求结束时可见。缺失字段单独计数，不填成零。

## 如何解读

- **scheduler queue 时间**：wait_queue_entry 到 forward_entry；它反映 scheduler 首次调度前的等待。
- **scheduler prefill 区间**：forward_entry 到 prefill_finished；可能包括 chunk 间调度，不等于纯 GPU 计算。
- **模型 GPU stream 区间**：CUDA events 之间的 stream 时间，包含该 stream 上的等待及提交不足造成的空档。
- **routed expert 区间**：包括自研专家分组、通信、计算与组装，不能叫纯 GEMM 时间。
- **客户端输出间隔**：HTTP/SSE 到达时间，包含服务、传输与客户端调度。
- **时间重叠**：提供关联证据，不自动证明因果。跨时钟关联必须同时看 anchor 校准区间；未抽样 decode 没有完整 GPU 分解。

所有新增结果标为 `instrumented_diagnostic` 和 `used_for_formal_performance: false`。本轮有额外观测开销，不能用它替换 e8498aa 的正式 C32 吞吐对照。相同负载持续运行 600 秒，不同组处理的请求数量仍可能不同。

## 执行与交付

入口是 `scripts/run_prefill_stall_diagnostic.py`，复用原 C32 控制器、完整 Hopper 回放包、四卡组装检查和已有缓存。额外先运行四卡 CUDA event 诊断 smoke。服务器仅用已有镜像、模型与包，无在线安装或镜像拉取；GPU 编号来自 GPUQ。

完整命令见 [服务器执行 Prompt](SERVER_PROMPT_PREFILL_STALL.md)。公开回传包括来源清单、smoke、客户端匿名时间、四 rank 时间线及分析汇总；输入、生成文本、token ID、激活、路由和完整日志留在服务器。

本机 CPU 测试和源码审阅只能确认实现；实际 CUDA event、SGLang hook 与停顿归因须由本轮服务器证据确认。
