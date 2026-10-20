# 原生未采样 / P0 采样：同一批请求配对检查

## 要回答的问题

服务器 `172bd60` 回传的 P0 诊断完成，但与上一轮原生组共同完成的265道题相比，输出长度中位数由3201降到96。本轮检查：在相同镜像、模型、输入和生成参数下，开启 P0 是否仍伴随生成长度、结束原因或输出内容变化。

源码审查尚未发现明确的错误张量写入、参数替换或原生双流依赖移除。已确认两类干预：

- 捕获阶段：B32图新增 external event 节点，并保留43层分组缓冲，改变显存分配的生命周期。较小 batch 不写同一套事件，但共享图内存池的分配布局仍可能受影响。
- 运行阶段：每128次实际 B32 replay 才同步并读回计时/分组数据，首次采样还采集一次 CUPTI trace。

这些是待检验的解释，尚不是根因证明。本次不修改原 P0 实现，以免在复现前改变被检查对象。

## 固定配对设计

| 项目 | 两组共同设置 |
|---|---|
| 输入 | 已验证 GPQA Main prepared 文件按seed42既有文件顺序的前32题 |
| 队列 | 一次32并发，不补题、不循环、不补长 |
| 预热 | 同32题，每题最多8token；预热后立即空闲清缓存 |
| 生成 | Think High已编码进输入；temperature0，正常EOS，每题最多32768token |
| 服务 | 两次独立新建的原生SGLang服务，TP4/EP4，full graph精确1..32，关闭padding |
| 流输出 | stream interval1、incremental streaming output，两组相同 |
| 默认顺序 | 原生未采样 → P0采样；支持显式 `--order p0,native`，不自动追加运行 |
| P0配置 | 保持stride128、capacity64；未采样组完全不启用P0 |
| 结束条件 | 32题均完整结束；每组默认3600秒超时保护，可按需增大 |

这不是两组各20分钟的吞吐实验。3600秒是卡死保护，不是固定统计窗口；完成即结束，不补空闲时间。prepared loader保留的 `source_workload_identity.seconds=1200` 是旧负载来源元数据，本轮实际时限由 `timeout_seconds` 明确记录。

实际输入顺序与去掉动态rid后的完整请求体均写SHA256承诺。已验证数据的32题选择SHA256：

`a5748d54c4605d57042784ad3e2de569ae017796411239a360f90aaabc9b3018`

请求和启动配置必须相同，允许的启动差别仅为P0环境变量及所属容器/输出目录。两组使用同一GPUQ分配和同一镜像ID；同一批请求不保证请求抵达时刻与每一步batch轨迹完全相同，图直方图分别保存。

## 输出与判读

每题记录输入承诺、完整输出token数、结束原因类型、UTF-8输出字节数、完整输出文本SHA256、客户端接收时间，以及在实际观察到的1、8、32、128、512、2048、8192个完成token处的累积文本摘要。

文本按服务器增量流累积，不把每次文本块误当作完整前缀。零token增量的文本块也计入摘要。没有观察到的前缀检查点保留缺失，不插值。有效结束但解码文本为空，也只记录空文本摘要。

摘要比较是**文本摘要比较，不是token ID逐位比较**。“第一个观测到不同的前缀”也不是精确的首次分歧token位置。原文、token IDs、matched stop字符串和请求rid不导出。

所有输出长度、结束原因和内容摘要差异均 `record_only`，不存在一致率或误差门限。只要请求身份、完整有效响应、来源和清理检查通过，即使32题内容全部不同，配对检查也正常完成。超时、HTTP/协议错误、非有限值、实际请求不一致、图回退或清理失败仍保留为运行问题。

### 采样是否真正发生

报告分别记录实际B32 replay和P0采样计数。短输出可能使batch很快低于32，导致整个32题cohort都没有达到第128次B32 replay。这时记录：

`p0_execution_scope: captured_instrumentation_without_runtime_sampling`

不会因此延长输出、重复题目、改变stride或宣称CUPTI已经采到。

- 如果生成不同，而测量前后采样次数都为0：本次运行中的CUPTI/D2H采样不能解释该差异，应继续检查捕获事件和缓冲生命周期等因素。
- 如果没有观察到差异且没有触发采样：只能说明这个32题cohort未复现，不能排除连续负载下触发采样后的问题。
- 如果出现差异且触发过采样：这是P0干预相关证据，尚不能只凭前后计数判定差异发生在首次采样前还是后。不要把某道题的第128个token当成全局第128次B32 replay。

## 入口与证据

```bash
python3 scripts/run_native_p0_pair.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --setup-timeout 7200 --warmup-timeout 960 \
  --request-timeout 3600 --drain-timeout 60 \
  --execute
```

只使用服务器现有Python3.12、镜像v0.5.17、只读模型和prepared数据。保留缓存，无在线安装或下载。GPU分配来自GPUQ，不固定物理卡号。

P0 CUDA代码与采样代码未变，入口重新核验并复用 `gpqa-c32-dcb642236285` 中来源和镜像一致的事件/CUPTI smoke，避免重复无关验收；不依赖Hopper候选计算证明。若证明或环境不一致，保留失败，不伪造结果或自动换镜像。

公共结果包括 `comparison.json`、`selection.json`、`launch-comparison.json`、两组 `paired-client-summary.json`、图前后快照和检查、运行身份与生命周期、checkpoint身份及smoke复用证明。成功/失败均导出带哈希manifest的白名单目录；完整日志和原始trace留在服务器工作盘。
