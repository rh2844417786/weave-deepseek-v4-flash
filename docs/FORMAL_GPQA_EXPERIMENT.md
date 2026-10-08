# 分组 tile、实测校准与正式 GPQA 对照

本轮交付的是新代码和完整服务器作业入口。Mac 已完成 CPU、主机 C++ 和语法校验；新 `tiled` CUDA、实测 profile 和正式速度结果需要服务器执行后产生。此前 e59e064 的单 token WMMA 通过记录不能替代这批新源码的验收。

## 服务器运行

先按 [离线准备说明](RUNTIME_0517.md#服务器网络与离线准备) 准备 Main 448 题 Think High token ID 清单。服务器使用已有 `lmsysorg/sglang:v0.5.17`、本地原版混合精度 checkpoint，不安装在线依赖或拉取镜像。

在 GPUQ 已分配物理 GPU **4,5,6,7** 的作业内，仓库根目录执行：

```bash
git pull --ff-only
python3 scripts/run_full_experiment.py
```

默认工作盘为 `<repo>/work`。其他工作盘需使用绝对可写路径：

```bash
python3 scripts/run_full_experiment.py --work-root /absolute/writable/work
```

作业有编译、全权重检查、校准和五次服务加载；GPUQ 总时限应覆盖这些准备时间及两组最多 40 分钟的正式运行时间。每次校准测量限时 1200 秒，默认最多四次调用恢复未完成数据；每组正式服务 ready 后的预热和测量合计限时 1200 秒。GPUQ 排队时间不计入这些窗口。

## 必需阶段

1. 编译新 CUDA，执行小尺寸检查、全部 43 层 scalar/WMMA/tiled 独立差分、空 rank、实际 stealing、graph replay；分组检查覆盖 K=1/2/4/8。
2. 使用原始第 0 层专家权重，对 c/K 与 token bucket 测量真实 CUDA events。保存四 rank 的原始样本；完整数据通过身份、哈希和重新推导检查后才能发布 measured profile。
3. 用实测 profile 启动原生、Weave eager 和 Weave graph，核对同一输入的 8 个输出 token，以及四 rank/43 层执行、graph replay 和清理。
4. 原生 full graph 正式组。
5. Weave tiled + 实测动态调度 + full graph 正式组。
6. 只对双方共同完成的原始 GPQA 行计算加权指标并导出证据。

失败或不完整校准会停止后续阶段。正式 Weave 使用 measured profile；固定配置的功能路径不能替代这一组。

## 计算与校准

设备端按专家和 chunk 把最多 16 个 token 分组，一个 CTA 用四个 BF16 WMMA warp 计算 64 个输出列。gate/up 与 down 分别领取 tile；down 依赖全部 gate/up tile 完成。通信 worker 的 stealing 单位也变为 tile。

模型直接加载原始 MXFP4/E8M0，保持 BF16 SiLU 与路由加权舍入边界。校准同时记录逻辑 FLOPs、实际分组数、填充后的 WMMA FLOPs、去重输入与 FP32 返回字节数。GPU 调度器为候选 K 计算实际路由的分组数，避免稀疏专家路由套用满 tile 的乐观工作量。

profile 绑定 kernel/source、H100 设备、rank、Torch/Python/CUDA、镜像和 checkpoint；包含 token bucket 1/64/256/1024/2048。加载时核对原始 timing 文件 SHA256，并从样本重新推导候选速率。通信模式包含 dispatch/return 和协议开销，因此其数值是该实现的有效带宽，不是纯硬件峰值。

## 正式请求与计时

Main 448 题按固定 seed 42 顺序，保留自然输入、Think High，temperature=0、单请求、正常 EOS、最多生成 32768 tokens。最多两条预热，预热窗口最多 120 秒；它占用同一个 1200 秒总预算。每条请求前清理 prefix cache。

复用 prepared 数据时，模型验收和正式入口共同校验 Main CSV 的固定 SHA256、实际 tokenizer/encoder 哈希、Think High 模式与 high effort、448 行顺序及 token 清单哈希。缺失或矛盾的 manifest 需按离线准备说明重新生成；仅 config.json 可以保留原准备器允许的本地差异，并须记录有效 observed 哈希与一致的 match 标记。

两组启用相同的惰性计时 hook、metrics、每 token streaming 和 incremental output。时间来自锁定版本 SGLang 的服务器输出处理边界，使用 native first/finished `perf_counter` 时间；单 token 的 first/last 相同。客户端 SSE 到达时间和 chunk 数不用于 decode 计时或 token 计数。

decode TPS 使用 `sum(n-1) / sum(last-first)`，reasoning 和答案 token 都计入；n<2 的 decode 指标为空。输出还包括 TTFT、TPOT、输入/输出 token 数、完成/超时数、finish reason 和共同完成行 ID。logprob 继续仅记录。长生成内容的 hash 差异是质量诊断。

超过预算、缺少终止事件、计数/计时身份异常的请求保存为部分记录，不进入完成样本的吞吐。20 分钟可能不足以完成 448 题：结果必须同时报告已完成、超时、未尝试及共同完成数量，不能称为全 448 题结果。共同完成集合为空或没有 decode 工作时，速度比为空。

## 恢复与证据

标准输出会给出本次 `work/full-experiment-<ID>/`。新源码通过报告可以复用；旧 e59e064 core 的源码身份不同，会被拒绝。

```bash
python3 scripts/run_full_experiment.py \
  --core-run-dir /absolute/work/cuda-functional.RUN \
  --calibration-dir /absolute/work/calibration.RUN
```

已经通过同一源码、权重、实测 profile 的模型验收时，另可指定 `--model-acceptance-dir`。校准恢复使用所有 rank 共同完成的 case 交集，保留被排除的旧样本；过期或篡改的文件拒绝加载。

完整作业成功、失败或被取消后都会尝试导出到 `reports/server/full-experiment-<ID>/`。导出错误记录在工作报告；目的地未创建时可手动导出已有证据：

```bash
python3 scripts/export_experiment_evidence.py --run-dir /absolute/work/full-experiment-ID
git add reports/server/full-experiment-ID
git commit -m "Record tiled measured GPQA experiment"
git push origin main
```

导出包括指标、运行身份、core rank 记录、校准原始样本、模型覆盖矩阵；生成 token IDs、题目和正文保留在工作盘 `gpqa/` 中。源文件逐一附 SHA256。清理只作用于本作业创建的命名容器；外部 GPUQ 取消后的物理显存释放仍需服务器观察。

模型验收的公开 logprob 诊断仅保留匹配状态、数量和 delta 汇总，不包含生成或候选 token ID；诊断仍不作为门限。

校准遇到 CUDA launch failure 时，按 [单点诊断与恢复](CALIBRATION_FAULT_RECOVERY.md) 执行。设备等待保护与1200秒测量窗口分开，失败点保留rank/case/iteration/阶段；退出码1仍停止，避免把可重复的设备故障当成普通截止。
