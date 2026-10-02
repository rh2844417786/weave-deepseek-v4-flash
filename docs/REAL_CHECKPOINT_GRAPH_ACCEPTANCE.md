# 真实原始权重与 SGLang graph 验收

本轮 BF16 舍入修正已在本地通过 60 项 CPU 测试和 Python 语法检查。服务器最新说明报告旧源码 checkpoint 门通过、Weave eager 数值门失败，graph 尚未执行；原始 JSON 尚未回传。修正后的 CUDA 源码须在 H100 重新编译并运行下列门槛。

## 保持的运行契约

- GPUQ 分配物理 GPU `4,5,6,7`；Docker 内 ordinal `0,1,2,3`。
- `lmsysorg/sglang:v0.5.17`、Torch 2.11.0+cu130、nvcc 13.0、Python 3.12.3、Transformers 5.12.1。镜像 content ID 在每次运行中记录。
- 模型为 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash` 原版混合精度 checkpoint，routed experts 使用其原始 MXFP4 bytes 与 E8M0 scales。
- 所有产物写入绝对可写的 `WORK_ROOT`，默认 `<repo>/work`。
- SGLang 源码锁定扩展到 graph runner、full graph backend、HTTP 输入输出与时间统计，以及专家舍入和 logprob 路径，共 20 个文件。原生和 Weave 都核对。

## 1. 只运行真实权重 CUDA 门槛

在 GPUQ 分配好的作业内：

```bash
CHECKPOINT_TESTS=1 CHECKPOINT_LAYERS=all bash scripts/run_cuda_functional_gpuq.sh
```

先运行历史小尺寸测试，再用 `tests/gpu/check_checkpoint.py` 检查 H=4096、I=2048、experts=256、top6：

- rank/expert 不同的随机原始格式权重；随后逐层读取原版 checkpoint，默认 0..42。
- 参考路径独立解包 E2M1/E8M0、FP32 PyTorch matmul，gate/up、SiLU 输出、mid/down、路由权重及逐专家加权贡献分别按 BF16 舍入。
- 同一层分别检查 scalar 与 WMMA、远程路由、空 rank、K2 和非零实际窃取。
- 每个实现捕获一次 graph，修改路由与有效 token 数后重放两次。
- CUDA 计数记录实际完成的 stolen tasks。仅设置 `steal > 0` 无法通过。
- 相对 L2 ≤0.025，另有逐元素 rtol=0.03/atol=0.02 检查。原始误差写入每个 case；失败不会生成 passed rank report。该参考检查 Weave 全局 combine；原生每 rank 的 BF16 部分和及 shared-expert/all-reduce 顺序仍由服务门检查。

`CHECKPOINT_LAYERS=0,3,42` 可以用于定位问题；完整模型流程要求全部 43 层报告。

新增 `wmma` 使用真正的 BF16 WMMA 累加。每个 warp 处理 16 个输出维度，当前一个 token 占 tile 的一行，其余 15 行 padding。它是 Tensor Core 数值候选实现，尚未建立高效 token 分组，也未把 stealing 单元改成论文的 GEMM tile。测量结果出来前不能称为完整优化版 Weave。

## 2. 一次作业串起 checkpoint 与服务门槛

服务器需先生成两组共用的 Think High token-ID 清单。服务器只能从 GitHub 获取材料：GPQA 来自作者固定版本的 GitHub 归档，随后使用本地模型文件和已有 SGLang 镜像离线编码。完整命令见 [服务器网络与离线准备](RUNTIME_0517.md#服务器网络与离线准备)，不要求服务器在线安装 Python 包。已经准备好的清单可以通过 `--records /absolute/path/main.jsonl` 指定。

在 GPUQ 分配好的作业内运行：

```bash
python3 scripts/run_model_acceptance.py
```

默认完成以下顺序：

1. CUDA 编译、小尺寸检查、全 43 层原始权重 scalar/WMMA 差分。
2. 原生 SGLang EP4，full decode graph，batch=1。
3. Weave WMMA eager，batch=1。
4. Weave WMMA full decode graph，batch=1。

服务阶段共用同一条自然长度 GPQA 输入，检查 8 个实际输出 token 的一致性。首 token 和 top10 logprobs 仅作为诊断记录：保存候选集合差异、逐项差值和最大差值，不设置 logprob 容差门限，不因 top-k 支撑集变化而中止验收。logprob 缺失或异常时标记 unavailable 并记录原因。测试为证明 decode 执行而显式要求 8 token、忽略 EOS；这仅属于功能验收。正式 GPQA 保持 Think High、32768 上限和正常 EOS。

每次请求前 flush cache，确保执行 prefill。四个 TP worker 的 localhost 30110..30113 `/coverage` 仅在 idle 时查询；查询会同步 CUDA，不能放到性能计时区间。逐层设备计数在 graph 中递增，Python 调用计数与 capture 计数单独记录。请求后的 decode device delta 大于 Python decode delta，且 capture 次数没有增加，才能建立 graph replay 证据。任何层或 rank 缺失都会失败。

每个阶段仅运行一个服务。验收入口拥有各阶段的独立进程组；请求完成、失败或 GPUQ 取消时，向该组发送 SIGTERM，由 manager 停止自身容器。即使 bash 先退出，也必须等到 manager 写出清理证据。服务清理没有确认 absent/stopped 就停止流程。此流程检查 Docker 生命周期；GPUQ 真正取消后的物理显存释放仍需服务器独立观察。

已有完整新源码 checkpoint 结果时，可以省去重新编译/逐层检查：

```bash
python3 scripts/run_model_acceptance.py --core-run-dir /absolute/work/cuda-functional.RUN
```

复用要求报告的 kernel source SHA256、checkpoint 内容 SHA256、两种实现、所有 case 和镜像 ID 匹配。内容身份覆盖 config、index 和全部权重分片；每次完整流程会重新读取约 149 GiB 权重计算 SHA256，之后在服务加载阶段核对文件属性未变。哈希与功能验收均不属于正式性能计时区间。

## 证据与后续门槛

- `<WORK_ROOT>/cuda-functional.*/runtime.json`：实际 runtime、源码 digest、core/checkpoint 状态。
- `checkpoint-progress-rank*.json`：中途进度；不能当通过报告。
- `checkpoint-rank*.json`：逐层误差、graph cases、实际窃取计数、完整 checkpoint 内容 digest。
- `checkpoint_identity.json`：config、index 与每个权重分片的 SHA256，以及这些内容组成的总体 digest。core 与验收入口分别保存。
- `<WORK_ROOT>/model-acceptance-*/acceptance.json`：整个门槛及各服务目录。
- `native-request.json`、`weave_eager-request.json`、`weave_graph-request.json`：完整请求、prepared manifest、输入 digest、完整返回、coverage 前后差分、同输入原生比较；含可解码数据，仅保留在 Git 忽略的 work 目录。
- `<WORK_ROOT>/native-ep-*/server.log`、`runtime.json`、`launch.json`、`container_lifecycle.json`：服务日志、源码锁和容器清理。

现有 `export_core_evidence.py` 会附带新的 checkpoint rank reports 和内容身份清单，并单独标记 checkpoint gate。导出仍是归档，不是新一次执行。

真实权重与服务门槛通过后，再实现/校准高效专家 tile 路径、tile stealing 与成本曲线，并执行两组正式 GPQA 对照。当前脚本没有用 synthetic rates 冒充 measured calibration，也没有把这个 8-token 功能请求记为 GPQA 正式速度结果。

## 2026-10-02 首 token 验收失败后的重跑

服务器提交了 [失败分析](MODEL_ACCEPTANCE_TOPK_GATE.md)，报告旧源码 checkpoint/native 通过，Weave eager 数值门失败，graph 未运行；原始工作目录 JSON 尚未回传。本轮修正 SiLU 中间结果、路由权重与逐专家加权输出的 BF16 舍入。旧参考实现也缺少这些边界，因此旧 checkpoint 门通过不能替代本轮验收。

新增 `expert_numerics.h` 已纳入扩展缓存和源码 SHA256。a934293 回传中使用的旧 CUDA 报告不能用于舍入修正后的源码；若尚未执行修正后的 checkpoint 门，需完整运行 `python3 scripts/run_model_acceptance.py`。按用户后续指示，logprob 和 top-k 支撑集比较已改为仅记录，不再阻断 eager 或 graph 阶段。`native_comparison` 使用 `status=recorded/unavailable`、`used_for_acceptance=false` 标识，保留差值明细与已匹配的生成 token 数。

两处修正尚未在 H100 编译/运行。EP rank 部分和与 shared-expert 合并的舍入顺序仍不同，不能据此保证端到端漂移已消失。这些数值差异继续作为诊断数据。当前服务门检查实际生成 token、输入与源码/权重身份、43 层执行、graph replay 及容器清理。单纯 logprob 差异不会导致失败。

本次 logprob 策略调整只涉及 Python 验收与记录，没有改变 CUDA 源码。已有与当前源码和 checkpoint 身份一致的全 43 层通过报告时，可使用上文 `--core-run-dir` 复用；历史失败作业的状态不因规则调整而改写，未执行的 graph 仍须实际运行。
