# Native Marlin peer-tail executor

方法身份：`native_marlin_peer_tail_v1`。成本表：`measured_tail_dag_v1`。
开发基线为 `33d071998fc5bcfbe82cf0bc526a1a3abae06b61`；上游为 SGLang
`0.5.17 / 29481685462732237d80d86076d6563e1f658102`。

本阶段以提高原生 SGLang 吞吐为目标，采用实施计划的 A 路线：保留复制输入和
本地 EP 专家计算，只重构 decode 的 down、局部合并和输出归约。没有新增
输入 dispatch all-to-all。原生 native-budget/native-cost 代码和历史报告保留。

## 已实现的源码与证据边界

| 阶段 | 源码及实际调用关系 | GPU 要检查的内容 |
|---|---|---|
| S1 | `native_tail_patch.py` 从固定 Marlin 提取 `run_marlin_down_tile`；静态 kernel 与统一 kernel 调用同一 helper | 真权重、全部 M block、完整 K、异步 scratch 复用、数值记录、资源与延迟 |
| S2 | `task_protocol.cuh / peer_reduce.cuh / transport.cu`，四 owner 的 IPC pull 归约 | peer/native atomics、cooperative graph、epoch、延迟 rank、取消、ack、slot 复用 |
| S3 | `executor.cu` 同一 cooperative resident grid 中领取 down、finalize、通信任务 | 精确任务集合、空本地工作、resident 上限、不同角色实际执行 |
| S4 | `native_tail_front.py / runtime.py` 拆出原生前半段；输出通道 chunks 1/2/4；decode helper 直接返回完整输出 | 43 层实际 graph、chunk 0 通信与后续 down 的同卡真实区间交叠 |
| S5 | COMM worker 在初始空隙内领取同一 down 队列，每个 tile 后重查 ready；保留一个进度 worker | COMM 实际算 down 后恢复通信，受控与自然观察分开 |
| S6 | `native_tail_cost.py / cost_model.h / native_tail_calibration.py`，实测分块 DAG、GPU 联合选择与计划共识 | 当前 replay 的 CPU/GPU 选择一致，完整 MLP 留出误差与 regret |
| S7 | `run_native_tail.py`，基础检查、memcheck、校准、测量表验证、两个完整窗口 | native 与 peer_tail 各1200秒、graph/调用覆盖、数据状态、完整清理 |

这些是**源码实现**。本地 macOS 的 CPU 测试不证明 CUDA 编译或 H100 执行，
也没有产生新吞吐数字。报告分别保留 `source_implemented`、
`gpu_mechanisms_verified`、`full_model_graph_execution_verified`、
`throughput_result_observed`。自然窗口 stealing 为零或计划不变化时照实报告。

S1 独立提交：`5b4b7dc04f79cbfbed36f6be5d595a76039effe9`。
S2 独立提交：`9b0668c73cce228772b7df3dd209f118631c5e72`。
后续统一源码应从本轮最终执行提交运行；不要把旧 S1/S2 提交当作完整服务版本。

## 2026-10-22：S1 编译失败及修复

服务器回传 `c079179` / `native-tail-816d1423a582` 在原执行提交 `2ff0ba7` 的
S1 编译阶段停止；NVCC 报告 dtype/weight-width 的 static_assert 和未解析的
`div_ceil`。通信、统一执行器、校准及两组正式窗口均未启动，旧失败证据保留。

修复提取器并重新生成 helper：将 scalar/weight/scale 恢复为具有固定原生默认值的
模板参数，使原生 `if constexpr` 的其他精度分支在实例化时正确丢弃；保留原生
断言，并约束实际入口仍为 BF16/MXFP4/E8M0。在 helper 函数内部显式导入
`device::div_ceil`，保持 host 配置函数对 `host::div_ceil` 的解析。
原生加载、反量化、MMA、CTA 归约和写回区域逐字保持，运行矩阵及 record-only
策略不变。新版 overlay 按内容生成新缓存身份，已有缓存和历史报告无需清除。

新增 CPU 编译契约检查读取实际生成头文件中的模板声明、类型断言、packed-width
分支与整除调用。它验证 C++ 类型依赖和名称解析，不模拟 CUDA，也不等于 NVCC
编译或 GPU 执行通过。请从最新服务器 Prompt 固定的修复提交重跑统一入口。

## 2026-10-22：受控延迟下的 CTA 分支修复

`04d431f` 回传的 `native-tail-c79f528d3b78` 已通过S1/S2；统一 executor 编译成功，
提交说明记录前80次replay完成，首次启用2,000,000-cycle计算延迟时发生非法访存。
该失败发生在S3，后续memcheck、校准和正式服务窗口均未运行。

源码中的延迟判断原先由每个线程读取 `clock64()` 独立求值，却包裹了含 CTA barrier
的 finalize/down 调用。不同warp在延迟边界可能作出不同选择，违反这些整CTA函数
的一致调用要求。现在由thread0计算 `compute_ready`，使用已有 `__syncthreads()`
广播后，全CTA统一决定是否进入任务。受控延迟值、stealing、算术及验收矩阵保留。

新增回归用例提取实际的poll与分支条件，在CPU上模拟跨warp时钟偏移；旧条件产生
部分CTA进入，新条件在128/256线程及边界前后保持一致。该检查确认源码控制流缺陷
已修正，不等同于GPU非法访存已消失；需要服务器在新源码上重跑原controlled用例
以及后续memcheck。旧失败报告保持原状态。

## 原生语义与方法映射

固定源码和已有图元数据表明，当前 TP4/EP4、`moe-a2a-backend=none` 使用
复制输入、本地专家、shared FP8 双流及最后的输出归约。原归约在 routed/shared
join 后，不能把它当成已有的通信/计算流水。

新 front 从原生 `fused_marlin_moe` 的 activation 后返回 `DownInputs`，不会
先执行旧 down。它保留 HashTopK 的 input IDs、原始 grouping、远端 expert=-1、
padding 和完整 tensor 引用。原生 prefill、attention 以及其他 collective 不变；
不存在全局 all-reduce no-op。

算术顺序按源码保持：

1. Marlin down 使用原始 repacked MXFP4/E8M0 权重、BF16 activation、原生
   Tensor Core MMA，并在 pair 输出中乘 top-k 权重。
2. 按原始 slot 顺序 FP32 求和，再舍入 BF16。
3. `shared + routed_scaling_factor * routed_sum` 舍入 BF16，恰好一次。
4. 新 peer 协议完成四 rank 输出归约，外层不再次加 shared 或 all-reduce。

本版明确支持已验证的 shared TP4 分片、归约前加入布局；其他 shared 布局会报错。
shared join 前移到 tail 入口，以免 resident grid 等待尚未运行的外部 shared kernel。
校准分别记录原生与新路径的 join 等待差，不能把前移等待视作免费操作。

与 [Weave §3.1–§4](https://arxiv.org/html/2609.21483v3) 的对应是：实际计算/通信
worker、就绪驱动流水、原子任务领取、初始窗口 stealing、尾部所有 worker 参与，
以及路由与实测曲线驱动的资源选择。本版的 output-channel chunk、四 owner 归约和
保留原生前半段属于适配；没有论文完整 dispatch→GEMM1→activation→GEMM2→combine
五阶段融合，不称完整 Weave 复现。gate/up 不属于可窃取计算量。

## 计算任务、驻留与协议

每项 down 任务是一个原生 expert group × 一个 N tile × 完整 K。M 分组不变；
取消原 Marlin grid stripe 和跨 CTA split-K 的锁伙伴依赖。加载、packed/scale 地址、
反量化、fragment、MMA、CTA 内归约、写回来自摘要固定的上游区域。每项任务结束前
排空 `cp.async` 并执行 CTA barrier。S1 会单独报告提取与取消 split-K 的性能代价。

统一 kernel 的线程数和 native specialization 在捕获前固定。动态 shared scratch
依据最终 kernel 的静态 shared 占用修正，并检查原生 pipeline 所需下界。cooperative
worker 数不超过最终 kernel 的 occupancy × SM count；不使用超额排队自旋。
这符合 [CUDA cooperative launch 的驻留限制](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__EXECUTION.html)。worker 配额不能解释为独占物理 SM。

队列使用有界 CAS，耗尽后不继续递增 cursor。prologue 从当前 native grouping
建立本地 group 列表、工作量和 slot 覆盖位图，容量由当前 shape 推导。每个 chunk
的所有 down 完成并由 finalize CTA acquire 后，才进行局部合并及发布。

IPC 连续布局是 `[chunk, owner, token, channel_in_owner]`。owner 任务融合拉取
四份 partial 与按 rank 顺序 FP32 归约，再写 BF16 shard；随后各 rank 拉取四个
owner 结果。每个 rank 只写自己的 workspace。每个载荷写线程执行 system fence，
CTA 汇合后 leader release 发布；各读线程 acquire 后读取。

session、layer、GPU sequence、两槽 epoch、ready、reduced、ack、done 保护重放。
所有消费者确认和 rank done 后才能复用。空本地专家仍执行 finalize/shared 及通信。
协议异常是终止错误，不能把中止后结果继续用于推理。CUDA context 被 trap 或
memcheck 判为无效时，报告清理失败/未验证，并由外部 owner 终止容器；不冒称图 reset。

COMM 数至少2时才允许有限初始 stealing，worker0保留为通信进度 worker。
stealing 调用与 COMPUTE 完全相同的 device helper。进入通信阶段后不反复回到
初始 stealing；计算和 finalize 完成后，COMPUTE worker 加入通信。serial 是
`comm=0,chunks=1,steal=0`，计算全部完成后才执行通信。

## 独立的实测成本模型

候选仅包含当前可执行机制：serial，以及 aligned chunks 1/2/4、通信 worker
1/2/4/8、steal 0/1；单通信 worker 的 steal 固定0。候选进一步受实际 resident
worker 数及 N tile / 四 owner 对齐约束。不是动态改变 captured grid 或线程数。

GPU 从实际 grouping 计算五项描述：有效本地 pairs、活跃专家数、原生 group 数、
专家行数平方和、最大专家行数。每个候选保留自己的四个实测 anchor，候选间特征
或 digest 不同只记录。不同 batch、结构类别和 rank 分别绑定表。

校准在真实 checkpoint 模型的原生 `capture_one` 非捕获 warmup 中运行，PyNccl
图状态已开启。每个结构类别选首个代表层，覆盖精确 batch 1..32；四个训练输入
为两种 spread、两种 repeated-row，留出两个不同输入。经过真实 router/front/shared，
不使用 GPQA 内容，也不填充旧 CTA timings。

完整 MLP objective graph 不包含阶段事件。另一个诊断图测量 front、shared join、
工作量准备、GPU 交换/搜索、down/finalize/owner/gather、结束确认。所有时间以
微秒保存，设备原始诊断区间使用同卡 nanoseconds。计时前的 Gloo 对齐位于 event
区间外，native 与 tail 使用同样的样本边界，避免上一轮读回拖延下一轮 peer 等待。
原始重复值、波动和实际观测对应关系一并记录，不用拼接中位数伪造一个负载样本。

DAG 先对**每个 rank 的完整前缀**求 front+prepare+planner，再取 rank 最大值。
planner 已包含 peer 到达等待，不能把独立阶段最大值再次相加。每个 chunk 的 owner
等待所有 source publication；结果收集等待 owner 完成。恢复通信成本只计 tile 结束
且本地已观察到 ready 之后的间隔，排除重复 ready 等待。前半段 marker、统计、计划
交换及搜索和 finish ack 的成本保留。

预测是有界近似：以实测同一卡 chunk 包络建模，通信资源按 chunk 顺序保守串行；
混合各 rank 的局部候选可能改变干扰，必须由留出和整模型窗口检验。不能把成本模型
的共同起点图当作观测到的跨卡时间线。GPU 每个 chunk count 保留各 rank 两个局部
候选、最多枚举16种组合，同时保留所有已测同一候选的四 rank 组合，选择预测最短的
完整路径。预测误差和 regression 均只记录。

表绑定实现/上游摘要、镜像、checkpoint、运行时、GPU、TVM-FFI 安装文件、精度、
shape、原生 specialization 和最终 executor resources。缺表、错表会明确要求运行
`run_native_tail.py` 生成，绝不回退到75%策略或猜测时间。

## 离线运行

使用已有 `lmsysorg/sglang:v0.5.17`、Linux x86_64、Python3.12.3、Torch2.11.0+cu130、
CUDA13及nvcc。原生头文件要求 C++20，统一 extension 从现有 TVM-FFI 安装解析
头文件和 runtime；不会下载依赖。模型只读挂载，Docker `--pull=never`。实际设备
来自 GPUQ allocation，不写死物理卡号。

只打印统一计划，不调用 Docker：

```bash
python3 scripts/run_native_tail.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl
```

先独立检查 S1/S2（新目录，保留缓存）：

```bash
python3 scripts/run_native_tail.py \
  --work-root /srv/weave-work \
  --stage foundations --execute
```

完整统一流程（四卡 GPUQ 分配内）：

```bash
python3 scripts/run_native_tail.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --native-tail-repeats 7 \
  --gpu-timeout 28800 --calibration-timeout 43200 \
  --setup-timeout 28800 --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

外层 GPUQ 时限至少覆盖这些阶段；初次编译、memcheck 和校准可能很久，按需要
延长分配时限，不缩短正式窗口。GPU 阶段与服务必须在同一实际分配内。
`--calibration-dir /srv/weave-work/历史运行/calibration` 仅显式复用完整匹配表，
仍执行当前 GPU 检查。默认不覆盖已有校准或历史报告。

流程顺序：S1真实权重 → S2通信能力/协议 → S3–S5统一执行器 → 实际
compute-sanitizer memcheck → 真实完整 MLP 校准 → measured GPU graph 验证 →
原始 native → peer_tail。只跑 S1/S2 不等于完成服务验证。

正式仅两组，各完整1200秒，full graph、精确1..32、关闭 padding，初始并发32。
自然 GPQA Main、Think High、temperature0、max_new_tokens32768、自然 EOS 沿用既有
客户端。不得重复题目、强制等长输出或缩短窗口。正式窗口不请求 `/tail-coverage`，
也不保存阶段事件/逐任务 timeline。额外 marker、统计、协议和 planner 仍实际运行。

预热排空后及窗口完成排空后读取 `/tail-coverage`，将4 rank×43层的 gate/up、
tail、planner、protocol commit 差分与真实 graph replay 对齐，报告计划直方图、
chunk/计算/stealing/peer bytes。真实 batch 分布、数据/补充队列耗尽、TTFT/ITL、
完整窗口 TPS 与 sustained-C32 是否可用按既有口径报告。

四个 `/tail-close` 必须并发请求：在 idle 时显式 reset 已捕获图、同步，再两阶段
关闭 peer 映射和本地 allocation。最后停止 owned container。校准私有图由校准器
单独 reset；启动失败或 context 中止保留真实清理状态。

## Memcheck 的 API 报告策略

从本次修复起，统一入口调用 `compute-sanitizer --tool memcheck` 时显式传入
`--report-api-errors no`，保留 `--target-processes all --error-exitcode 99`。
这是关闭工具的 CUDA API 错误报告；memcheck 的内存/硬件错误检查继续执行。
应用自身任何非零退出仍为硬失败，包括我们用 `check()` / `NT_CUDA` 检查后抛出的
CUDA API 异常。只有工具进程成功且四rank完整检查、来源及清理通过，才记 verified。

策略依据是服务器回传 `a6ebb0e` 的 `native-tail-06ee57bda6e0`：四rank各完成200次
统一执行器replay；随后memcheck因116条API探测错误退出99，服务器检查未报告访存违规。
已知类别是torch导入阶段的 `cuGetProcAddress_v2` 符号探测及NCCL的
`cudaFuncGetAttributes` kernel属性探测。旧失败报告保持原状态。

`configs/native_tail_memcheck_api_reference.json` 保存该历史总计、类别、来源提交和
公开manifest摘要。历史数据来源为服务器提交说明及用户对私有日志的检查；没有公开
原始日志，也没有捏造各类别的拆分数量。

每次 `executor_memcheck/check.json` 的 `memory_check` 同时记录选项、实际进程退出码、
应用非零退出规则及 `api_error_observations`：

- `historical_reference`：上述历史116条及两类API，明确不是本次测量。
- `current_run.filtered_count` / `categories`：`null`。`no` 模式不再输出这项数据，
  状态记为 `not_observable_with_report_api_errors_no`，不能写成0或复制历史116。

成功和失败出口都保留这些聚合证据，仍按原白名单导出；不新增GPU审计运行或实验组。

## 回传与解释

控制器打印 `work_directory`、`export_directory`。公开目录为静态白名单 JSON 与
SHA256 manifest；上传前重新验证每个摘要。原题、token IDs、生成文本、原始路由、
激活、权重、逐任务事件/位图、完整日志和 JIT overlay 留在服务器私有目录。
GPU rank 原始聚合报告由 controller 进一步缩减为公开 `check.json`；不要上传整个
run目录。

有限 numerical/token 差异、预测误差、计划不变化、stealing为零或吞吐下降均为
`record_only`。非有限、无效访存、编译/运行、畸形 routing/tensor、精确任务覆盖、
协议、来源、graph replay、超时和清理仍为有效性检查。

至少分别解释：静态 down 提取/取消 split-K、队列及统一编译资源、shared join
前移、peer 通信、chunk、stealing、planner 的观测与代价。S1/S2是独立成本；
统一受控测试只能证明机制，最终收益以两组完整模型窗口为准。
