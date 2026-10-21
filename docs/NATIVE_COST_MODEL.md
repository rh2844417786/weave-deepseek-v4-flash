# Native Marlin measured MoE cost model

本阶段的方法标识为 `native_marlin_measured_cta_v2`，表策略版本为 `measured_moe_surface_v2`。目标是在原生 SGLang 中提高模型吞吐。初始并发32是本次对照的控制变量。`run_native_budget.py`、`native_marlin_dynamic_cta_v1` 及已有报告保留原有语义；本阶段使用新的 `scripts/run_native_cost.py`。

## 成本模型与执行范围

### 与原文及当前原生路径的对应

依据 [Weave §3.1–§4](https://arxiv.org/html/2609.21483v3)，本轮采用“路由确定工作量、实测资源性能、设备端预测并选最小成本”的决策原则。原文的通信 SM 数、chunk 流水和 stealing 依赖其统一 persistent 执行器；本轮使用现有 Marlin 的有效 CTA 控制。

固定源码 [DeepseekV2MoE.forward_normal_dual_stream](https://github.com/sgl-project/sglang/blob/29481685462732237d80d86076d6563e1f658102/python/sglang/srt/models/deepseek_v2.py#L938) 先建立备用流依赖，主流执行 routed 分支，备用流执行 shared 分支，主流 join 后再归约。当前 `moe-a2a-backend=none` 使用复制输入和本地专家计算，没有跨卡 dispatch/combine all-to-all。已有 [四 rank 图元数据](../reports/server/fixed-prefix-6e89ff3a9198/p0_full/measure/graph-after.json) 的43层均记录 `forward_normal_dual_stream`，`shared_tp1=false`；shared FP8 gate/up 分片为 `[1024,4096]`，down 为 `[4096,512]`。该元数据用于确认入口和布局，不用受 P0 干扰的时间估计正式重叠收益。

| 原文机制 | 本轮对应与范围 |
| --- | --- |
| 路由后的计算/通信工作量 | 当前有效路由、专家行数分布、padding blocks 和两个 GEMM 的实际几何；归约消息尺寸由 batch/hidden/dtype 决定 |
| 校准带宽与计算吞吐后预测耗时 | 校准完整原生 MoE 的候选性能曲面，同时记录两次 GEMM、shared、all-reduce 的阶段时间 |
| 通信与计算取重叠后的最大值 | 本路径的归约依赖 join；直接测量包含真实依赖的完整 MoE 图，保留串行 collective，不假设通信可提前重叠 |
| 联合搜索资源配置 | 搜索 gate/up 与 down 的九个 CTA 预算对，按最近两个已测负载锚点预测完整路径耗时 |
| 单 persistent 内核中通信/计算 SM 分工、chunk、stealing | 本轮未实现；不把 CTA 数解释成物理 SM 数，不设置假定的 c/K 候选 |

结构相同的层共享代表层曲面，权重值导致的访存/cache 差别和不同 rank 独立选择导致的归约等待差别均可能造成预测偏差。表记录代表层及全部映射层，并用独立输入和最终整模型实验检验这项近似。

固定的九个候选为 gate/up 与 down 两个预算的有序对：`(8,8)、(8,4)、(8,2)、(4,8)、(4,4)、(4,2)、(2,8)、(2,4)、(2,2)`。数字表示原生候选 launch grid 的八分之几，实际 active CTA 还受可用 tile iterations 限制。保留原生按寄存器资源选择 tile/grid 的算法；捕获的物理 grid 固定，每次 replay 根据当前真实路由选择参与工作的 CTA。

成本表的目标是完整原生 MoE 图耗时，包含原生 gate/topk、grouping、routed 两次 Marlin、shared expert 的竞争、stream join、串行 collective 和适用的 post-add。单独 gate/up、down、shared、all-reduce 耗时只作为解释证据，不能各自选最快后拼出所谓最优完整 MoE。当前原生 TP4/EP4、`moe-a2a-backend=none` 路径没有 dispatch/combine all-to-all 的通信预算。

运行时每层只执行一次有界单 CTA planner，从已有 grouping 计算五个特征：本地路由数、活跃本地专家数、padding 后的本地 blocks、专家行数平方和、最大专家行数。按结构类别和精确 batch 查表，用归一化 L1 距离选取最近两个实测 anchor，为九个候选预测完整 MoE 成本并取最小值。预测成本用完整 MoE 耗时减去固定 planner 开销，再加上搜索 planner 开销。两个 GEMM 复用同一个持久 plan；空本地工作也各记录一次逻辑调用。

这是固定 grid 内的软资源控制。它不声明精确物理 SM 所有权，不实现论文的持久通信/计算角色、通信 c、chunk K、流水线或 stealing。每个 rank 的 collective 实测时间包含跨 rank 就绪等待；把固定候选校准用于各 rank 独立选择时，是需要用 held-out 结果检验的近似。

### 跨候选路由差异只记录

自策略 `measured_moe_surface_v2` 起，五项工作量特征与 `native_routing_sha256` 的跨候选一致性不再作为门限。特征不同、摘要不同或两者同时不同，均继续校准；表校验也不要求各候选共享相同特征。

每个候选在 `workload` 中保存自己的实测特征、总行数、分组块数和路由摘要，在 `workload_difference` 中记录相对候选0的 `features_match`、`feature_delta`、`native_routing_sha256_match` 与 `global_blocks_delta`，策略为 `record_only`。表行绑定该候选自己的实测点。共享的搜索 planner 开销另记其实际 `search_workload` 及差异，不把它的路由假称为所有候选的路由。

独立验证也记录各固定候选和实际 cost replay 的工作量；CPU 预测使用当前 cost replay 的特征。若不同候选观察到不同工作量，误差与 regret 仍发布，按同一名义输入下的实际测量解释，不称为严格固定路由的比较。历史失败报告保持原状，新规则需要新目录重新校准。

## 默认完整流程

从仓库根目录运行以下命令。默认只打印离线预览，不启动进程、不检查 Docker、不读取模型或联网。

```bash
python3 scripts/run_native_cost.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl
```

在 GPUQ 分配的四卡作业内加 `--execute` 执行完整流程：

```bash
python3 scripts/run_native_cost.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --native-cost-repeats 7 \
  --calibration-timeout 43200 \
  --gpu-timeout 7200 \
  --setup-timeout 7200 --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

控制器依次执行：

1. **完整模型校准服务。** 从原始只读 checkpoint 加载真实模型，在原生分布式 `capture_one` 的非捕获 warmup 中（PyNccl 图状态已启用，跳过前置 autotune），对每个结构类别的首个代表层、每个精确 batch1..32 校准。类别涵盖实际 topk 类型、shared TP1 配置、权重形状及 dtype。四种 spread、两种 repeated-row 的确定性合成输入经过真实原生 gate/topk；不使用手工强制路由。记录实际特征和九个候选的原始重复样本、统计值、事件测量开销及分阶段耗时。默认每项7次。
2. **独立 held-out 完整 MoE 检验。** 每个类别/batch 另用两种输入模式记录九条实测曲线、预测误差、所选候选 regret 及原始 native 对成本模式的有限数值差异。此证据包含在表文件中，不使用 GPQA 文本。只有服务正常 ready、四份表完整并与四 rank 的 `/budget-coverage` 完成快照一致，才完成这一阶段。释放临时图和绑定，再停止校准服务。
3. **独立实际 GPU 检验。** `tests/gpu/check_native_cost.py` 通过已有镜像的四 rank torchrun 执行真实 checkpoint Marlin、变化 grouping 的图 replay、empty/heavy 本地工作、locks/guards、有限数值、来源和清理检查。默认 batches7、32。四份报告必须完整绑定本次表摘要；失败停止后续服务对照。
4. **原始 native 与 cost 的完整模型吞吐。** 分别启动服务，每组测量1200秒。两组都使用原始 native prefill、原始 mixed precision/MXFP4、相同 TP4/EP4、精确1..32 full CUDA Graph、无 padding、相同自然 GPQA Think High 输入及现有客户端契约。客户端 temperature0、max_new_tokens32768、正常 EOS。校准事件、临时 grouping 保留与其他 profiler 不进入正式吞吐窗口。

新实验目录为 `WORK_ROOT/native-cost-<id>`。校准表名严格为 `calibration/native-cost-rank0.json` 至 `native-cost-rank3.json`。只有加 `--calibration-dir /绝对/WORK_ROOT/旧目录/calibration` 才明确复用完整表；控制器和 runtime 仍执行身份及覆盖核验，不能因缺表或失配自动转成75%或其他 heuristic。

校准就绪超时默认12小时；校准服务自己的 worker watchdog 同步使用此值。正式服务仍使用原有1800秒 watchdog，单次模型加载/捕获由 `--setup-timeout` 控制。外层 GPUQ 作业预留至少12小时，通常应留更长时间以覆盖校准、GPU 编译检验、两次加载捕获、两个1200秒窗口和清理；需要更久就延长外层作业和校准超时，不能缩短正式1200秒窗口。

## 启动器接口与离线环境

`scripts/run_native_ep.py` 新增以下接口：

| 参数 | 含义 |
|---|---|
| `--native-budget off` | 原始 native 路径 |
| `--native-budget cost` | 本阶段实测完整 MoE 成本模型 |
| `--native-budget heuristic` | 显式 v1 启发式入口，传入 runtime 的模式仍为 `adaptive` |
| `--native-budget adaptive` | 保留旧别名及原有行为 |
| `--native-budget full` | 保留 v1 补丁控制模式 |
| `--native-cost-table-dir PATH` | 正式 cost 必需的完整表目录 |
| `--native-cost-calibrate-dir PATH` | 校准输出目录，与 table-dir 互斥 |
| `--native-cost-repeats 7` | 校准重复次数，至少2次，无额外最大值限制 |
| `--native-cost-calibration-timeout 43200` | 只作用于校准服务 watchdog |

cost 模式要求 native backend、full graph C32、request timing，拒绝混用 P0、stage profile、自定义 expert 实现等介入。table/calibrate 路径必须在 `WORK_ROOT` 内；执行时要求已核验 checkpoint digest。直接正式启动缺少四个表会给出 `run_native_cost.py` 生成命令。新模式不传入 heuristic target-percent。

环境保持已有镜像 `lmsysorg/sglang:v0.5.17`、锁定的 Python/Torch/CUDA/SGLang 契约、只读本地模型、已有 prepared GPQA 与缓存。所有 Docker 使用 `--pull=never`、离线 Hugging Face 设置；JIT 缓存复用 `WORK_ROOT/cache/weave-native-cost`。只从 GPUQ 的 `CUDA_VISIBLE_DEVICES` 获取四张实际卡，容器内使用相应逻辑设备号。服务和四 rank 检验保留 localhost/host networking。服务器不在线 pip、apt、Hugging Face 下载或拉取镜像。

现有 GPQA Main prepared 文件及相邻 manifest 必须通过客户端原有身份检查。缺数据时仅采用既有固定 GitHub 来源和 SHA256 校验，在已有镜像离线准备；不填充、拼接、重复题目以制造输入长度或延长队列。

## 身份、覆盖和解释

表绑定 cost 实现源码、原生 Marlin 上游来源、镜像 ID、完整 checkpoint digest、model config、SGLang commit、rank、GPU 型号/架构/SM数/内存、运行时版本、精度、模型几何与图契约。表内容有独立 canonical SHA256；runtime 在 capture 前完整匹配当前身份。控制器将每 rank 的表摘要和完整 identity 再与正式服务快照比较。复用表也必须先通过实际 GPU 检验。运行结束重新核验代码、数据、checkpoint 文件状态与表摘要，防止测量途中换源。

正式窗口内不请求 `/budget-coverage`。该端点同步 GPU 并回读设备状态，只能在客户端预热已排空且开始标记尚未发布时，以及窗口结束、客户端停止并排空服务后读取。原始 native 不读取预算端点。

- 两组 `graph-check.json` 核验真实 replay、四 rank 一致、实际1..32 batch 直方图、B32 覆盖、无窗口内 eager decode 回退或 recapture。
- `cost/budget-check.json` 核验四 rank ×43层 ×gate/up和down，共344项 GEMM 调用差分均等于全部实际 replay 次数，包含空本地工作。每层 planner 调用差分也必须等于 replay 次数，而不是两倍。九候选对直方图之和等于 planner 次数；非法 plan 计数必须为零。
- 每层保留最后一份32项 plan、预测耗时及候选对直方图。`plans_with_pair_variation=0` 是可接受且应原样报告的观察，不能声称本次服务发生动态切换。多个候选对也可能受实际 batch/shape 影响，单靠计数不能隔离路由因果。
- `comparison.json` 报告两组完整1200秒的 decode tokens/s、`cost_vs_native_whole_window_ratio`，并保留请求、补充队列耗尽与未完成情况。`sustained_c32_ratio_available=false` 时持续32并发比值为空，不能把尾部低并发当成持续 C32 性能。

数值和 token 差异始终为 `record_only`；没有 relative-L2、absolute-error、allclose 或替代阈值。预测误差和 regret 也是报告项，不决定是否接受吞吐。保留非有限输出、非法内存、编译/运行、malformed tensor/routing、来源、覆盖和清理检查；CPU 与 GPU planner 确定性选择一致性仍属算法有效性。完整 MoE 有限数值差异记录在 held-out 证据中；自然 GPQA 吞吐客户端不计算答案准确率或 token 一致率，明确 `generation_agreement_evaluated=false`。

## 回传与失败

控制器成功或失败都输出 `work_directory` 与 `export_directory`。公开目录只包含静态白名单的 aggregate JSON：实验/来源/身份、四份校准表和 held-out 误差、GPU聚合报告、图与预算计数、吞吐结果、launch/runtime/lifecycle 及逐文件 SHA256 manifest。原始题目、token IDs、生成文本、路由数组、激活、权重、完整日志及 JIT overlay 保留在服务器私有目录。

任何有效性失败保留阶段、错误类型、现有聚合证据与清理状态；具体错误文字写私有日志，避免把数据误带入公开报告。只回传新公开目录，不使用 `git add .`，不改写旧结果。校准 ready、CPU 测试、GPU 算子检验和43层真实服务吞吐是不同完成层级，只有本次服务结果能支持吞吐结论。
