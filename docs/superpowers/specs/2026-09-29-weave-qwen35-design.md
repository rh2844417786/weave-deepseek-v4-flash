# Qwen3.5-35B-A3B Weave 复现设计

> 历史设计：目标模型已切换为原版 DeepSeek-V4-Flash 混合精度权重。新决策与待验证项见 `docs/DEEPSEEK_V4_FLASH_MIGRATION.md`；本文的 Qwen BF16 内核和维度合同不适用于新目标。

状态：需求和方法设计已形成；GPQA 已下载并完成 CPU tokenizer 长度审计。CUDA 实现、SGLang 后端、Docker 镜像及 GPU 实验均尚未执行。本文供 GPT-6 Sol 后续编码使用。

## 1. 用户已确定的约束

1. 目标模型固定为 Qwen3.5-35B-A3B，BF16，服务器权重目录 `/mnt/public_data/Qwen/Qwen3.5-35B-A3B`。
2. 目标是完整模型真实自回归生成，输出 decoding 速度；没有必须获得加速的阈值。
3. 无可复用的 Weave 代码，依据论文自行实现；最终全部在 Docker 中运行。
4. 编码端无法连接 GPU 服务器；通过 GitHub 同步，服务器 git pull 后执行，再回传运行证据。
5. 数据为 GPQA，batch size 1；只输入文本问题与四个选项，保留模型的完整文本生成架构与共享专家语义。
6. 开启 thinking，max_new_tokens=32768（32K tokens），包含 reasoning 与最终答案的全部生成 token；允许正常 EOS，不为了长度而忽略 EOS。启动前核对有效上下文长度足以容纳输入与此输出上限，不静默下调上限。
7. 仅比较 SGLang 与自行复现的 Weave；不增加其他框架基线、模型规模、量化方案或性能实验矩阵。
8. 第一阶段使用同一组 4 张 H100，EP=4；TP/attention 布局根据现有 SGLang 配置核实并在两组保持一致。
9. 每组以服务就绪为起点，预热与测量合计最多 1200 秒。镜像构建、编译、权重加载另行记录耗时。

服务器环境为用户提供的信息，尚未由编码端验证：单机 8 张 H100 80GB SXM5/NVSwitch，Ubuntu 24.04.3，driver 580.173.02，CUDA Toolkit 13.0.88，Docker 可用，宿主系统盘紧张。原环境存在损坏的 conda env，不应默认使用。新构建缓存与结果必须落到显式挂载的足额工作盘。不要覆盖现有 baseline 配置或清理用户其他环境。

## 2. 对比定义与方法边界

选择共享 SGLang serving/model runtime 的架构，两个模式分别是：

- `sglang_native`：锁定版本的原生 MoE 路径。
- `weave_full`：同版本 SGLang，完整 routed-MoE 路径替换为自行实现的 Weave persistent megakernel，prefill 与 decode 都实际执行这个后端。

复用 SGLang 的权重加载、attention、Gated DeltaNet/线性 attention 状态、共享专家、采样、请求生命周期与 KV/cache 管理。自行实现 Weave 的通信、专家计算调度、成本模型、chunk pipeline、bubble stealing 及对应同步。使用 CUDA/ThunderKittens 等底层原语不代表存在官方 Weave 源码；必须记录复用的原语和自行实现的模块。

不重写整个服务框架，以控制实现范围并提高比较的可解释性。不能仅改一个 SGLang 配置或 SM 数便将结果标作完整 Weave。不能仅在 prefill 调用 Weave 而将默认 decode 的速度标作 Weave decode。

论文在 4×H100、BF16、EP=4 上以 prefill 为主进行评估，论文 §5.3 的 E2E 是 attention/pre-MoE 时间与 MoE kernel 时间组合。本项目的 GPQA 自回归生成是方法在 decode 场景的扩展评估，不是论文原数字的直接重现。batch=1 decode 每步通常只有一个全局新 token；即使上下文很长，这也不会变成有数千输入 token 的 MoE prefill。若空间/时间重叠收益不足甚至减速，应保留真实结果。

## 3. 模型正确性合同

启动时从实际权重读取并校验：`text_config.hidden_size=2048`、`moe_intermediate_size=512`、`num_experts=256`、`num_experts_per_tok=8`、`num_hidden_layers=40`、`shared_expert_intermediate_size=512`。原始模型是多模态架构，GPQA 使用纯文本输入；不能粗暴替换成不匹配的旧 Qwen 配置。

沿用原生 router 与 top-k 结果、权重归一化、SwiGLU、shared-expert sigmoid gate、残差与层顺序。共享专家必须恰好计算与相加一次；正确处理现有版本把共享专家融合为额外专家槽的情况。不能无意把 routed top-8 改成 top-9，也不能漏掉共享分支。

若两组为控制变量统一关闭 shared-expert fusion，必须在两组同时关闭并记录为对原服务器配置的调整，不能单独削弱 baseline。更优先保持原生共享专家路径，并只替换 routed experts 的实现。

EP 专家映射固定连续分块：EP=4 时每 rank 64 个 routed experts。若 SGLang 返回 TP 复制的 hidden states，需要明确 unique token owner，避免同一个 token 被四个 rank 当作四个不同请求重复派发。消费端需要 TP 复制输出时，只在 adapter 边界执行一次明确的恢复/集合通信。不得把 partial contribution 再次求和或重复 all-reduce。

## 4. Weave 组件与通信合同

### 4.1 原子实现单元

完整 MoE routed path 为 dispatch → gate/up GEMM → SiLU-mul → down GEMM → combine。目标是在一个 persistent megakernel 的 execution phase 中完成这些阶段；显存分配、IPC 建立和权重布局转换在初始化阶段完成。

使用 Hopper BF16 Tensor Core 原语，FP32 累加。首版专用于 2048/512、256 experts、top-8、EP=4，tile 形状通过编译合法性与实测选择，不扩展到其他模型或量化格式。所有 padding tile 必须受 valid-token mask 约束。

每个 logical worker 对应一个持久 CTA。启动前查询 SM 数和 occupancy，验证资源约束下每个 SM 可驻留目标 CTA 数；不能仅凭 `grid=132` 就声称进行了物理 SM 绑定。基于 CTA worker 数解释 c，并验证实际驻留行为。

### 4.2 通信与依赖

首选单机 peer-access + CUDA IPC/ParallelKittens PGL 原语，由初始化导出/导入缓冲区句柄，使用稳定地址。所有 rank 参与每次调用，包括本 rank 输入 token 数为零的情况。NCCL 可用于进程初始化、控制信息和 adapter 必需的复制恢复，核心 dispatch/combine 不得悄悄回退到顺序 NCCL kernel。

每个 token 带 `(origin_rank, origin_token_index)`；dispatch 对同一 token 到同一目标 rank 去重，但同一目标 rank 上的多个 expert 仍保留独立 token-expert 计算任务。回传按 `(origin_token, topk_slot)` 写入唯一槽，乘路由权重后固定顺序聚合。禁止跨 GPU 对 BF16 输出做无保护竞争写。

同步使用文档明确的 system-scope release/acquire、数据写入后发布 ready、消费者 acquire 后读取；为每轮调用设置 epoch，避免读到前一轮标志。GEMM0 完整输出到齐才执行对应 SiLU；GEMM1 的完整 reduction 到齐才发布 combine。debug 模式记录依赖事件、tile 执行次数与异常；正式测速关闭细粒度 trace。

路由元信息与本 rank inbound/outbound counts 必须在 GPU 上可用。其生成和必要通信开销全部包含在完整请求时间中；不能在计时区外生成当前请求的路由统计。设备端 prologue 根据已发布的路由元数据选择计划，不能在每层热路径使用 `.item()` 或 CPU 决策。

### 4.3 空间与时间调度

空间调度：按论文 §3.1，依据 local/remote token-expert pairs、去重后的 remote tokens、BF16 字节数、实测通信带宽和 GEMM 吞吐曲线选择通信 worker 数 c。

时间调度：按 §3.2，dispatch 不分成 K 个 pipeline chunk；专家计算和 combine 按 token chunks 形成依赖链。dispatch 后的集中空闲窗口允许 comm workers 从原子队列领取有限的 ready GEMM tiles；不能重复执行或领取未就绪 tile；最后 combine 所有 worker 可参与。

成本模型：按 §3.3 联合选择 c、K，包含 chunk GEMM 效率修正与 combine tail。steal tile 数按 §3.2.2 计算并裁剪为真正可执行的整数 tile 数。保留论文公式的 CPU 参考实现与 GPU 实现一致性测试。

起始搜索域由本机 SM 数生成：c=4,8,...,N-4；无远程通信时额外允许 c=0；K 属于 1,2,4,8，过滤空 chunk 和无效配置。具体域属于复现工程选择，不能声称就是作者未公开的搜索代码。对 global token=1 的 decode，允许 K=1、steal=0；仍执行本项目的正确后端，不添加伪 token 来制造并行收益。

校准表针对 shape、dtype、硬件、SM/worker 配置和有效 token bucket 建立；runtime routing 改变时重新决策而非使用按 sequence length 固定的 c。未完成校准、过期 profile 或不支持的 shape 明确报错，不用编造曲线填补。校准也有每次 1200 秒上限并落盘完成状态，可分组恢复。

### 4.4 CUDA Graph

baseline 与 weave 必须采用相同的 graph 设置。先通过 eager 正确性门槛，再完成 stable buffer、动态 token/routing 输入、epoch reset、跨 rank 协调与 graph replay 测试。一次捕获不能冻结后续每步的路由或 c/K 选择。

若 graph 尚未可用，两组 eager 的结果只能标记为 eager 比较，不能替代用户现有启用 graph 的 baseline 性能。完整验收包含与最终 baseline 配置一致的路径。

## 5. GPQA 与生成设置

使用 GPQA Main 448 题，数据源/散列见 `data/gpqa/source_manifest.json`。Diamond/Extended 已下载用于数据完整性和长度审计，不自动增加正式实验。

正式 prompt 使用 `scripts/audit_gpqa_lengths.py` 的 `build_message` 定义，官方模板开启 thinking，seed=42 固定题序，四选项按原始行号确定性打乱。正确答案标签、解释、验证反馈不进入输入。两组共用完全相同的 prepared manifest 与 token IDs。

实际 Main 输入长度：min 74、median 213、P95 442.95、max 2780，8K 样本 0。Diamond 同样无 8K；Extended 仅一个 12323-token 样本。首轮保留自然长度，不拼接或 padding。输出上限 32768 与输入 prefill 长度分开记录。

生成默认使用 greedy（temperature=0）以方便对齐，开启 thinking，允许正常 EOS；单请求/单序列，服务 `max_running_requests=1`。同一组内不插入其他请求。MTP/推测解码默认关闭，prefix cache 复用默认关闭；待现有 baseline 配置确认后两组使用同样策略。若用户要求保留推测解码，则必须另行修改 token 计时与后端覆盖合同，不私自扩展本计划。

## 6. 计时与 20 分钟预算

每个 backend 为一组：服务 ready 后开始 1200 秒 wall-clock deadline；前最多 120 秒进行至多两条 warmup，随后正式请求使用剩余预算。两组顺序执行在同一组 GPU 上，不并行互相争抢资源。两组比较合计最多 40 分钟的 ready 后运行时间，构建/加载另计。

每个请求设置可取消的 deadline，超时请求标记 timeout，不当作完整样本；剩余时间不足时停止发新请求。外层进程 watchdog 在 1200 秒截止，先原子保存已完成记录，再取消请求，终止本次创建的测试服务；不能误杀原有 baseline 服务或其他任务。

主指标是 server-side token-ready 时刻定义的 decode tokens/s。计数来自真实生成 token IDs，reasoning 与 answer 都计入，EOS/截断策略一致；不能使用去掉 reasoning 后的可见文本长度。令 n 为实际输出 token 数，first/last 是第一个/最后一个输出 token 就绪时间，decode 秒数为 last-first，decode token 数为 n-1，n<2 时该指标为空。

聚合用 `sum(n-1)/sum(decode_seconds)`，同时给出 TPOT、TTFT、输出 token 数、完成/超时数、EOS/length finish reason、输入长度、后端调用数。不得用 n/整个请求耗时冒充 decode tokens/s，也不能用 SSE chunk 数或字符数冒充 token 数。

优先使用锁定版本已有 scheduler token-ready 时间统计；检查其含义与 streaming interval。若现有字段只标首个输出批次，则在两种模式同一位置增加首次/末次真实 token 提交时间，不新增逐 token `cuda.synchronize()`。无法得到符合定义的时间时保留 client-observed latency 并明确标签，不能输出伪 server decode 指标。

固定题序，deadline 下两组可能完成不同数量的题。主比较只使用双方都完成的同一批题 ID，并报告 unmatched/timeout 数；不能因为更慢的一组完成较少就把两组不同问题的均值直接作为加速比。若两组前缀和输出轨迹明显不同，同时报告长度分布与逐题结果。

## 7. 依赖与未获取的外部输入

已有 SGLang 配置是唯一未提供的关键工程输入：需要镜像 digest、版本/commit、启动命令、GPU IDs、TP/EP/DP、MoE/attention backend、CUDA Graph、shared-expert fusion、MTP/推测解码等实际值。第一项编码任务提供配置采集器，用户在服务器执行并回传 JSON。若镜像已有不明修改，必须导出源码差异或换成同一版本的可审计原生构建，禁止凭镜像名判断功能。

已只读检查 SGLang v0.5.19 对应 commit `0bcd822377da7b5718e674eaf9c870d349424dd1` 的 Qwen/MoE/时间统计源码，作为 adapter 设计参考；这是候选参考，未确认是服务器实际版本。实际 baseline 版本锁定后重新验证插入点，两个 mode 共享完全相同的 runtime commit。

Docker BASE_IMAGE 从实际 baseline 锁定，不使用浮动 latest；安装 CUDA 扩展所需 toolkit 由该镜像及上游原语支持条件确定，宿主驱动不需要随意修改。记录 nvcc 与 PyTorch CUDA ABI，并把版本约束写入锁文件。允许选择隔离 toolkit 是规划假设；若用户要求必须 CUDA 13，则以其约束重新验证。

GitHub 目标仓库 URL 尚未提供，当前本地没有 origin，也没有提交。源码与文档可继续完成，push 前需要具体目标 URL。服务器工作根目录通过必填 `WORK_ROOT` 指定，脚本拒绝空值和不可写目录。模型只读挂载为 `/models/Qwen3.5-35B-A3B`，项目与数据挂载到 `/workspace/Weave`，缓存/结果放到 `WORK_ROOT` 的子目录。

## 8. 交付与完成定义

交付包括完整源代码、带版本锁的 Docker 构建、GPQA 下载与准备、原生/Weave 启动脚本、限时测速、原始计时 JSONL、汇总 CSV/JSON、正确性验证、后端实际覆盖证据，以及一条明确的服务器运行流程。

CPU 检查、CUDA 编译、单层 GPU 正确性、完整模型生成、正式性能结果分别记录状态。编码端无法运行 H100，不能把本地测试通过或 GitHub push 成功写作 GPU 复现成功。远程失败时先读用户回传的完整错误和版本信息，再修复并生成下一个可拉取提交。

## 9. 原始依据

- Weave v3 §3–5：https://arxiv.org/html/2609.21483v3
- Qwen 官方配置与模板：https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754
- SGLang EP 文档：https://docs.sglang.io/docs/advanced_features/expert_parallelism
- SGLang 候选 MoE 插入点：https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/models/qwen2_moe.py
- SGLang Qwen3.5：https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/models/qwen3_5.py
- GPQA 官方下载说明：https://github.com/idavidrein/gpqa/tree/56686c06f5e19865c153de0fdb11be3890014df7
