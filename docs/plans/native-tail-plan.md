## 一、结论与推荐主线

**可行，但需要新增真正的设备端任务执行器，而不是继续扩展现有 CTA 预算控制。推荐选择 A：保留复制输入和本地 EP 专家计算，将 decode 的“down projection＋局部合并＋输出归约”重构为统一的通信—计算 worker 执行器。**

具体边界如下：

> **保留原生 router、grouping、Marlin gate/up、activation、shared experts 和 prefill；从原生 Marlin 提取可由 CTA worker 直接调用的 down-tile 计算单元；新增设备端分块归约通信；在同一个 resident kernel 内实现任务领取、就绪发布、通信优先级和通信 worker 窃取计算任务。最后才加入路由感知成本模型。**

这条路线能够实现你要求的四项机制，但其范围是 **MoE 后半段的 Weave 机制适配**，不是论文五个阶段全部融合的完整 Weave。

必须放宽两项实现约束：

1. **保留 Marlin 的算术与权重布局，但不能要求 down projection 继续完全使用原封不动的独立 `__global__ Marlin` kernel。**
2. **保留输出归约的模型语义，但优化路径中的这一次 decode MoE all-reduce，必须允许由新的设备端通信协议替代。** Attention、prefill 和其他位置的原生 collective 不变。

如果同时坚持“Marlin 是不透明的独立 kernel”和“通信只能调用现有不透明 NCCL collective”，最多能做 kernel 级流水和资源让步，无法实现“同一个通信 worker 领取并执行 Marlin tile”。

---

# 二、源码依据与当前状态

## 2.1 本次分析固定的版本

本次读取时，仓库最新 `main` 为：

```text
仓库：
rh2844417786/weave-deepseek-v4-flash

分析基准：
33d071998fc5bcfbe82cf0bc526a1a3abae06b61

固定 SGLang：
0.5.17
29481685462732237d80d86076d6563e1f658102
```

最新 main 包含 2026 年 10 月 21 日的 measured-cost 服务结果；对应运行使用的父提交为 `dccd705403bc9cc049d8fd95bc9a8ca431bf2c42`。上游文件的固定 SHA256 在 `configs/native_budget_sources.json` 中。 

已读取你列出的预算文档与源码，以及 `native_cost.py`、`native_cost_patch.py`、`native_cost_runtime.py`、`native_cost_calibration.py`、`csrc/native_cost.h` 的关键实现，核对了最新正式结果和固定上游 Marlin 的计算、条带划分、归约、写回部分。

Weave HTML 入口未能直接读取，改读了页眉明确标记为 **2609.21483v3** 的 PDF，并检查 §3.1–§4 及对应图示。没有访问服务器私有路由、激活或日志，也没有在本次分析中执行 GPU 实验。[arXiv](https://arxiv.org/pdf/2609.21483)

## 2.2 用户给出的当前路径描述，基本得到源码确认

| 描述 | 核对结果 |
|---|---|
| 原始混合精度 checkpoint、routed MXFP4 Marlin | 当前部署身份与固定源码支持 |
| routed/shared 双流 | 当前文档引用的真实图元数据记录了 `forward_normal_dual_stream`；原生函数确有双流 fork/join |
| `moe-a2a-backend=none` | 最新 native 启动记录确认 |
| 复制输入、本地专家计算、后续归约 | 当前 native 成本模型文档明确按此路径定义 |
| 动态预算只调整两个 GEMM 的有效 CTA | `native_budget_patch.py`、`native_cost_patch.py` 明确如此 |
| 物理 captured grid 固定 | kernel 内使用逻辑 worker 数重新划分原有工作 |
| 没有该 native 路径的通信 worker、chunk、stealing | 当前 runtime 身份和成本模型文档明确承认未实现 |

现有补丁把 Marlin 中三个影响工作划分的 `gridDim.x` 使用位置替换为逻辑 worker 数，停用 CTA 在访问锁和异步 shared-memory 操作前退出。**这是 GEMM 工作参与数量控制，不是通信—计算角色调度。** 

最新启动记录还确认：原生组关闭 custom all-reduce，使用精确 batch 1–32 的 full graph、关闭 graph padding，prefill 走原生。无需重新做一轮泛化 P0 才能承认这些事实。

另需区分：仓库的历史 `experimental/hopper_persistent/` 中存在另一套通信/计算机制实现，但它使用自己的 Hopper tile 和 origin/dispatch/combine 路径，**不能视为当前 native Marlin 已经获得这些能力**。新路线可以参考其协议组织，不能直接继承它的计算性能、成本表或服务证明。

## 2.3 已有服务证据应直接复用

最新完整窗口结果为：

| 组别 | 有效 decode 吞吐 |
|---|---:|
| 原始 native | 1893.149 token/s |
| measured CTA cost | 1809.243 token/s |
| 比值 | 0.955679 |

两组均为 full graph、自然 GPQA、初始并发 32、每组 1200 秒；没有数据或补充队列耗尽。该轮成本模型始终选择预算对 `(8,8)`，未观察到候选对切换。不能把约 4.4% 的下降确定归因于某一个开销，也不能把这轮结果当作通信调度机制的验证。 

**因此，下一步缺的是可被调度的执行机制，而不是再给现有预算表增加特征。**

---

# 三、架构选择：选 A，不建立另一套 dispatch 路径

## 3.1 A 与 B 的取舍

| 项目 | A：复制输入＋本地计算＋可调度归约 | B：dispatch/compute/combine |
|---|---|---|
| 输入组织 | 保留当前副本 | 重新定义 origin、dispatch 和接收布局 |
| 原生计算复用 | gate/up、activation、shared 可以完整保留 | 需要更广泛改造输入行映射与两次 GEMM |
| 新通信执行器 | 只处理 MoE 输出归约 | 输入 dispatch、结果 combine 都要处理 |
| 与论文的距离 | 角色、就绪流水、窃取适配；通信拓扑不同 | 更接近论文五阶段结构 |
| 原生吞吐风险 | 集中在 down 重构、归约实现和新增协调成本 | 还可能重新引入原生不需要的输入搬运 |
| 推荐 | **主线** | 本轮不实施 |

选择 A 的原因不是它更容易被称为 Weave，而是它与项目目标一致：**不为复现论文结构而给原生路径增加本来没有的工作。**

第一版只把 down 和输出通信纳入统一调度。gate/up 尚不参与 worker stealing，因此可优化空间也应按这一范围解释，不能把整个 MoE 的计算量都算成可重叠计算。

## 3.2 当前与目标结构图

### 当前：独立 GEMM 的有效 CTA 预算

```text
复制的 hidden、原生路由
          │
          ├──────────────→ 原生 shared FP8 分支 ───────────────┐
          │                                                  │
          ▼                                                  │
      原生 grouping                                          │
          │                                                  │
     GPU budget planner                                      │
          │                                                  │
          ▼                                                  │
  独立 Marlin gate/up kernel                                 │
          │                                                  │
      原生 activation                                        │
          │                                                  │
  独立 Marlin down kernel                                    │
          │                                                  │
    原生 routed sum ──────────────────────── join ─────────────┘
                                                 │
                                          shared 加法
                                                 │
                                         原生 all-reduce
                                                 │
                                             完整输出

预算只改变 GEMM 的逻辑 worker 数。
不存在通信任务队列，也不存在通信 CTA 执行 Marlin tile。
```

### 目标：原生前半段＋统一 tail worker 执行器

```text
复制的 hidden、原生路由
          │
          ├──────────────→ 原生 shared FP8 分支 ───────────────┐
          ▼                                                  │
      原生 grouping                                          │
          ▼                                                  │
  原生 Marlin gate/up                                        │
          ▼                                                  │
      原生 activation                                        │
          │                                                  │
          └─────────────── join：保证 shared 已就绪 ────────────┘
                                  │
                         prepare / 静态或 GPU 计划
                                  │
                                  ▼
       ┌──────── 同一个 resident tail kernel ────────────────┐
       │                                                    │
       │ compute workers                                    │
       │   领取 Marlin-derived down tile                    │
       │     → 分块完成计数                                 │
       │     → 本地 slot sum＋shared partial                 │
       │     → 发布 partial-ready                           │
       │                          │                         │
       │                          ▼                         │
       │ comm workers                                       │
       │   拉取已就绪 peer partial                          │
       │     → owner 分片归约                               │
       │     → 发布 reduced-ready                           │
       │     → 拉取完整结果分片                              │
       │                                                    │
       │ 通信未就绪时：允许领取同一个 down 计算任务队列       │
       └────────────────────────────────────────────────────┘
                                  │
                         完整、已归约的 MoE 输出
                                  │
                    不再执行本层原来的重复 all-reduce
```

**第一版明确将 shared 的 join 前移到 tail 入口。** 这是有代价的架构选择：可能失去原生 shared 与部分 down 的重叠，但避免 persistent kernel 占据资源后等待一个尚未完成、甚至无法获得资源的外部 shared kernel。该损失必须单独计量，不能隐藏。

---

# 四、Marlin 怎样进入统一任务调度

## 4.1 现有 Marlin 不是现成的 device-side tile 库

固定源码中的 Marlin 是 `__global__` kernel。其工作条带、跨 CTA 部分和以及 `locks_off` 与 grid 几何共同确定；不能把任意一段循环直接作为可窃取任务，继续沿用原锁编号。  

能够复用、但需要提取的部分已经存在：

| 可复用内容 | 上游位置 |
|---|---|
| packed 权重、E8M0 scale 和索引规则 | `marlin_template.h`、量化准备代码 |
| `cp.async` 多阶段加载 | `fetch_to_shared`、`wait_for_stage` |
| shared→register 双缓冲 | `fetch_to_registers`、fragment 数组 |
| FP4/scale 解码与 Tensor Core 运算 | `dequant_data`、`matmul`、`mma/mma_trans` |
| CTA 内归约 | `thread_block_reduce` |
| BF16 写回、routing weight 处理 | `write_result` |

这些并不是当前独立导出的 device API，而是嵌在全局 kernel 内的局部状态与 lambda，需要重构成显式上下文。  

## 4.2 第一版采用“完整 K 的 down tile”

建议任务定义为：

```text
DownTask {
    invocation_id
    local_group_id
    output_tile_id
    output_channel_begin
    chunk_id
}
```

一个任务处理：

```text
一个原生 expert group
× 一个原生 N tile
× 完整 down reduction K=2048
```

具体决定：

**M 分组保留原生选择。** 不把不同专家的行混成一个 GEMM，也不重新退回旧自研 M16 组织。

**N tile 和 K 内循环优先采用已有 Marlin specialization。** 第一版不另写逐元素 WMMA 计算，不引入新的 FP8 激活量化。

**任务独占自己的输出区域。** 完整 K 在一个 CTA 内完成，保留 CTA 内归约，暂时取消跨 CTA split-K。这使任务可以被任意 worker 领取，而不需要等待原 grid 中某个固定归约伙伴。

**权重仍使用原生重排布局。** `output_channel_begin` 必须进入原生 packed 权重和 scale 的寻址公式，不能对重排后的权重随意做一个普通 Tensor 切片。

**任务结束必须清空异步流水。** `cp.async` 的未完成操作、shared scratch 和 fragment 状态不能泄漏到下一项任务。

拟议接口是：

```cpp
// 全 CTA 一致调用；不是启动一个子 kernel。
__device__ void run_marlin_down_tile(
    const DownArgs&,
    const DownTask&,
    WorkerScratch&);
```

这一步会改变原生跨 CTA 归约组织，不能保证逐位相同；有限差异只记录。更重要的是：**复用了 Marlin 算术，不等于自动保留了原生吞吐。** 必须单独测量提取后静态映射版本，定位寄存器、流水启动、完整 K 和任务边界带来的成本。

## 4.3 原生前半段需要一个明确接口

从 `fused_marlin_moe()` 拆出 decode 专用前半段，返回长期存活的：

```text
DownInputs {
    activated_pairs       # 原生 activation 的输出
    sorted_token_ids
    local_expert_ids
    num_tokens_post_padded
    topk_weights
    original_slot_mapping
    w2_weight / w2_scales
    native_geometry
}
```

不能先调用完整 `fused_marlin_moe()`，再额外跑新 tail；那会重复执行 down。

原生的 slot 索引、被屏蔽专家和 padding 必须保留。新 local finalize 只能读取有效本地 slot；未执行的槽位不能因为未初始化而污染求和。

路由缩放沿用原生实际位置。固定 Marlin 路径的 GEMM2 已处理 top-k 权重，MXFP4 finalize 随后执行求和；新 tail 不得再无条件乘一次 routing scale。

---

# 五、通信执行器与提前交付：采用输出通道分块

## 5.1 为什么不沿 token 维切 B32

本方案选择沿 **down 输出通道** 分块，而不切专家的输入行：

\[
H=4096,\qquad C\in\{1,2,4\}.
\]

第一版先实现 \(C=1\)，随后 \(C=2\)，最后再开放 \(C=4\)。

这样，同一个专家 group 的 M 不因流水分块进一步碎片化；gate/up 和 activation 也不重复计算。每个 down 输出 tile 仍只执行一次。

这与论文的 token chunk 不同，是针对当前原生小 M 路径的适配，而不是原样照搬。论文将 GEMM、activation、combine 按 token chunk 组织，dispatch 不分 chunk。[arXiv](https://arxiv.org/pdf/2609.21483)

分块仍有成本：输入 activation 的缓存复用可能变差，调度和通知次数增加，消息变小，数据布局整理也可能增加。不得把“不增加 M padding”解释成“分块没有损失”。

## 5.2 什么条件下可以发布一个 chunk

定义 rank \(r\)、输出 chunk \(k\) 的本地贡献：

\[
P_{r,k}
=
\operatorname{NativeLocalFinalize}
\left(
\{\text{该 chunk 的本地专家加权输出}\},
\text{shared partial}
\right).
\]

只有以下条件全部满足才允许发布：

1. 该 chunk 的**全部本地 down 任务**完成；
2. 原 top-k slot 的本地求和完成；
3. shared partial 已按当前原生语义加入；
4. 数据写入对消费者可见。

不能“某一个专家完成就宣布整个 token 的输出完成”。

当前文档记录 `shared_tp1=false`，因此 shared 是归约前的分片贡献。本方案第一版明确支持该已验证契约；遇到其他 shared 模式必须显式报出不支持，不能悄悄套用错误加法位置。未来支持 TP1 shared 时，应在全局归约后加入。 

## 5.3 最小通信方案：设备端 owner 分片归约＋结果收集

不新增外部通信依赖。使用已有 CUDA IPC 能力建立 peer workspace，由 worker 直接执行 peer 读写任务。

每个输出 chunk 再划成四个固定 owner 分片，分别归 rank 0–3：

```text
本地 partial-ready
        │
        ▼
owner 拉取其他 rank 的对应 partial
        │
        ▼
owner 按固定 rank 顺序求和
        │
        ▼
owner 发布 reduced-ready
        │
        ▼
各 rank 拉取四个 owner 的结果分片，恢复复制输出
```

对应设备任务：

```text
LOCAL_FINALIZE(chunk, owner)
FETCH_PARTIAL(chunk, owner, source_rank)
OWNER_REDUCE(chunk, owner)
FETCH_RESULT(chunk, owner)
```

推荐第一版采用 **pull 协议**：每个 rank 只写自己的 workspace，其他 rank 在 ready 后读取。这样不需要依赖多个远端 writer 同时更新同一个输出或远端累计计数。

通信缓冲区显式使用连续布局，例如：

```text
[chunk, owner, token, channels_per_owner]
```

最终由 `FETCH_RESULT` 写回 `[B,H]`。不能假设原 `[B,H]` 的任意通道切片是连续通信缓冲区。

对于四卡、等分 shard，若完整输出大小为 \(S=2BH\) 字节，该设计每 rank 的理想远端读取载荷为：

\[
2\frac{4-1}{4}S=1.5S.
\]

这只是所选协议的载荷核算，不包含控制通知、缓存事务放大和整理操作，也不表示比 NCCL 更快。

**与论文不同，本路线的通信载荷主要由 batch、hidden 和 dtype 决定，不按远端 token-expert pairs 增长。路由主要影响结果何时就绪以及各 rank 的计算偏斜。**

## 5.4 保留 NCCL 能做到什么

保留现有调用方式，可以控制 collective 的调用时机、stream 依赖及应用层分块；不能通过这个调用接口，让自定义通信 CTA 领取 NCCL 内部工作，再在空闲时执行 Marlin tile。

因此，本方案只替换 **decode MoE tail 的这一次归约**。新 tail 返回完整复制输出后，当前层原来的外层 all-reduce 和 shared 加法不能再次执行。

这项替换应在明确的 decode MoE helper 中完成，不要通过全局 monkey patch 把整个模型的 `tensor_model_parallel_all_reduce()` 变成 no-op。

---

# 六、统一 worker、任务协调与真正的 stealing

## 6.1 驻留与角色

每个 rank 启动一个 persistent tail grid，所有 CTA 编译相同的计算与通信能力。运行计划给出：

```text
W_r：该 captured kernel 的 resident worker 总数
c_r：其中承担通信优先角色的 worker 数
W_r - c_r：计算优先 worker 数
```

这里的 `c_r` 已经是**实际执行通信任务的 worker 数**，不再是少启动几个 GEMM CTA 的别名。

但它仍不自动等于独占的物理 SM 数。第一版允许多个 worker 共驻一个 SM，不强行沿用“一卡 132 CTA、每 SM 一 CTA”的组织，以免再次牺牲 Marlin 的并行度。实际角色分布用 `block_id、role、smid、时间区间` 证明，而不是用配置值推断。

建议使用 cooperative launch，并按**最终编译 kernel 的实际资源占用**限制 grid：

\[
W_r
\le
\operatorname{maxActiveBlocksPerSM}\times\operatorname{SMCount}.
\]

CUDA 对 cooperative launch 明确规定了这个上限。第一批 GPU 验证应定点检查该 launch 在现有 CUDA13/PyTorch/SGLang full graph 中的 capture/replay；未通过时修复入口或报告能力阻塞，不允许退化成未经证明可前进的 oversubscribed 自旋 kernel。[NVIDIA Docs](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__EXECUTION.html)

## 6.2 队列和状态机

第一版不必实现复杂的通用 GPU 调度框架。采用固定容量描述符、原子领取与依赖状态即可：

```text
WAITING → READY → CLAIMED → DONE
```

其中：

- down 任务在 tail 入口已具备 activation，按 chunk 优先级入队；
- `LOCAL_FINALIZE(k, q)` 等待该 chunk 的本地 down 完成计数；
- `FETCH_PARTIAL` 只在对应 peer ready 后可领取；
- `OWNER_REDUCE` 等待四份 partial；
- `FETCH_RESULT` 等待 owner reduced-ready。

**禁止先领取一个未就绪任务，再占住 worker 无限等待。** 依赖未满足时，worker 应尝试其他 ready 工作。

唯一任务 ID 应包含：

```text
invocation、stage、group/tile 或 chunk/owner/source
```

领取使用 CAS 或单调 cursor；详细验证模式使用访问位图检查任务集合，而不只比较“总完成数相同”。总数相同仍可能隐藏一项重复、一项遗漏。

## 6.3 第一版 stealing：集中、有限、通信优先

建议先采用一个简单且明确的策略：

> **在第一次通信任务可执行之前，空闲通信 worker 可以从同一个 down 队列领取最多 \(s_r\) 个 tile；每完成一个 tile 都检查通信 ready。一旦进入通信阶段，本次调用不再反复切换回初始 stealing 窗口。**

当 \(c_r\ge2\) 时，至少保留一个通信 worker 持续推进 ready 检查；其他通信 worker 才允许初始 stealing。第一版 \(c_r=1\) 时令 \(s_r=0\)。

这对应论文的“集中窃取窗口”思想，但因为没有 dispatch，本方案把窗口放在 **tail 启动至首次可通信结果出现之间**，明确属于适配。论文的原窗口位于 dispatch 完成后、combine 开始前。[arXiv](https://arxiv.org/pdf/2609.21483)

实际 stealing 的判据必须是：

```text
初始角色为 COMM 的 CTA
    → 成功领取 down task
    → 在同一 persistent kernel 内执行 run_marlin_down_tile()
    → 完成该 task
    → 恢复通信任务领取
```

host 先启动一个 GEMM、再启动通信 kernel，不属于该机制。

当全部计算和 local finalize 完成后，计算 worker 可以加入剩余通信任务，避免尾部只有少数 worker 工作。

## 6.4 内存可见性与跨 rank 进度

协议需要在代码中明确区分 device-scope 和 system-scope：

| 数据 | 发布与读取要求 |
|---|---|
| 同卡 down 输出、依赖计数 | 写入完成后发布 device-scope release；消费者 acquire |
| peer partial、reduced shard | 发布 system-scope ready；peer acquire 后读载荷 |
| task descriptor | descriptor 写完后才能发布 READY |
| workspace 复用 | 所有消费者确认完成后才允许覆盖 |

特别注意：**thread 0 的一次 fence 不能替其他线程完成写入排序。** 应把“所有写入线程完成写入/必要 fence、CTA 汇合、leader 发布状态”封装成统一 helper。CUDA 的 fence 约束针对执行它的线程，不能把 `volatile` 当作完整同步协议。[NVIDIA Docs](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)

每次 invocation 使用：

```text
session_id、layer_id、device_sequence、buffer_slot
```

ready/done/ack 带序号，不使用反复清零的裸布尔量。Graph replay 中由 GPU 推进序号，不能捕获一个永远不变的 host epoch。

各 rank 必须一致的协议字段包括：

```text
invocation、有效 B、chunk 数、dtype、分片布局、协议版本
```

本地 `c_r`、steal 配额可以不同；chunk 数和通信布局不能各选各的。

首版在 kernel 结束前完成所有结果消费确认，防止快 rank 复用缓冲区覆盖慢 rank 尚未读取的数据。空本地专家任务也必须参与协议，因为它仍可能持有 shared partial、承担 owner 归约或需要获取最终输出。

死亡 rank、超时和非法状态走明确 abort/失败清理，不宣称可容错继续推理。

---

# 七、成本模型：只搜索已经落地的参数

## 7.1 真正可执行的搜索空间

新增模型使用独立身份，不能继续把旧的九个 CTA 配额对叫作通信调度候选。

| 参数 | 实际作用 |
|---|---|
| `c_r` | 改变 resident CTA 的通信优先/计算优先角色 |
| `C` | 改变 down 任务顺序、完成依赖和通信分块布局 |
| `s_r` | 限制通信 worker 初始窗口实际领取的 down tile 数 |
| `mode=serial` | 所有计算完成后再通信，不使用提前交付或 stealing |

第一批候选可以限定为：

```text
c ∈ {1, 2, 4, 8}，且 c < W
C ∈ {1, 2, 4}，满足 N tile 和四 owner 分片对齐
s ∈ {0, 1, 2}，c=1 时 s=0
另含 serial：C=1、c=0、s=0
```

`W`、CTA threads、编译 specialization 和 shared-memory 大小是 capture 前确定的执行几何，**不伪装成同一张图内可以任意改变的参数**。

## 7.2 工作量与成本项

当前路由提供本地有效 pairs、专家行数分布、原生 groups、每 chunk 的 down task 数和倾斜程度。

本方案可调度的主要矩阵计算量是 down：

\[
F_{\rm down,useful}=2P_rHI,
\]

不能把 gate/up 也计入可窃取工作，再使用论文完整专家的 \(6P_rHI\) 预测收益。

校准至少包含：

| 曲线 | 必须包含的实际成本 |
|---|---|
| down worker 吞吐 | 真实 Marlin-derived tile、实际 groups、寄存器和任务边界 |
| 通信性能 | IPC payload、ready 检查、owner reduce、结果收集 |
| 并发干扰 | 权重读取、peer 流量、L2/HBM 竞争 |
| 分块损失 | chunk 数、额外任务、整理、消息数量和通知成本 |
| stealing 成本 | tile 时间、角色恢复延迟、首次通信被推迟的时间 |
| 固定成本 | prepare、统计、planner、协议头、结束确认 |

旧 measured-cost 表的目标是“两次独立 GEMM＋shared join＋串行 collective”，不适用于新 DAG。**可以复用身份绑定、原始样本保存、CPU/GPU planner 一致性和 held-out 工具，不能复用旧表中的耗时作为新执行器成本。** 

## 7.3 预测完整关键路径

用分块 DAG 预测，而不是直接套：

```text
max(compute, communication) + tail / C
```

对每个任务阶段：

\[
t_v^{\rm finish}
=
\max\left(
\max_{u\in pred(v)}t_u^{\rm finish},
t_{\rm resource\ available}
\right)
+
\widehat d_v.
\]

模型至少要体现：

```text
各 rank 的 chunk 就绪
→ peer partial 到达
→ owner reduction
→ 结果拉取
→ 最慢 rank 完成
```

完整 MoE 还包括保留的前半段、前移 shared join 的代价、统计和搜索成本。

GPU planner 不必模拟每一个 tile。可以按 rank/chunk 聚合计算工作，并使用实测 worker 曲线与有界阶段模拟；最终用真实完整 MoE 回放检验预测误差和选择 regret。误差与性能下降均只记录。

## 7.4 GPU 搜索与跨 rank 共识

推荐控制流程：

```text
各 rank 发布本次 workload descriptor
            ↓
逻辑 rank 0 的设备 planner 读取四份描述符
            ↓
枚举有界、执行器支持的计划
            ↓
发布一个共同计划：
{C, c[4], s[4], mode, epoch, protocol_version}
            ↓
四 rank 验证并采用该计划
```

第一版可对每个 `C` 为每 rank 保留两个局部候选，再枚举至多 \(2^4\times3=48\) 个组合，另保留 serial 和最佳固定计划。筛选近似造成的误差也要记录，不把它称为穷举最优。

所有描述符和计划地址预分配；GPU 搜索是 captured graph 中的真实执行工作，不使用 `.item()` 或测量窗口内 CPU 路由读回。Graph 的固定地址与执行结构约束仍然适用。[PyTorch 文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html)

以下结果都应允许出现：

- 始终选择 `C=1`；
- 始终不 stealing；
- 某些层选择 serial；
- 动态计划与最佳固定计划相同；
- 搜索成本超过收益。

这些不是数值有效性失败，也不能通过强制选择其他计划来制造“动态机制生效”。

---

# 八、按依赖排序的实施阶段

以下新增文件名是本次建议，**不是声称仓库已经存在这些实现**。

## 8.1 建议文件组织

```text
experimental/native_tail/
    marlin_down_tile.cuh
    task_protocol.cuh
    peer_reduce.cuh
    executor.cu
    cost_model.h                 # 后期加入

src/weave/
    native_tail.py
    native_tail_patch.py
    native_tail_runtime.py
    native_tail_calibration.py

configs/native_tail_sources.json
docs/NATIVE_TAIL_EXECUTOR.md
scripts/run_native_tail.py

tests/test_native_tail_*.py
tests/gpu/check_native_tail_compute.py
tests/gpu/check_native_tail_transport.py
tests/gpu/check_native_tail_executor.py
```

保留 `native_budget*`、`native_cost*` 和历史报告原样。新方法建议使用：

```text
native_marlin_peer_tail_v1
```

不要通过覆盖旧方法身份，把新执行器结果写入旧 CTA-budget 实验。

## 8.2 阶段总表

| 阶段 | 完成的具体能力 | 依赖 | 最小运行形式 |
|---|---|---|---|
| S1 | Marlin down 可调用任务单元 | 固定上游来源 | 单卡真实权重计算回放 |
| S2 | 设备端分片归约与 epoch 协议 | IPC/驻留定点检查 | 四卡预生成 partial |
| S3 | 静态通信/计算 worker 统一执行 | S1＋S2 | `C=1、steal=0` |
| S4 | 提前交付流水＋原生 decode 接入 | S3 | `C=2`、完整模型 graph |
| S5 | 通信 worker 实际 stealing | S4 | 固定有限 steal 配额 |
| S6 | 新成本模型和 GPU 联合选择 | S5 | 同一图内改变实际计划 |
| S7 | 两组完整服务对照 | 相应功能有效性检查 | native vs 新方法，各 1200 秒 |

---

## S1：提取 Marlin down 计算单元

**修改点。** 在私有 overlay 中，从固定 `marlin_template.h` 提取加载、fragment、反量化、MMA、CTA 内归约和写回；新增 `marlin_down_tile.cuh`。`native_tail_patch.py` 负责核对固定来源、复制所需头文件、建立独立 JIT 身份，不修改安装目录。

**接口。** 输入为原生 GEMM1/activation 生成的 `DownInputs` 和一个 `DownTask`；输出为该任务独占的加权 BF16 pair tile。第一版完整 K、不使用原 grid 的全局 reduction locks。

**最小版本。** 一个普通测试 kernel 静态映射 task，调用同一个 device helper；暂时没有任务队列和通信。

**CPU/GPU 验证。** CPU 检查 group/slot/channel 覆盖及 packed 偏移边界；GPU 使用真实 checkpoint，记录与原生 GEMM2 的有限误差、非有限输出、canary、资源和耗时。覆盖原生 M block、空本地组以及 B32 之外的测试形状。

**机制证据。** 编译的调用链确实进入 device helper；记录 tile 数、有效 slot 数、使用的原生权重摘要与 specialization。

**性能归因。** 比较原生 GEMM2 与静态提取版本。此时的差距属于计算组织/提取成本，不能归因于通信调度。

同阶段只做两项与架构直接相关的能力检查：GPUQ 四卡 peer IPC/系统级同步能力，以及 cooperative kernel 的 graph capture/replay。不是重启 P0。

---

## S2：实现独立的设备端归约协议

**修改点。** `task_protocol.cuh`、`peer_reduce.cuh`、`native_tail_runtime.py` 的 workspace、IPC handle 交换和清理。

**接口。** 输入为四 rank 已生成的 BF16 local partial 和人为控制的 ready 顺序；输出为每个 rank 的完整复制结果。包含 session、epoch、slot、ready、ack 和 abort 状态。

**最小版本。** `C=1`，不执行 Marlin；专门验证 `FETCH_PARTIAL→OWNER_REDUCE→FETCH_RESULT`。

**CPU/GPU 验证。** CPU 状态机模拟乱序完成、慢 rank、零载荷与重复通知。GPU 检查 payload 拷贝身份、非法访问、非有限结果、跨 rank 延迟、重复 graph replay 和 buffer-slot 复用；浮点归约差异仅记录。

**机制证据。** 实际 peer payload 字节、每类任务完成数、每次 invocation 的消费确认；不得只提供“IPC enabled”。

**性能归因。** 所有 partial 一开始就 ready 时，对照原生 all-reduce。这测量通信执行器自身的固定成本和带宽，不包含计算重叠收益。

---

## S3：统一 resident worker，先不做流水和 stealing

**修改点。** `executor.cu` 连接 S1 的计算 helper 和 S2 的通信任务；实现 task 状态、原子领取、完成计数和角色分配。

**接口。** `TailArgs＋StaticPlan＋PeerWorkspace → 完整 MoE 输出`。开始固定 `C=1、s=0`，采用明确的少量通信 worker，例如先用 `c=4` 作为功能配置，而不是宣称最优。

**最小版本。** 同一 kernel 内，计算角色领取 down/local-finalize，通信角色执行真实 peer 任务。通信等待时暂不参与计算。

**CPU/GPU 验证。** 每个 down tile 恰好一次，每个通信任务恰好一次，空本地任务仍完成协议；所有 worker 可前进，没有等待尚未调度的外部 kernel。

**机制证据。** 在同一个 kernel invocation 内记录角色、执行任务类型和实际 SM 时间区间。通信角色必须产生真实 peer 数据操作。

**性能归因。** 比较静态 tile、队列 tile、统一执行器三个层次，分离任务队列和统一资源编译带来的寄存器/shared-memory 成本。

---

## S4：加入提前交付，并接入原生 decode

**修改点。**

- executor 增加 `C=2` 的 down 优先级和 chunk 完成依赖；
- `fused_marlin_moe.py` 的私有副本拆出 front；
- `native_tail_runtime.py` 为 `DeepseekV2MoE` 增加明确的 decode helper；
- `scripts/run_native_ep.py`、`timing_hooks/sitecustomize.py` 增加互斥的新模式入口；
- graph observer 增加 tail coverage，保留现有 graph 检查。

**接口。** decode helper 调用原生路由和 front，启动原生 shared，join 后调用 tail，直接返回已全局归约的输出。prefill 调用原始函数。

必须沿用原生 hash/top-k 所需的 `input_ids` 等参数，不能重写成一个只适用于普通 router 的简化版本。

**最小版本。** 原生完整模型、静态 `c`、`C=2`、`s=0`、full graph。

**验证。** 初始继续使用已验证的精确 batch 1–32 graph、不 padding；再在独立 GPU 测试覆盖更大 batch。容量由配置推导，不能复制旧 cost 实现中的 `total<=192` 为算法限制。

新 coverage 记录“一次 native gate/up＋一次 tail”，不能继续套用旧“每层两次原生 Marlin 调用”的检查。

**机制证据。** 必须观察到 chunk 0 的实际 peer 传输与 chunk 1 的 down 计算在同一卡时间线上交叠。跨卡用 invocation/ready 因果关系关联，不直接相减不同 GPU 的绝对时钟。

**性能归因。** 对比同一执行器的 `C=1` 与 `C=2`，并单独记录 shared join 前移的损失。通信时间移动位置而总时间未缩短，不算吞吐收益。

---

## S5：加入真正的 bubble stealing

**修改点。** executor 的通信 worker 循环、共享 down 任务领取接口、初始 stealing 配额与通信恢复逻辑。

**接口。** 增加每 rank 的 `initial_steal_tiles`，不改变 compute task ID 和数学函数。

**最小版本。** 静态 `c、C、s`，至少一个通信进度 worker不 stealing。

**验证。** 通信和计算 worker 竞争同一 task 时不会重复；任务结束后异步流水全部排空；通信 ready 后只能被当前有限 tile 推迟，不会继续领取无限计算任务。

**机制证据。** 记录通信角色执行的真实 down tile 数、对应调用区间、通信 ready-to-start 延迟。受控测试可制造可复现的初始空隙，但必须标记为机制测试，不能作为自然服务收益。

**性能归因。** 固定 `c、C`，比较 `s=0` 与 `s>0`。只有角色真的执行了 down helper，才记录 `stealing_observed=true`。

---

## S6：加入实测成本模型与 GPU 搜索

**修改点。** 新 `cost_model.h`、`native_tail_calibration.py`，以及 executor prologue 的 workload 发布、leader 搜索和计划发布。复用旧校准和证据框架，不叠加旧 CTA planner。

**接口。** `workload[4]＋measured_table → 共识 Plan`；Plan 包含实际 `C、c[4]、s[4]、mode`。

**最小版本。** 先固定测量完整候选，再上传有来源绑定的有限表，最后运行设备端搜索。没有表或来源失配时明确报错；显式 static 模式仍独立可用，但不作为缺表后的隐式回退。

**验证。** CPU/GPU 在相同观测特征和表上选择一致；协议字段跨 rank 一致；同一 captured graph 内改变路由可改变真实角色/分块/steal 行为。

跨候选路由差异继续只记录，不能重新加入已经撤销的工作量一致性门限。

**机制证据。** 不只输出计划直方图，还要关联该计划下的实际 worker 角色数、chunk 发布数和 stolen tile 数。计划全部相同或 stealing 全零都如实记录。

**性能归因。** 比较最佳每层/bucket 固定计划、动态计划，以及保留完整搜索成本的正式运行，区分选择收益和 planner 开销。

---

## S7：正式两组服务对照

**修改点。** `scripts/run_native_tail.py` 复用现有自然 GPQA streaming 客户端、统一窗口、graph 检查、OwnedChild 生命周期和白名单导出；不另造一套吞吐口径。

正式只保留：

```text
native：原始 SGLang，无新 overlay/调度
peer_tail：原生前半段＋新 tail 执行器
```

两组 full graph、初始并发 32、各完整 1200 秒；自然题长、Think High、temperature0、max_new_tokens32768、正常 EOS 沿用当前配置。加载、预热、排空和清理另计。

正式窗口关闭阶段 event、逐 task 时间线等诊断；协议必需计数、工作量统计、GPU 搜索和控制消息不能关闭后另报“理想速度”。

报告全窗口有效交付 decode tokens/s、实际 batch、TTFT/ITL、完成/未完成/失败请求、数据耗尽、graph replay 和清理状态。数据耗尽时完整窗口结果仍保留，但持续 C32 比值按现有规则标记不可用，不能复制题目或提前结束窗口维持好看的数字。

性能下降是有效实验结果，不标作数值失败，也不新增性能门限阻断对照。

---

# 九、第一批可以直接交给编码代理的任务

第一批限定在 **S1＋S2 的基础实现和测试入口**，不先实现新的成本表。

| 任务 | 明确交付 |
|---|---|
| 1. 固定新方法合同 | 新文档写明 tail 范围、A 路线、shared join 前移、原生 prefill、record-only 政策；AGENTS 增加新阶段，不改历史身份 |
| 2. 建立来源验证和私有 overlay | 新 source manifest 覆盖 Marlin helpers、runner、MoE 外围及实际修改文件；独立 JIT 名称与缓存身份 |
| 3. 拆出 `DownInputs` | 从原生 front 获得 activation、grouping、slot、w2/scales；保持 Tensor 生命周期，禁止重复运行原生 down |
| 4. 实现完整 K 的 device down tile | 提取原生加载/解码/MMA/CTA reduction/writeback；移除 grid stripe/global-lock 依赖 |
| 5. 写静态映射 GPU harness | 原生 GEMM2与提取版本使用相同私有输入；记录有限误差、资源、覆盖和时间，不设置误差门限 |
| 6. 定义并实现 peer 协议 | session/epoch/ready/ack、四 owner 分片、pull partial、owner reduce、pull result；先 `C=1` |
| 7. 加入定点能力和协议测试 | cooperative graph、四卡 IPC、空本地工作、延迟 rank、slot 复用、取消/超时清理；仅输出聚合证明 |

第一批不应包括：重新搜索 Marlin CTA 比例、添加更多旧 cost 特征、重写 gate/up、改成 W4A8、加入 MTP，或切换到 dispatch all-to-all。

服务器执行继续使用既有镜像与安装包、只读本地模型和 GPUQ 实际分配。所有新增材料经固定版本与摘要的 GitHub 路径交付；不得在线 pip、apt、HF 下载或拉镜像。原始路由、激活、权重、生成内容和完整日志留在私有工作目录。

---

# 十、与 Weave 的逐项对应及证据等级

| 机制 | 论文对应 | 本方案决定 | 当前 native 状态 |
|---|---|---|---|
| 通信/计算 worker | §3.1、§4 | 同一 resident tail 内真实角色分工；以 worker 数而非虚构独占 SM 数计量 | 未实现 |
| 路由驱动资源选择 | §3.3 | 选择实际 `c、C、s`，不是两个 GEMM CTA 比例 | 只有 CTA-budget 选择 |
| 就绪后提前 combine | §3.2.1 | 改为输出通道 chunk 的设备端归约 | 未实现 |
| 原子任务领取 | §4 | 采用；增加显式 task 状态和依赖 | 当前 native 没有统一任务队列 |
| bubble stealing | §3.2.2 | 初始通信空隙内，通信 CTA 执行同一个 down helper | 未实现 |
| 全 worker 参与尾部 | §4 | 计算结束后加入剩余通信 | 未实现 |
| token dispatch/dedup | §4 | 不引入；复制输入已满足需求 | 不适用 |
| 五阶段 megakernel | §4 | 本轮不采用；只统一 down/局部合并/输出通信 | 不宣称完整复现 |

论文机制的对应依据是其 §3.1–§4；本方案的通道分块、owner 归约、原生前半段保留和 worker 驻留方式均属于针对当前执行路径的设计适配。[arXiv](https://arxiv.org/pdf/2609.21483)

最终应分别报告四个层级，而不是一个笼统的“实现成功”：

```text
source_implemented
gpu_mechanisms_verified
full_model_graph_execution_verified
throughput_result_observed
```

例如，源码有 stealing、受控 GPU 测试也执行过 stealing，但自然服务始终选择 `s=0`，应分别记录这些事实；不能称自然服务中的 stealing 已产生收益。

**这份实施大纲的首要交付不是一个更复杂的 planner，而是两个可以组合的真实部件：可由 worker 领取的 Marlin down 计算单元，以及可由同一批 worker 推进的设备端归约协议。静态组合、提前交付和实际窃取成立之后，成本模型才有值得选择的执行机制。**