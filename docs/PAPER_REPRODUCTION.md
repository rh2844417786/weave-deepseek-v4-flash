# 回到 Weave 论文机制：Hopper persistent 集成

## 身份与声明范围

原 `hopper_fp8 + NCCL slot SUM + finalizer` 路径是 **operator_candidate**，没有执行论文的SM/chunk调度。历史报告原样保留，不重新解释成论文结果。

新路径标识为 `hopper_persistent_paper_v1`，decode/prefill 都使用 `hopper_persistent`。它在 persistent kernel 内执行设备端 Hopper tile、去重dispatch、动态通信/计算SM划分、chunk combine和bubble stealing。源码中的实现状态为 **paper_candidate / paper_mechanisms_implemented_gpu_pending**，不把源码、CPU测试或候选算子吞吐称作完整Weave。

服务器证明分别给出 `kernel_mechanisms_verified` 与 `paper_service_execution_verified`，并继续保留 `complete_weave_claim:false`。这些状态表示明确范围内的硬件执行证据；论文原始实验协议、性能消融、成本预测精度等仍须独立核对。参考：[Weave论文设计](https://arxiv.org/html/2609.21483v1#S3)。

## 本次真实执行结构

独立CUDA扩展位于 `experimental/hopper_persistent/`。不修改历史 `csrc`、Hopper standalone或prefill候选CUDA。

```text
每层原始路由 + origin本地token
    ↓
epoch准备
    ↓
一个resident persistent grid
    ├─ 根据路由量和M64分组成本，设备端联合选择c/K与steal预算
    ├─ 通信worker：去重拉取payload → 有界偷取计算tile → chunk返回
    ├─ 计算worker：ready group的gate/up、SwiGLU、down tile → 加入尾部返回
    └─ origin按slot顺序聚合、发布完成epoch
    ↓
SGLang共享专家、缩放与外层TP处理
```

routed路径没有独立Hopper算子调用、NCCL slot SUM或子kernel发射。外层SGLang通信保持原语义，不能把“routed路径无NCCL”扩展成“整个模型无NCCL”。

### Hopper tile

复用当前FP8/MXFP4计算约定：M64N64、packed权重TMA、E4M3 WGMMA、每K32的E8M0缩放、BF16 gate/up/activation/down边界。将原输入改为从persistent dispatch cache或group_mid读取并量化到shared memory，使同一worker CTA直接执行tile。gate/up共用当前K128的输入量化。

**已知成本：输入量化仍会按输出tile重复。** 此版本恢复执行结构，没有宣称这一实现已经达到高效standalone GEMM水平；之后应针对真实GPU数据优化复用与流水。权重绑定和tensor map均在捕获前完成，descriptor作为grid-constant参数保持设备可访问。

### SM与调度

每CTA128线程，动态shared memory预约超过每SM容量的一半，要求occupancy=1、grid=SM数，并记录/检查每个CTA实际SM ID唯一。不能仅用CTA数量冒充SM划分。

调度按ready group/chunk顺序领取gate/down输出tile，不等待尚未ready的输入组或未完成gate的down任务。通信worker结束dispatch后有界偷取tile，然后按chunk完成状态返回输出；记录实际偷取、dispatch与计算同时活跃、返回与计算同时活跃的计数。

### 成本表

新后端重新实测，不复用BF16或standalone Hopper成本表。每rank分别记录本地token上限8/32/128/512/2048的桶，候选c为1/8/16/32、K为1/2/4/8。计算模式和通信模式保留IPC协议；有效FLOP/s、有效byte/s由实际事件耗时和work/traffic计数推导。

v1明确使用保守模型常数efficiency=1、overlap=0；这些常数不是伪装成拟合测量。实际每层/每rank的c/K选择仍在GPU上根据当前路由和分组数完成。预测模型准确度与性能收益尚待测量，不等同于论文原始硬件曲线已经完整复现。

## 服务器验证顺序

统一入口：`scripts/run_paper_reproduction.py`。

1. **小尺寸四卡机制检查**：H128/I128/E32，固定无chunk控制、dispatch流水、chunk返回流水、stealing，以及实测成本表驱动的动态和偏斜路由；捕获B32图后改变输入/路由再次重放。检查实际SM唯一、group/tile守恒、routed collective调用数为0。受控案例必须观察到各机制的活动计数，未观察到就不能宣称验证成功。
2. **真实尺寸成本校准**：H4096/I2048/E256，原checkpoint第0层，五个token桶，四rank分别实测并保存成本表。
3. **原权重检查**：43层分别运行B32，另对第0层运行偏斜路由B128以覆盖M64分组边界。与原Hopper参考比较覆盖完整batch；独立FP8/BF16 oracle只覆盖全局前两行，JSON明确记录 `numerical_coverage`。所有有限误差只记录，无精度门限。
4. **真实服务**：原生SGLang与新paper persistent各32并发、full decode graph、GPQA Think High各1200秒；新paper路径prefill也走persistent，不能回退operator candidate。

正常子进程退出、完整四rank证明、CUDA和Python源码身份、成本表digest及当前checkpoint/image一致后才能进入下一阶段。捕获图先reset再销毁进程组；父控制器保留120秒清理超时。局部reference错误在进入相关collective前通过Gloo协商。

服务验证逐层核对device decode计数=graph replay数、prefill计数、GPU动态选择调用数、实际c/K直方图，以及偷取/重叠计数。若实际服务某机制未出现，报告明确写 `*_observed:false`，不能用“开关已打开”替代活动证据；动态算法在小负载下选择K=1或steal=0也不强行改成别的值。

## 交付与限制

使用已有镜像、包、只读模型、GPUQ分配和GitHub材料。现有候选对照保持可用，历史结果不改写。原始输入、路由、激活、生成文本、token IDs和完整日志不公开；导出仅白名单标量与hash。

本机不能编译/运行CUDA。首轮服务器可能暴露NVCC/PTXAS、TMA/寄存器资源或IPC进度问题，须按失败阶段修复，不能跳过验证或自动退回旧算子。运行预算按校准/加载需要延长，建议首次GPUQ给12小时；阶段超时4小时，正式两组各20分钟不变。

执行Prompt见 `docs/SERVER_PROMPT_PAPER_REPRODUCTION.md`。
