# B32 peer-tail：校准瓶颈、无重叠原因与修复

日期：2026-10-24。分析输入为服务器回传 `26e314f` 中的
`reports/server/native-tail-e7b1ad7a62c0`，运行源码 `ccc4d46`。
该轮两组完整1200秒，native **1901.44 token/s**，peer-tail **221.10 token/s**；
Graph 与两阶段清理均完成。以下是对已有证据的分析，本次源码修改尚待 H100 验证。

## 1. 慢在哪里

读取四份真实权重校准表的 B32 项，共4 rank ×2结构组。每组包含两种留出输入，
所有时间单位均为微秒，完整 MLP 采用独立的无阶段事件图测量。

| B32校准指标 | 四rank、两结构组中的范围 |
|---|---:|
| 搜索 planner 的诊断中位数 | 1104.38–1212.42 us |
| 留出输入的原生完整 MLP 中位数 | 121.57–310.43 us |
| 留出输入的自动选计划完整 MLP 中位数 | 2127.81–3450.37 us |
| 同一留出输入中测到的最快固定计划中位数 | 977.79–1498.02 us |

固定计划也明显慢，说明仅改变候选选择不能解决基础执行成本。自动选择还要承担
毫秒级的 planner；fixed 与 search 的差值也包含不同计划和工作量差异，不能全部
记作纯搜索时间。每个候选实际路由仍按原策略记录，不要求相同。

具体例子：rank0、g0、B32、第一个校准 anchor、候选9（COMM2、chunks2、无steal）：

| 阶段 | chunk0 | chunk1 |
|---|---:|---:|
| down计算区间 | 162.56 | 157.95 |
| finalize：top-k求和、shared加入、partial发布 | 1118.46 | 1056.26 |
| owner拉取并归约 | 145.92 | 149.50 |
| 结果拉取 | 147.46 | 150.78 |

这是一份诊断曲线，不能将跨chunk、跨rank区间直接相加作为完整层耗时。
校准使用真实模型权重和代表性输入，不是正式 GPQA 窗口的逐层计时分解。

## 2. 为什么没有计算通信重叠

旧代码中，一个chunk所有down完成后，**一个CTA负责整个chunk的finalize**。
B32、hidden4096、chunks2、128线程时，这个CTA要处理65536个输出元素，
每个元素依次读6个专家结果、舍入BF16、加shared、写partial。chunk ready要等这些
工作全部完成，其他rank才能读取它。

已有rank-local时间戳能直接验证这一依赖。选取服务中常见的候选9（C2、COMM2）和
候选15（C4、COMM1），每个候选均有4 rank ×2组 ×4 anchors ×7重复 = **224个样本**：

| 同卡时序差 | C2候选9 | C4候选15 |
|---|---:|---:|
| chunk0 finalize结束相对全部本地down结束，最小值 | +694.30 us | +331.33 us |
| 首个peer任务开始相对全部本地down结束，最小值 | +745.82 us | +341.86 us |
| 上一行按rank/组统计的中位数范围 | +897.90–938.91 us | +397.86–432.29 us |

全部为正：这些样本中，开始通信时，本卡已经没有后续down可供重叠。
这与executor检查中的 `overlap_observed=false` 一致。
旧peer任务起点包含成功领取任务前的扫描，因此它是一个偏早的起点；即使这个起点
也晚于全部down结束，上述“没有重叠”的结论仍成立。旧表没有精确chunk ready时间，
不把finalize结束伪装成该时间。新版本会单独记录ready发布后的本地时间戳。

源码还让396个驻留CTA的leader每轮重复扫描peer headers、ready与abort。
这是已识别的重复系统域流量，但现有证据没有隔离测量它单独贡献了多少微秒。
本轮同时将该流量限制到实际需要推进通信的worker。

## 3. 本轮代码修改

### 3.1 CTA协作候选搜索

候选集合、double算术、短名单、tie-breaking及CPU oracle保持原定义：
先并行插值4 rank ×最多22个计划，再并行评估22个同构候选及最多48个混合候选，
最后按确定的字典序选最小值。所有线程在同一CTA内按阶段同步；rank间发布/ack协议
保留。没有逐token CPU读回，也没有新增planner kernel launch或重新捕获。

### 3.2 多CTA分块finalize

每项finalize任务处理 `threads ×8` 个连续输出元素。B32/128线程时，C2每chunk
64项、C4每chunk32项。各元素原有的FP32 top-k累加顺序、BF16舍入和shared加法保持。
worker用有界CAS领取不相交的切片。每个写线程先做system fence，CTA汇合后以
acq_rel计数发布完成；最后完成的CTA才能发布整个chunk ready。
这一发布顺序依据 [CUDA 13.0内存栅栏和同步语义](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#memory-fence-functions)，
仍需服务器memcheck与真实重放验收。

每次诊断检查finalize的expected/claimed/completed与已发布chunk数，缺失、重复、
提前发布或非法访存仍是有效性失败；有限数值差异不设门限。

### 3.3 有界通信推进

block0负责跨rank abort和完成状态推进；其他计算CTA读取本地stop状态。
peer headers验证成功后在该次invocation内缓存；计算CTA不再每个tile扫描跨卡ready。
全部本地chunk完成后，最多额外启用每个实际peer任务一个通信助手（5×chunks），
防止所有396个CTA同时争抢少量通信任务。原COMM保留，有限初始stealing保留。
worker仍是CTA配额，不能解释成独占物理SM。

## 4. 验证与输出

- CPU用真实 `select_tail_plan_parallel` 函数和线程屏障执行120组测试，对照独立串行
  C++及既有Python oracle；包含不同候选数量、缺少第二候选和相同成本的tie。
- CPU检查finalize切片对输出域的完整、互斥覆盖，以及通信助手上界。
- 既有跨warp时钟偏移回归保留；GPU的serial/C1/C2/stealing、空本地工作、Graph重放、
  canary、锁与memcheck矩阵保留，不追加正式服务实验组。
- GPU检查新增 `finalize_task_coverage_verified` 和 `execution_timing_summary`。
- 现有诊断图使用任务的真实区间并集求交，记录chunk0通信与后续down重叠时长。
  通信计时从成功领取任务后开始，避免把等待/轮询计入有用重叠。
- 新 `calibration/performance-analysis.json` 从已有校准生成，不增加GPU运行；纳入
  原公开导出白名单。原始任务事件仍只留服务器私有目录。
- 正式1200秒窗口继续关闭阶段事件与逐任务trace。原有聚合统计、协议和planner开销
  均计入正式吞吐。

单独复核旧报告（CPU，无下载、无GPU）：

```bash
python3 scripts/analyze_native_tail.py \
  --report-dir /srv/weave-push-repo/reports/server/native-tail-e7b1ad7a62c0 \
  --output /srv/weave-work/native-tail-e7b1ad7a62c0-b32-analysis.json
```

脚本核对四份输入校准文件的manifest SHA256。该输出为聚合分析，不改写原报告。

## 5. 仍需GPU回答的问题

1. 新kernel的实际寄存器、local memory、shared memory及resident workers是多少？
2. planner/finalize的实际成本是否下降；chunk ready是否赶上后续down？
3. 无注入B32/C2场景是否观察到真实重叠？若仍为零，按新同卡时序照实报告，不伪造延迟
   来产生性能重叠，也不把受控stealing实验当作自然负载收益。
4. CPU/GPU当前workload的选择是否一致，所有任务和发布是否完整？
5. 两组完整1200秒吞吐是否改善？即使改善，也不能在没有证据时声称超过原生。

提取的full-K Marlin tile、单CTA owner payload传输、shared join前移，以及近似DAG
预测仍可能影响性能。本轮针对已测出的planner和finalize大头修复；不承诺解决全部
8.60倍差距。旧表绑定旧实现，必须重新校准，旧缓存和历史证据原样保留。
方法仍为 `native_marlin_peer_tail_v1`，不是完整Weave论文复现。
