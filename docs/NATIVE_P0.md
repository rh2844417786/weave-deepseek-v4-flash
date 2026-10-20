# 原生 SGLang 吞吐优化：P0 诊断

## 本轮目标

2026-10-20 用户将主线改为：在原生 SGLang 上提高 DeepSeek-V4-Flash 的模型吞吐。首轮使用 C32 保持控制变量；后续可以研究其他并发和实际 batch。论文复现路径与历史结果保留，不作为原生优化的必经步骤。

本轮只观察原生执行，不修改 Marlin grid、CTA 配额、权重精度、激活、shared experts 或 collective。希望回答：

1. 实际捕获的是 normal、dual-stream，还是其他 MoE 分支？shared experts 的 TP 配置是什么？
2. routed 的 gate-up、激活、down、分组与输出归约分别用多久？shared 分支何时开始和结束？
3. 两分支实际重叠多少，哪一支暴露在另一支完成之后？join 后的工作和 TP all-reduce 区间多长？
4. Marlin 实际用了什么 grid/block、线程数、寄存器、共享内存和 M/N/K tile？真实专家行数分布如何？
5. 这些现象集中在哪些层、哪些 rank？

只有时间区间重叠或 shared tail，还不能证明资源争用导致减速。后续若值得试 CTA 配额，需要保持原生计算语义，再以有/无配额、串行/双流的配对实验判断因果和吞吐收益。

## 运行契约

- 已有镜像 `lmsysorg/sglang:v0.5.17`；SGLang `29481685462732237d80d86076d6563e1f658102`，Torch 2.11/CUDA 13/Python 3.12。
- 原始只读权重 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`，routed MXFP4/E8M0 保持原样。
- GPUQ 分配四张 H100；使用分配的 `CUDA_VISIBLE_DEVICES`，无固定物理卡号。
- TP4/EP4、原生 MoE；full CUDA Graph、精确捕获 1..32、关闭 padding；客户端并发和 max-running-requests 均32。
- GPQA Main 自然输入，Think High、temperature=0、max_new_tokens=32768、正常 EOS。既有准备文件 SHA256 为 `cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888`。
- 单组原生服务测量1200秒，加载/预热/清理另计。保留既有数据耗尽、取消、缓存清理、请求和 token 计数语义。
- `measurement_mode: instrumented_diagnostic`，`formal_gpqa: false`，`numerical_policy: record_only`。本轮吞吐只是带采样的运行记录，不与历史未采样吞吐计算加速比。

## 独立入口

```bash
python3 scripts/run_native_p0.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --setup-timeout 7200 --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

使用服务器已有 Python 3.12，`--records` 可替换为已有、哈希一致的绝对路径。省略 `--execute` 只打印计划，不访问 GPU 或 Docker。

入口先执行 Torch 自带算子的四 rank 双流图采样 smoke，再启动原生模型。它不执行 Hopper/persistent 候选的计算验收。默认读取 `reports/server/gpqa-c32-9f1635df2afe` 中原生对照使用的 checkpoint 内容证明，仅用于确认模型身份；不引用候选算子数值状态作为门限。

若模型各文件 stat 与证明一致，复用历史完整内容哈希，并重新核对 config/index 内容；若 stat 不同，重新哈希全部本地 shard，结果必须仍是同一模型。报告明确区分这两种身份检查方式，不把 stat 检查称为重新读取全部权重。

## 采样方法与边界

- 图捕获前分配并初始化 external CUDA timing events。在 B32 捕获时包装原生调用；其他 batch 和 eager/prefill 不写这些计时事件。
- 每128次实际 B32 replay 读一次事件，保留最近64个样本；每个样本记录实际 replay sequence 和 B32 ordinal。
- 事件在原调用所在流记录。采样结束同步外层结束事件，先读并冻结时间，再读 grouping 聚合，避免后续 replay 覆盖事件。
- 首个选中 replay 开启一次 Torch/CUPTI trace。只导出核函数名称、启动资源和标量时间；原始 Chrome trace 留在服务器私有目录。该 replay 不进入普通阶段时间统计。
- 43层每层两个 routed Marlin kernel，要求共86个实际记录。按 GPU 时间排序映射到层和角色，并核对捕获时的 block size、M/N/K 兼容性、量化类型、EP/bias 标志。
- grouping 复用原生 B32 分组缓冲，仅在样本完成后汇总有效本地 pairs、groups、活跃专家数、最大行数和行数直方图。不上传 token/expert ID 或原始数组。
- 公共 `/graph-coverage` 快照只读已冻结的 CPU 数据；四 rank 使用测量前后实际 replay 差分。仅汇总测量窗口内的样本，并要求四 rank sequence 对齐。
- “critical rank”是每个样本整图时间最长的 rank，其所有阶段值取自同一 rank。不同 rank 的阶段最大值不能相加。

### 时间字段怎么读

`intervals_relative_to_graph_ms` 给出各层各阶段相对同一张图起点的 start/end/duration。`branch_overlap_ms` 是 routed/shared 区间交集。`shared_tail_ms` 和 `routed_tail_ms` 是各分支在另一分支结束后仍执行的区间长度。`moe_post_branches_ms` 是两个分支结束至整个 MoE 返回的区间。

原生 `_shared_expert_tp1` 路径在 all-reduce 后还可能直接原地加 shared 输出。此加法未单独包装：元数据中的 `shared_tp1_post_reduce_add_uninstrumented` 和 `shared_merge_includes_shared_add` 会说明边界，不能把 `shared_merge` 一律称为完整 shared 加法。

all-reduce 事件区间包含等待，不能直接当成纯网络耗时。可选阶段没有执行就不提供对应字段，不补零。`outside_moe_ms` 包含注意力及其他图内工作，不能全部归因于 attention。重叠分支的 duration 不可相加作为层延迟。

事件节点、采样同步、CUPTI 和 grouping D2H 都可能扰动执行；分组缓冲保留也可能影响图内存复用。因此本轮用于选优化方向，不是正式性能证明。

## 证据与失败处理

- `native-p0-summary.json`：四 rank、43层的阶段区间、暴露尾部、分组统计、启动资源、critical-rank 汇总。
- `native/graph-before.json`、`native/graph-after.json`、`native/graph-check.json`：图是否实际 replay、实际 batch 及原始采样标量。
- `native-p0-gate/`、`native-p0-proof-check.json`：双流事件、CUPTI 元数据、更新输入、NCCL 与图释放检查，明确不是模型执行证明。
- `native-checkpoint-proof.json`：独立的原生 checkpoint 身份检查。
- `native/client-summary.json`、`native/launch.json`、`native/runtime.json`、生命周期与 manifest：负载、环境、代码与清理记录。

缺失 CUDA/CUPTI 元数据、源文件不匹配、非有限输出、运行/清理异常或覆盖不全，都保留失败证据。数值误差和生成内容差异不设门限。原始 trace、题目、token IDs、生成文本及私有日志不进入 GitHub。
