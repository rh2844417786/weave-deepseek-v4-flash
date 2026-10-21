# 固定前缀：原生重复与 P0 干预拆分

## 本轮修复的诊断问题

`8cddecf` 中，同32题的 P0 组更早出现短输出，但运行时采样次数为0。未读回数据仍会执行B32图中的1032个阶段事件节点，并保留43层分组缓冲。自由生成一旦分歧，后续token、KV内容和batch也会变化，整段输出长度无法定位首次计算差异。

本轮控制同一批输入token、生成前缀、batch32、序列长度和KV逻辑位置，记录强制选词之前的原始logits。原生重复作为波动参照；P0事件和缓冲保留独立开关。未宣称已找到P0短输出的根因，也不以数值或top1差异作为门限。

## 检查安排

使用原有prepared文件前32题，输入共6059token，最长343，仍保留自然题长。每次请求使用原生批量input_ids接口，一批32题，固定8个预测位置（prefill后第1个预测，加7步decode）。仅此计算诊断使用ignore_eos=true，确保同一固定轨迹；正式GPQA自然EOS规则未改变。

| 组 | P0阶段事件 | 保留分组缓冲 | P0运行时读回 |
|---|---|---|---|
| native_a | 无 | 无 | 无 |
| native_b | 无 | 无 | 无 |
| p0_full | 有 | 有 | 禁用 |
| p0_events | 有 | 无 | 禁用 |
| p0_buffers | 无，连Event对象都不创建 | 有 | 禁用 |

五组均使用相同的固定前缀采集接口。每组启动全新原生SGLang服务、同样预热和空闲清缓存。全部组统一关闭CPU/GPU重叠调度，避免未来token占位符和异步调度干扰输入核验；GPU图内仍执行原生模型计算。该条件用于诊断，不代表正式服务吞吐。

native_a先产生8token私有teacher前缀，再清缓存并沿该前缀进行正式短对照。native_b和三种P0组都消费相同teacher。原生采样器选出teacher token后再更新ngram状态，不在模型返回之后修改token。

实际消费的decode输入必须等于上一步指定teacher token，32行必须在同一预测位置；不能用仅有配置开关代替执行证据。每组要求7次实际B32 graph replay。KV逻辑位置与长度受控，KV池槽位索引记录摘要；没有宣称不同运行的KV数值或显存地址完全一致。

## 采集边界

- 使用固定版本的ModelRunner.sample上下文和原生custom logit processor，在图外保存原始logits，再强制指定token。没有在模型层中增加新的tensor复制节点。
- 所有原始logits张量保存到服务器工作目录，四rank、8个位置均保留。只对有限值比较；非有限输出仍作为运行问题处理。
- CPU比较先算native_a/native_b，再算各P0组/native_a，输出relative L2、max/mean abs、top1匹配数、top1 margin及摘要。首次观察到的算术/top1差异只是定位信息，不是误差门限。
- 输出token必须确实等于teacher，属于“控制输入有没有施加成功”的核验；原始模型argmax可以与teacher不同，差异正常记录。
- native_a还单独跑2个预测位置：prefill采样及ngram更新完成后开启CUPTI，于下一步采样前停止，取得1次实际B32 decode的kernel时间线。原生图没有P0阶段事件。trace包含原生采样器附近的额外操作，不能直接当成吞吐实验。

客户端、采集同步和CPU拷贝本身仍是干预。因此未复现只说明当前固定前缀条件下的结果，不能自动证明原自然生成场景已修复。根据首次差异位置、原生重复波动、事件/缓冲两组的结果继续定位。

## 使用与产物

```bash
python3 scripts/run_fixed_prefix.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --setup-timeout 7200 --phase-timeout 360 \
  --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

使用现有v0.5.17镜像、Python3.12/Torch2.11/CUDA13，GPUQ分配4张H100。所有依赖来自现有镜像，不安装、不拉镜像或模型。固定前缀接口锁定9个原生源码文件；P0默认原有full/readback行为保留，新模式只由显式参数启用。

先执行更新后P0代码的4rank事件smoke，再执行五组服务，最后在已有镜像中用CPU比较私有张量。CPU比较使用`--network=none`，服务保留localhost及rank通信。

公共文件：`experiment.json`、`comparison.json`、输入配置与teacher摘要、每组各阶段的graph检查、四rank阶段证明、原生trace的公开kernel资源字段、启动/运行/清理证据和manifest。

私有文件：teacher token IDs、每个位置的原始logits `.pt`、完整Chrome trace、完整日志。原始logits预计占数GiB本地工作盘，保留用于复查；永不复制到公开报告或GitHub。数值差异始终record_only。
