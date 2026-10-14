# Staged-direct decode 与 C32 服务对照

## 本轮范围

P0/P1 的真实 trace 回放已经在09f905e回传：直接tile映射相对staged132的配对时间比中位数约1.68。这个结果支持把候选接入服务，但不是服务吞吐结果。本轮新增实际decode入口、在线GPU分组、跨rank汇总及同口径C32流式测量。

继续使用原版混合精度DeepSeek-V4-Flash、本地只读checkpoint、四张调度器分配的H100、TP4/EP4和现有`lmsysorg/sglang:v0.5.17`。服务器不安装包、不下载模型、不拉镜像。历史P0/P1源码gate不能替代本轮新源码gate。

## 服务路径

`scripts/run_native_ep.py --decode-implementation staged_direct`显式启用新路径。默认仍是`persistent`。新路径要求tiled、hot-cache、固定计划`1,1,0`、eager、decode batch1..32；拒绝profile bank、execution-profile、replay-capture或graph组合。

每个decode层的执行顺序：

1. 将实际输入、原始router IDs及权重复制到预分配workspace；GPU验证IDs并按token/slot顺序分组，每组最多16行。
2. 直接tile映射运行原有gate/up和down数学实现。
3. 写本rank负责的BF16舍入后routed products到FP32 slot buffer；非本rankslot置零。
4. 在现有TP device group做一次slot buffer的NCCL SUM。每个slot恰有一个专家所有者。
5. 按slot0..5顺序FP32求和，再转BF16，只保留本rank的origin rows。
6. 返回现有SGLang共享专家、缩放及外层TP汇总路径。

这样保留了当前专家权重、routed-product舍入、slot求和顺序和shared-expert边界。**本版每层比原路径多一次slot all-reduce，现有外层TP reduction也保留；这些开销全部进入服务计时。** 不把回放中的纯计算比例当作最终收益。后续若要去掉冗余通信，需要另做数值与服务验收。

Prefill明确使用缓存后的persistent路径；全部decode使用所选实现，没有native专家回退。层级计数与model CPU batch histogram在idle后导出，用于验证43层实际执行。正式窗口不开设备阶段探针，不做逐层CPU路由下载或显式GPU同步。

## 在线分组与缓存

每层raw权重只绑定一次，bound handle持有张量和workspace。全模型顺序复用一个staged workspace及缓存好的B1..32前缀视图。在线分组不调用GPU `.item()`。

实际group_count留在GPU。直接grid使用容量上界，超出实际group_count的CTA退出；不是根据每步实际group_count在CPU计算grid。生产B32的group容量上界72，gate grid上界2304，down grid上界4608，资源记录明确保存这些数值。P1冻结回放的实际容量grid与本服务上界grid不是完全相同的launch，因此服务收益必须重新测量。

## 运行顺序

先提交新源码core，在已经取得四卡分配的GPUQ作业内：

```bash
export WORK_ROOT=/srv/weave-work
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
export CHECKPOINT_TESTS=1
export CHECKPOINT_LAYERS=all
export CORE_RUN_NAME="cuda-functional.staged-c32-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
bash scripts/run_cuda_functional_gpuq.sh
export STAGED_C32_CORE="$WORK_ROOT/$CORE_RUN_NAME"
```

保留GPUQ设置的`CUDA_VISIBLE_DEVICES`。core失败时停止，不复用72592ee的旧结果。

再执行完整C32控制器：

```bash
python3 scripts/run_c32_comparison.py \
  --work-root /srv/weave-work \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --dataset /srv/weave-work/bench-concurrency32-dataset.jsonl \
  --core-run-dir "$STAGED_C32_CORE" \
  --seconds 600 \
  --setup-timeout 7200 --gate-timeout 7200 \
  --acceptance-timeout 1200 --drain-timeout 600 \
  --execute
```

数据必须是原有9种输入各200次、共1800行的custom JSONL，SHA256：
`7d7bcdb9996f9293b63d337f21ee977f39013107ef23c2c411de2310d64501c2`。
客户端按文件顺序遍历同一批自然输入，不改写、不补齐输入长度。路径可以改变，内容摘要必须相同。

控制器先在四卡上运行`tests/gpu/check_online_staged.py`：每rank6个动态batch合成用例和43个真实权重B32用例，共49项。核对GPU分组、空rank、切换权重、稳定地址、原始slot汇总、独立reference和缓存persistent输出。门限保持relative-L2<=0.025、rtol0.03/atol0.02。

通过后依次运行三组：

| 组 | 图模式 | 用途 |
|---|---|---|
| native | eager/disabled | 原生SGLang对照 |
| persistent_cached | eager/disabled | 本次优化控制组，hot-cache开启 |
| staged_direct | eager/disabled | 新decode候选，prefill为cached persistent |

每组先用32个请求、每请求8token做生成验收，后两组与native实际token ID逐项比较。原始ID仅保存在工作盘；公开记录只含匹配数量和摘要。Logprob数值不作为门限。验收失败停止后续依赖步骤。

然后执行同一600秒窗口：concurrency32、max_new_tokens1024、temperature0、ignore_eos=True。每组有窗口外warmup。服务器ready、验收和加载准备时间不进入窗口分母。用户已允许按需要延长GPUQ时间；作业总上限应覆盖所设各阶段预算和清理余量。所有输出使用新目录，本入口不混合旧组结果续跑。

## 吞吐口径

主指标为**客户端在固定窗口内观察到的decode tokens/s**：累计流式`meta_info.completion_tokens`的增量，计入到截止时仍未完成的请求，并扣除每个有输出请求的首token。窗口为`[start,deadline)`，按完整SSE事件接收时间归属。它包含真实服务的prefill/排队影响及客户端传输成本，不能称为纯GPU decode kernel速率。

客户端验证每个返回RID与发送RID相同；每次运行的wire RID使用独立UUID前缀，避免旧请求的延迟abort误取消新请求。验收使用稳定逻辑ID做跨组比较。

同时记录完整/失败/未完成请求数、输出token计数、TTFT及chunk间隔。只有相邻两个事件都新增一个token时才记录token ITL；多token chunk不伪造逐token时间。窗口提前耗尽、计数缺失/倒退、错误响应、清理失败或没有达到32个在途请求，结果均不通过。

Weave两组用idle前后coverage差值确认所有43层实现、额外slot collective计数和实际B32 forward。Native收集测量阶段的scheduler decode running-request日志计数；这是调度器采样证据，不是GPU event时间。三组统一eager和客户端计时，不能再引用历史native fullgraph日志TPS作为同口径比值。

## 输出与回传

成功状态为`comparison_complete`。`comparison.json`包含staged/native及staged/persistent两个比值，后者标注为优化控制。仅完整且通过验收的三组才生成有效比较。

程序打印`work_directory`和`export_directory`。只把后者复制到`reports/server/<run-name>/`并推送。公开包有逐文件SHA256、源码/模型/镜像身份、新core与online gate、三组launch/runtime/验收摘要/客户端统计/coverage/清理状态。原始prompt、token IDs、生成文本、完整日志、acceptance-private.json都留在工作盘。

失败时保留公开包及私有错误日志，报告具体阶段和错误；不改门限、不切换图模式或绕过staged执行。当前本机只完成CPU测试/源码审查，服务器CUDA和实际吞吐结果尚待回传。
