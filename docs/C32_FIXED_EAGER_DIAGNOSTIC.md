# C32 固定计划 eager 阶段诊断

目标是测清已有 `tiled + eager + (c,K,steal)=(1,1,0)` 路径的耗时。入口为 `scripts/run_concurrency_diagnostic.py`，只启动 Weave 诊断组。沿用上一轮九条 SWE-Bench 输入重复200次的 custom JSONL、并发32、temperature0、输出1024 tokens 和600秒负载窗口。SGLang benchmark 默认忽略EOS以达到指定长度。该入口不运行原生对照或调参扫描。

设备阶段探针沿用既有 CUDA 实现，源码身份仍为 `95e8d6c87ab67514c2b090834d7cc43ec352acfb24f088bb749f8482e5a9d7c0`。可复用该身份下已通过的全部43层 checkpoint/core 证据。新的 Python 采样、启动和收集代码另行绑定 SHA256。本次诊断不需要重新跑374点校准，也不加载 measured bank。

固定计划下，一个通信CTA负责dispatch，其余131个CTA可以领取计算任务。K=1，通信CTA不steal。`WEAVE_CAPACITY=2048` 是本次MoE调用每个origin的行容量；global decode batch32通常每origin8行，与累计KV上下文长度不同。

入口会完成以下流程：

1. 检查调度器分配的四张GPU、已有镜像、本地checkpoint与core身份，校验已有custom数据的SHA256及9×200结构。
2. 使用现有镜像，在四张GPU上测试完整forward CUDA event ring的记录、覆盖和reset。此smoke不加载模型、不下载内容。
3. 显式启动tiled、eager、固定`1,1,0`，服务max-running=32，保留graph max-batch参数32但graph处于disabled。开启设备阶段探针和完整forward采样，默认stride16/capacity8。
4. 使用同一镜像中的SGLang benchmark与只读本地tokenizer。压测镜像不分配GPU。benchmark完成自身1条、最多32输出token的warmup后，在首个1024输出token请求发出前，清零四个rank的探针并写入窗口开始标记。
5. 到600秒停止本次负载容器，向本次独占服务发送abort-all，等待idle flush成功，再收集四rank。记录从停止负载到idle的时间；设备统计包括这段drain，不能冒充严格600秒吞吐。
6. 校验固定计划、无bank、eager、源码/模型/镜像/设备身份、43层采样及完整forward计数。四rank必须都有至少两个与完整forward对齐的batch32耗时样本。逐rank输出单独的C32 layer摘要。
7. 清理本次创建的容器，生成公开导出和逐文件SHA256。失败也保存已经取得的公开证据；原始日志、prompt、generation留在工作盘。

SGLang benchmark本身可能在请求失败时仍返回退出码0。wrapper单独记录formal请求started/succeeded/failed/cancelled和已完成请求输出token数。请求失败、负载提前退出或缺少C32计时样本不能标记`diagnostic_complete`。其中未完成请求的流式token不在completed计数中，所以该计数不作为全窗口TPS。

默认只生成命令预览：

```bash
python3 scripts/run_concurrency_diagnostic.py \
  --work-root /srv/weave-work \
  --core-run-dir /srv/weave-work/cuda-functional.2349166e7413 \
  --dataset "$C32_DATASET" \
  --export-root /srv/weave-work/public-reports
```

`C32_DATASET` 必须从上一轮实际负载命令定位，不能把9行prepared token-ID文件当成1800行custom JSONL。GPUQ实际授予四张H100后，在作业内执行同一命令并增加`--execute`。保留GPUQ设置的`CUDA_VISIBLE_DEVICES`，不手动指定物理编号。若上面的core目录不存在，可定位其他同源码、全部43层已通过的core目录，入口会严格验证。

主机Python只需标准库。服务与压测依赖使用已安装的`lmsysorg/sglang:v0.5.17`，所有docker run使用`--pull=never`。模型、tokenizer读取本地只读目录，HF离线变量开启。四GPU event smoke使用`--network=none`；服务与压测使用host网络访问localhost。benchmark shim校验锁定SGLang提交`29481685462732237d80d86076d6563e1f658102`的serving/custom数据读取源码Git blob身份，若镜像中的代码不同则停止并要求导出源码核对，不在线安装。

单独操作底层launcher时，新开关为`--execution-profile-fixed-plan`，必须同时给出`--execution-profile --kernel tiled --decode-graph disabled --checkpoint-content-sha256 <verified digest>`，且不能传`--profile`。统一入口会从通过的core中取得digest。此前使用measured bank/full graph的诊断保持原有校验。

输出在`/srv/weave-work/public-reports/c32-diagnostic-<id>/`，包括：

| 文件 | 内容 |
|---|---|
| `experiment.json` | 阶段、最终状态、固定配置、源码/数据/模型身份、GPUQ job、窗口与drain时间 |
| `launch.json` / `runtime.json` | 实际Docker argv及镜像、依赖、硬件检查 |
| `load-window.json` / `load-progress.json` | formal窗口开始时间及请求计数，不含请求文本 |
| `forward-profile-smoke.json` | 四GPU event ring验证 |
| `weave/execution-profile-rank0.json` 至rank3 | 43层prefill/decode设备阶段及完整forward采样 |
| `weave/diagnostic-summary.json` | 实际batch分布、按sequence对齐的完整forward与43层persistent时间、单独的C32 layer汇总 |
| `*-lifecycle.json` | 本次容器清理状态 |
| `manifest.json` | 公开文件SHA256 |

完整forward的`gpu_stream_ms`来自CUDA events，覆盖当前stream从模型forward开始到返回期间的GPU工作、依赖等待和host发射间隙；相关其他stream只有经当前stream依赖才包含。`host_enqueue_ms`是CPU发射该forward所用时间，不是GPU完成时延。它们均不包含完整HTTP请求队列、所有外部采样/发送流程，不等于用户TPOT。

设备阶段使用`%globaltimer`。`kernel_span_ns`从persistent worker进入到退出完成屏障，不含prepare、trace尾部写入和其他外围GPU操作。各CTA累计耗时不能直接作为层latency，阶段envelope会重叠，不能相加算百分比。队列和结束等待也可能由其他rank计算/发射较慢引起，不能直接解释成链路传输慢。

摘要的`aligned_recent_decode`仅组合相同rank、相同phase sequence的完整forward与各层样本。`outside_persistent_ms`为完整forward GPU event时间减去43层persistent span之和，包含attention、shared experts、prepare、copies、外层all-reduce、host发射间隙等，也受两种设备计时方式精度影响；它不能被直接称为CPU开销。若出现负值，应回看原始时间与对齐信息，不强行归因。

`layers_batch32`只汇总对齐后valid_rows=32的样本。普通`layers`可能混合不同batch，不能替代C32结果。`fraction_decode_calls_at_batch32`是调用次数比例，不是时间占比。TP4复制的模型行数只取rank0，不把四rank相加。所有文件固定标记`instrumented_diagnostic`与`used_for_formal_performance=false`。

本地已验证CPU身份检查、采样ring/实际batch、warmup窗口钩子、请求失败统计、C32样本门限和导出隐私。GPU计时行为及真实43层C32诊断仍由本次服务器运行验证。仓库完整CPU suite中已有GPQA provenance/order测试冲突，本次未修改相关数据读取逻辑。
