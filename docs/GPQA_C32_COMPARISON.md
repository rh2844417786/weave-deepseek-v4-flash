# GPQA Main：两组 C32、各20分钟对照

> 方法身份更正：本文中已有的 Hopper/“Weave”实验组是独立算子候选，未执行论文persistent调度；新论文机制路径见 [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md)。历史数值保持原样。

## 固定实验设置

| 项目 | 设置 |
|---|---|
| 数据 | 原始GPQA Main448题，seed42固定顺序，选项使用既有固定打乱方式 |
| 输入 | 已离线编码的自然长度token IDs；不拼接、不补齐、不重复题目 |
| 推理模式 | DeepSeek-V4-Flash Think High |
| 生成 | temperature0，max_new_tokens32768，ignore_eos=False，允许正常EOS |
| 组别 | Native SGLang；Weave Hopper FP8 prefill+decode |
| 并行 | 同一组GPUQ分配的4张H100，TP4/EP4；默认 eager，可显式选择两组 full decode graph |
| Batch | 服务max-running-requests32，客户端32个并发worker持续补充 |
| 时长 | 每组正式窗口1200秒；模型加载、证据核验、预热和清理单独计时 |

这里沿用当前C32服务比较的固定测量窗口，不调用旧的batch1 GPQA预算入口。实际decode batch随调度、prefill和请求完成变化，报告保存原生scheduler统计和自研all-rank batch histogram；不填充输入来强迫每个forward都是32。

本轮比较性能，`accuracy_evaluated:false`。测量请求不采集logprob，减小逐token诊断开销；两组设置相同。数值差异/生成内容一致性不作为性能门限。

## 计时和请求生命周期

先完成32个最多8token的自然EOS预热请求，确认缓存立即清理成功。客户端通知控制器空闲，控制器记录覆盖和日志起点后放行。此后才启动1200秒窗口。

主指标是窗口内客户端收到的累计decode token增量除以1200，包括未完成请求的已输出部分，每请求排除首token。TTFT与可用单token ITL也是客户端接收口径；它们不是纯GPU时间。请求RID必须与响应一致，错误/异常终止不能当作正常EOS。

到点停止新请求，取消读取并中止本客户端拥有的未完成请求。公开行结果只保留原CSV行号、计数、时间和终止类型；不保存输入、生成文本、token IDs或原始RID。两组汇总列出完成/未完成/未尝试数量及共同完成行号。

448题只跑一次。如果某个worker完成请求后已经没有下一题可补，即使其他请求尚未完成，也会记录 `refill_exhausted` 与时间并停止提供持续C32加速比；仅发出最后一题而仍有32个活跃请求不会触发它。若全部题目提前完成，还会记录 `dataset_exhausted` 与完全空闲时间，等待至固定窗口结束。观测token计数和吞吐保留，不能把未满负载的窗口称为持续C32速度。

## 数据与GPU证据

专用GPQA校验器核对固定prepared SHA256、448题顺序/长度、Think High、EOS和输出设置、tokenizer/encoder身份，以及实际本地模型metadata。通用prepared-data loader保持原语义。原文件SHA256：

`cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888`

复用95a27da中的完整 `reports/server/c32-prefill-63dac24bf44a`。先核对98个清单文件、全部GPU案例、原回放/core证据和当前native/decode/prefill runtime/image/checkpoint身份；保留历史Python源码身份。未改CUDA，因此默认不重新执行208例GPU检查。当前GPQA服务必须重新提供43层prefill/decode真实执行、各自collective计数和实际B32覆盖。

所有模型、镜像和包均来自服务器已有环境。需要数据准备时，仅从固定作者GitHub archive取得并核验SHA256，然后在已有镜像中离线编码。准备输出使用0664，保留GPUQ继承ACL可读性；私有prepared目录不进入GitHub。

## 入口和产物

`python3 scripts/run_gpqa_c32.py --records <绝对路径> --prefill-evidence-dir <完整公开证据目录> --work-root <绝对路径> --execute`

脚本固定两组、32并发和1200秒，不接受其他时长/并发来悄悄改变实验。默认端口30001，覆盖端口30110–30113。完整操作见 [服务器执行Prompt](SERVER_PROMPT_GPQA_C32.md)。

每组产物包括 `client-summary.json`、`measurement-summary.json`、`gpqa-rows.json`、`load-window.json`、runtime/launch/lifecycle与覆盖记录。总汇总为 `comparison.json`；原GPU证明按原字节保留在 `prefill-proof/`。公开导出按白名单核验，原始日志留在工作盘。

CPU测试和HTTP模拟只验证脚本行为；GPQA结果需等待服务器实际执行。

## CUDA Graph 对照

增加 `--decode-graph full` 可运行两组相同的 full decode graph。该模式先执行四卡 graph smoke，再验证两组实际重放和 Weave 各层设备计数；prefill 保持 eager。完整说明见 [C32 Graph 适配](HOPPER_GRAPH_C32.md)，执行见 [Graph 服务器 Prompt](SERVER_PROMPT_HOPPER_GRAPH_C32.md)。默认 eager 入口与历史结果语义保留。
