# Compact Hopper FP8 prefill

## 改动依据

5629aaf 的诊断已定位到 persistent prefill 的 routed-expert 执行区间。8035 行合批 prefill 的完整 GPU stream 时间约37.8秒，其中约99.5%落在专家区间；约68.5秒的最长客户端输出间隔与 prefill 几乎完全重叠。现有 Hopper decode 有效，但只支持B1..32。

本轮新增独立的大行数 prefill 路径。已有 decode CUDA、BF16 CUDA和decode组装不改。旧结果保留，新的 GPU结果需由服务器验证。

## 新实现

新扩展 `experimental/hopper_prefill` 支持1..8192行，使用原始MXFP4/E8M0专家权重和既有FP8激活数值约定：

1. GPU验证原始宽度的路由ID、每token无重复专家、有限routing。
2. 每个专家的CTA把token按稳定顺序分组，生成连续slot表。
3. GPU为每个专家生成多个64行tile，所有专家的tile放入紧凑缓冲区。
4. TMA加载tile-major激活和expert-major权重，WGMMA执行gate/up/down；缩放、SwiGLU与BF16边界沿用已有候选的顺序。
5. 写出本rank的FP32 routed slots；在现有TP device group做一次SUM。
6. 按top-k顺序累加，转换成BF16，保留本rank的origin行，再交给原有SGLang shared/scaling/外层TP汇总。

正式服务没有32行分块回退、CPU路由或读取GPU active count。工作区在安装时一次分配，原始权重逐层绑定并复用。分配容量为 `ceil(max_batch*topk/64)+local_experts` 个tile，实际launch上界随本次batch收紧。

原始模型、EP4、maxB8192下为832个tile，新增工作区张量约2,369,723,140字节，即2.207GiB/卡。这是张量分配估算，不含CUDA上下文、NCCL、编译器和allocator缓存；以服务器显存证据为准。单次prefill超过8192行会明确报错，无静默回退。

## 接入与对照

显式参数：`--prefill-implementation hopper_fp8`。要求 `--decode-implementation hopper_fp8`、tiled、hot-cache、fixed(1,1,0)、eager。默认仍为persistent，历史入口语义保持。

新入口 `scripts/run_prefill_comparison.py` 依次跑：

| 组名 | Prefill | Decode |
|---|---|---|
| native | SGLang原生 | SGLang原生 |
| hopper_fp8 | 旧persistent | 既有Hopper FP8 |
| hopper_prefill | 新多tile Hopper FP8 | 同一既有Hopper FP8 |

三组均为原custom数据、EP4、C32、eager、1024输出上限、每组600秒。这样两自研组的区别集中在prefill；slot SUM和外层汇总开销仍计入服务时间。输出prefill-vs-native和prefill-vs-old-prefill吞吐比值，同时检查TTFT和输出间隔。

`--stall-diagnostic` 可按相同矩阵重用已有计时，但带观测的结果会标记非正式性能并移除加速比；默认运行不启用诊断。

## 验证

复用完整1376点decode回放/core证据，检查旧原生源码身份未变。新增GPU检查每rank52例：7个合成batch/空rank/偏斜/缩小场景；43层真实权重B129；真实layer0的B8035和B8192。检查真实元数据、有限输出、缓存地址、绑定生命周期、一次prefill slot SUM和确定性输出组装。

数值对照包括所有行的旧FP8实现（仅验证脚本按<=32行调用），以及两个token位置的独立FP8/BF16 oracle和BF16实现。两token oracle属于有限抽样，不能声称覆盖所有token的独立oracle。所有有限误差只记录，无relative L2/绝对误差/allclose门限。生成token内容差异也只记录。

服务覆盖必须证明全部43层实际执行 `hopper_prefill`，且prefill与decode的调用/额外collective分别准确对应。无新路径覆盖、身份漂移、非有限值或运行错误会失败并保留公开诊断。

本机验证限于CPU测试、可移植C++布局契约和源码审阅。新增CUDA编译/TMA/WGMMA运行、模型可用性和性能仍待服务器回传。

执行见 [服务器Prompt](SERVER_PROMPT_HOPPER_PREFILL.md)。服务器继续使用已有镜像、本地只读模型和GitHub；保留已有编译/数据缓存，不在线安装依赖或拉镜像。
