# Hopper FP8 接入 EP4 decode 与 C32 服务对照

## 已有证据与本轮目标

服务器提交 `dbe03bd` 已完成 `reports/server/hopper-candidate-115da040aca7` 的全部 1376 点：4 rank × 43 层 × 2 个真实 B32 样本 × 4 种计算模式。FP8 相对 BF16 的配对时间比几何均值为 eager 2.749、graph 2.768；这是本地专家计算回放，不包括通信和完整服务。

本轮把同一 FP8 候选接到实际 SGLang decode。三组均为 TP4/EP4、eager、C32，依次运行 native、BF16 staged_direct、hopper_fp8。需要服务器测量后才能判断完整模型 token 吞吐是否提升。

## 服务路径

`--decode-implementation hopper_fp8` 在 B1..32 decode 使用原始 MXFP4/E8M0 专家权重与 FP8 激活候选：

1. `FP8Workspace.run` 对完整复制的输入执行本 rank 专家计算，写入 FP32 routed slots。
2. 在现有 TP device group 上对活跃 slot 前缀执行一次 NCCL SUM。
3. 独立 Triton finalizer 按 slot 0..5 顺序执行 FP32 `add.rn.f32`，一次 BF16 RNE 转换，并只保留本 rank 连续 origin 行。
4. 沿用 SGLang shared expert、缩放与外层 TP 汇总。

每层绑定一次原始权重；工作区与 B1..32 输出视图预分配。Prefill 仍使用缓存后的 persistent 路径。禁止 graph、额外诊断 profile 和 native 专家回退。BF16 staged 组采用同样的 slot 汇总布局；通信和外围开销都进入服务计时。

本轮未修改已验证的原生候选或 BF16 CUDA：

- candidate: `bcabd0f4d1ea9430c480f4637940b13dcb8becd13bb016fdf6fda658e3d3b2ce`
- BF16: `25b51a34436de7494a80d739afb03ad0065f63639e4208b784270adf3632498f`

新 finalizer/runtime 有独立源码身份；服务 manifest 绑定 adapter、launcher、测试、配置。旧回放的 Python 身份保留为历史值；其原生源码、模型、镜像与 tokenizer 必须匹配当前实验。

## 验证与统计口径

统一入口 `scripts/run_c32_comparison.py --comparison-suite hopper`：

- 读取完整 schema2 回放包，验证 29 个文件的 SHA256，重算 1376 点汇总，验证原始 core/checkpoint 证据。
- 先执行四 rank 组装 smoke：B1/7/17/32/7/1、空专家 rank、空 origin rank、缓存地址、绑定生命周期、每调用一次 collective，以及独立顺序求和 fixture。它使用真实候选/runtime，不以 Torch 计算替代候选。
- 顺序启动三组服务，每组先完成 32 请求 × 8 token 响应检查，然后执行相同 C32、1024 token 上限、600 秒流式测量窗口。
- 输出数值误差和生成 token 一致率均只记录。内容差异不停止吞吐测量；无效响应、缺失请求、身份错误、非有限输出、运行时错误、源码不一致和覆盖缺失仍失败。
- 覆盖证据必须包含全部 43 层实际 decode 路径、B32 histogram 和每次自研 decode 恰好一次 slot collective。Hopper 还必须匹配 candidate/assembly 身份。

吞吐来自 `client-summary.json.decode_tokens_per_second`：窗口内流式 token 增量，包括未完成请求，排除首 token。汇总给出 Hopper/native 和 Hopper/BF16 staged 比值。此 custom 数据实验不代表 GPQA 准确率，也不代表 fullgraph 性能。

旧默认 staged 套件的历史严格比较保持原语义；执行本轮必须显式选择 `--comparison-suite hopper`。

## 运行条件与输出

沿用已有 `lmsysorg/sglang:v0.5.17` 和原始只读本地模型，不下载依赖或镜像。Triton 来自已有镜像。四卡编号由 GPUQ 提供，不硬编码。完整回放包已在 GitHub，默认使用包内 core，不需再跑计算回放。

每次运行创建新目录；父进程管理容器和清理。公开导出包含来源证明、四卡 smoke、覆盖、客户端统计、三组汇总和生命周期；原始输入、token ID、生成文本、trace、完整日志留在服务器。

具体命令见 [服务器执行 Prompt](SERVER_PROMPT_HOPPER_C32.md)。本机只完成 CPU 检查与源码审阅；新增 Triton 编译、四卡 smoke 与 C32 服务结果待服务器回传。
