# DeepSeek-V4-Flash 实验迁移与可行性阶段

状态：用户已决定更换模型为原版 DeepSeek-V4-Flash；尚无 DeepSeek GPU 启动、完整生成或 Weave 核心实现的运行证据。

## 已确定

- 使用用户指定的原版权重，首选 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`；`/mnt/docker_data/models/DeepSeek-V4-Flash` 是用户报告的副本。两份索引的 MD5 相同是用户提供的信息，尚不能代替全部分片完整性检查。
- 使用 `CUDA_VISIBLE_DEVICES=4,5,6,7`，两组 routed-MoE 均以 EP=4 为目标，保持相同专家划分及 attention/dense 并行布局。先验证原生 SGLang 对原版混合精度权重的 EP 支持，再接入 Weave。原生 TP-only 与 Weave EP 的速度差不作为本实验的同拓扑主对照。
- GPQA Main，batch/concurrency=1，Think High，最多生成 32768 tokens，允许正常 EOS；正式对照仅为原生 SGLang 与完整 Weave，每组服务 ready 后最多 1200 秒。
- 用户提供的 `launch_sglang_original.sh` 和 `launch_sglang_dace.sh` 属于旧 Qwen/DACE 环境：两组使用 `/opt/sglang-v4/bin/python`、GPU 4–7、TP=4、context-length=131072、完整 decode CUDA Graph 等；三处差异是模型路径、外部模型注册、MoE runner。它们不能直接当成 DeepSeek-V4-Flash 的已验证启动配置，DACE 不加入当前正式对照。

## 模型与运行时边界

DeepSeek 官方原版配置为 `DeepseekV4ForCausalLM`，hidden size 4096、MoE intermediate size 2048、256 个 routed experts、每 token 选 6 个、43 层，expert dtype 为 FP4；与旧 Qwen 的 BF16/2048/top-8/40 层合同不同。`torch_dtype: bfloat16` 不代表整套权重是 BF16。[原版 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)

用户提供的权重细节：routed experts 是 block 32 的 MXFP4（两个 FP4 值打包一字节，配 FP8 块缩放）；attention 与 shared experts 主要是 FP8，gate/norm/embed 为 BF16，`attn_sink`/`hc_*` 为 FP32。这些是服务器权重的描述；实际副本仍须按 `config.json`、权重索引和分片检查。官方模型卡也把发布版权重描述为 FP4 专家加 FP8 其他主要参数。[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

此前查阅的 SGLang Hopper 配方将原版混合精度版本（MXFP4 routed experts）的路线列为 TP-only；这仅反映该公开配方，不能证明服务器实际版本的所有 EP 路径均不支持。实验目标保持 EP=4，须对锁定版本的 expert runner、dispatch/combine、权重布局和 CUDA Graph 逐项验证；不支持时记录具体缺口。[SGLang DeepSeek-V4 配方](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx)

TP 与 EP 参数可以共存。SGLang 的 EP 文档给出 `--tp 8 --ep 8` 等配置，并说明若干通信后端要求 EP size 与 TP size 相等。因此本项目可验证 `--tp-size 4 --ep-size 4` 这一并行规模组合；具体 MoE runner/a2a 参数须由版本和硬件支持决定。这不是已验证可运行的完整启动命令。[SGLang EP 文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)

旧 `data/gpqa/length_audit.json` 只证明 Qwen tokenizer 下的输入长度。DeepSeek 官方模型卡说明该版本没有 Jinja chat template，而是使用 `encoding/` 中的 `encoding_dsv4`；DeepSeek 的 GPQA prompt、Think High 编码和实际输入长度需要重新审计，不能复用旧 token ID 清单。[官方编码说明](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

## 第一阶段：只读证据与原生可行性

1. 锁定服务器实际 SGLang 源码版本、Python 环境、Docker 镜像 digest 和可用工作盘 `WORK_ROOT`。私有仓库已在 `rh2844417786/weave-deepseek-v4-flash` 创建。
2. 在能读取原版权重的环境运行只读检查：

   ```bash
   CUDA_VISIBLE_DEVICES=4,5,6,7 /opt/sglang-v4/bin/python scripts/preflight_deepseek_v4.py \
     --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
     --output results/deepseek_v4_preflight.json
   ```

   此脚本检查模型架构、`expert_dtype=fp4`（仅指 MXFP4 routed experts）、索引列出的分片是否存在及其总字节数，记录配置/索引 SHA256、Python/包版本和 GPU 清单。它不读取全部分片内容，也不证明模型能生成。
3. 在锁定的同一 SGLang runtime 上，用 GPU 4–7 验证原生 DeepSeek-V4-Flash 的 EP=4 路径，记录完整命令、实际 EP size、每 rank 专家归属、dispatch/combine 后端、MoE/attention 后端、权重显存和 CUDA Graph 状态；运行固定短输入并保存真实输出及错误。仅有服务 ready 或 TP 参数不能证明 EP 已生效。失败时定位并记录具体的格式/内核/通信/内存缺口。
4. 按原生组相同的 EP=4 划分设计 MXFP4 专家计算、top-6 路由、共享专家和 SGLang V4 adapter。先通过数值/路由/多卡/graph 正确性，再做同拓扑对照。若原生 runtime 确需新增 EP 适配，该改动及基线名称必须如实记录，不能称为未经修改的原生支持。当前没有任何这一阶段的 GPU 结果。

## 待外部回传

- 服务器工作盘 `WORK_ROOT`；GitHub 仓库虽已创建，源码推送状态须单独核验。
- 预检 JSON、实际 SGLang/Docker 版本与 EP=4 原生启动及短生成日志，包括实际专家划分与通信后端证据。
