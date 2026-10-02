# Weave / DeepSeek-V4-Flash 实验项目

本项目目标已从 Qwen3.5-35B-A3B 切换为原版 DeepSeek-V4-Flash 混合精度权重。GPQA 官方数据已下载，并用锁定的 DeepSeek tokenizer/编码器完成本地输入长度审计。DeepSeek 的 GPU 可行性、CUDA 实现、Docker 构建与 GPU 实验尚未验证。

## 入口

- [当前服务器镜像、版本与工作目录契约](docs/RUNTIME_0517.md)
- [CUDA 功能核心、动态调度与 SGLang adapter](docs/CUDA_FUNCTIONAL_CORE.md)
- [历史 SGLang 0.5.16 接入核对](docs/SGLANG_V0516_INTEGRATION.md)
- [EP 路由参考当前编码计划](docs/superpowers/plans/2026-09-30-deepseek-routing-reference.md)
- [DeepSeek-V4-Flash 迁移与当前决策](docs/DEEPSEEK_V4_FLASH_MIGRATION.md)
- [需求与架构设计](docs/superpowers/specs/2026-09-29-weave-qwen35-design.md)
- [详细编码计划](docs/superpowers/plans/2026-09-29-weave-qwen35.md)
- [GPT-6 Sol 交接](docs/HANDOFF_GPT6_SOL.md)
- [GPQA 数据、长度与复核方法](data/gpqa/README.md)
- [DeepSeek GPQA 输入长度审计](data/gpqa/deepseek_length_audit.json)

当前目标：原版 DeepSeek-V4-Flash 混合精度权重，在 GPU 4–7 上以 EP=4 比较原生 SGLang 与完整 Weave。两组采用相同的专家划分和 attention/dense 并行布局；原生运行时对 MXFP4 专家 EP 路径的支持须先验证。SGLang 的 TP 参数可与 EP 参数同时使用，不能把 `--tp-size 4` 等同于 MoE 只使用 TP。GPQA Main、batch=1、Think High、最多生成 32K（32768）tokens，每组服务 ready 后预热与测量共最多20分钟。旧 Qwen 设计、计划和交接文件保留为历史记录，不能直接作为 DeepSeek 实现合同。

已有代码包括：只读预检、DeepSeek GPQA token ID 准备、decode 计数、EP 路由去重/合并 CPU 参考，以及原生 EP4 Docker 启动入口。现已加入 CUDA IPC persistent 功能核心、GPU 端 c/K 选择和连续 stealing、raw MXFP4 专家加载器，以及源码核对后的 SGLang 0.5.17 eager adapter。默认使用当前服务器已有的 `lmsysorg/sglang:v0.5.17`，工作目录为 `<repo>/work`。后端名为 `weave_functional`：H100 上的 CUDA 编译、两轮四卡小张量功能与 core graph replay 已通过并核对原始报告；当前计算单元仍是标量 GEMV，真实模型尺寸、非零 stealing、Tensor Core tile 优化、SGLang graph 与性能实验待验收。证据见 `reports/core_acceptance.json`，状态见 `reports/implementation_status.json`。

旧 Qwen 输入长度审计的本地复核（不适用于 DeepSeek）：

```bash
python3 scripts/fetch_gpqa.py --verify-only
.venv-data/bin/python scripts/audit_gpqa_lengths.py
```

新服务器/新 checkout 的 GPQA 数据下载与旧 Qwen 审计命令（DeepSeek 需另用其编码器重新审计）：

```bash
python3 scripts/fetch_gpqa.py
python3 -m pip install -r requirements-data.txt
python3 scripts/audit_gpqa_lengths.py --tokenizer-dir /mnt/public_data/Qwen/Qwen3.5-35B-A3B
```

DeepSeek Think High 输入审计覆盖 GPQA Main 448 题：中位数 191、最长 2550 tokens，无 8K prefill 样本。用 `python3 scripts/prepare_gpqa.py --model-dir <DeepSeek模型目录>` 生成两组共用的 token ID 清单；清单位于 Git 忽略目录。上述数字是 CPU 编码结果，不是 GPU 生成或速度结果。原始数据与生成文本不加入 Git；同步来源和下载脚本。[私有 GitHub 仓库](https://github.com/rh2844417786/weave-deepseek-v4-flash) 已创建，DeepSeek 实际 SGLang EP 启动配置尚待核验。
