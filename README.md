# Weave / DeepSeek-V4-Flash 实验项目

本项目目标已从 Qwen3.5-35B-A3B 切换为原版 DeepSeek-V4-Flash 混合精度权重。GPQA 官方数据已下载，并用锁定的 DeepSeek tokenizer/编码器完成本地输入长度审计。DeepSeek 的 GPU 可行性、CUDA 实现、Docker 构建与 GPU 实验尚未验证。

## 入口

- [DeepSeek-V4-Flash 迁移与当前决策](docs/DEEPSEEK_V4_FLASH_MIGRATION.md)
- [需求与架构设计](docs/superpowers/specs/2026-09-29-weave-qwen35-design.md)
- [详细编码计划](docs/superpowers/plans/2026-09-29-weave-qwen35.md)
- [GPT-6 Sol 交接](docs/HANDOFF_GPT6_SOL.md)
- [GPQA 数据、长度与复核方法](data/gpqa/README.md)
- [DeepSeek GPQA 输入长度审计](data/gpqa/deepseek_length_audit.json)

当前目标：原版 DeepSeek-V4-Flash 混合精度权重，在 GPU 4–7 上以 EP=4 比较原生 SGLang 与完整 Weave。两组采用相同的专家划分和 attention/dense 并行布局；原生运行时对 MXFP4 专家 EP 路径的支持须先验证。SGLang 的 TP 参数可与 EP 参数同时使用，不能把 `--tp-size 4` 等同于 MoE 只使用 TP。GPQA Main、batch=1、Think High、最多生成 32K（32768）tokens，每组服务 ready 后预热与测量共最多20分钟。旧 Qwen 设计、计划和交接文件保留为历史记录，不能直接作为 DeepSeek 实现合同。

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
