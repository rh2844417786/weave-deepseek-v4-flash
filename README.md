# Weave / DeepSeek-V4-Flash 实验项目

本项目目标已从 Qwen3.5-35B-A3B 切换为原版 DeepSeek-V4-Flash 混合精度权重。GPQA 官方数据已下载；现有 tokenizer 长度审计只适用于旧 Qwen 模型。DeepSeek 的 GPU 可行性、CUDA 实现、Docker 构建与 GPU 实验尚未验证。

## 入口

- [DeepSeek-V4-Flash 迁移与当前决策](docs/DEEPSEEK_V4_FLASH_MIGRATION.md)
- [需求与架构设计](docs/superpowers/specs/2026-09-29-weave-qwen35-design.md)
- [详细编码计划](docs/superpowers/plans/2026-09-29-weave-qwen35.md)
- [GPT-6 Sol 交接](docs/HANDOFF_GPT6_SOL.md)
- [GPQA 数据、长度与复核方法](data/gpqa/README.md)

当前目标：原版 DeepSeek-V4-Flash，先在 GPU 4–7 上验证 TP=4 原生 SGLang 可运行性，再开发 MXFP4 routed-expert 的 Weave 跨卡 EP。正式对照如何处理 TP 与 EP 的并行方式差异仍待确定。GPQA Main、batch=1、Think High、最多生成 32K（32768）tokens，每组服务 ready 后预热与测量共最多20分钟。旧 Qwen 设计、计划和交接文件保留为历史记录，不能直接作为 DeepSeek 实现合同。

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

Main448题在旧 Qwen tokenizer 下的输入中位数213、最长2780tokens，无8K prefill样本。DeepSeek 的输入长度需要用其编码器重审；上述数字不是 GPU 生成结果。原始数据与生成文本不加入Git；同步来源和下载脚本。[私有 GitHub 仓库](https://github.com/rh2844417786/weave-deepseek-v4-flash) 已创建，DeepSeek 实际 SGLang 启动配置尚待采集。
