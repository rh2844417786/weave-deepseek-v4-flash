# GPQA 本地数据与输入长度审计

> 下方旧表格使用 Qwen3.5 tokenizer/template。模型目标已改为 DeepSeek-V4-Flash；GPQA 原始数据可复用，DeepSeek 的输入长度审计见 [deepseek_length_audit.json](deepseek_length_audit.json)。

DeepSeek Think High 的本地 CPU 审计使用模型修订 `60d8d70770c6776ff598c94bb586a859a38244f1` 的 `encoding_dsv4.py` 与 `tokenizer.json`，保留自然题目、原始行号决定选项、seed 42 决定题序。Main 448 题输入长度最短 64、中位数 191、P95 394.65、最长 2550 tokens，无 8K 输入。服务器如使用不同 tokenizer 或编码器，正式输入必须重新审计。准备脚本不打印题目内容，生成的 token ID 清单位于 Git 忽略的 `data/gpqa/prepared/`。

```bash
python3 scripts/prepare_gpqa.py --model-dir /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
```

已从作者官方 GitHub 仓库下载原始加密 ZIP，并按作者公开的密码解压。当前执行状态是数据下载与 CPU tokenizer 审计完成；没有进行模型推理或 GPU 性能实验。

## 数据位置

- `raw/gpqa_main.csv`：448 题，本项目首轮正式测速默认使用这一版本。
- `raw/gpqa_diamond.csv`：198 题，已下载，尚未加入实验矩阵。
- `raw/gpqa_extended.csv`：546 题，已下载，尚未加入实验矩阵。
- `source/dataset.zip`：官方原始归档，SHA256 为 `461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc`。
- `source_manifest.json`：源版本、文件散列与行数。
- `length_audit.json`：实际输入 token 长度审计结果。

Diamond 是 Main 的子集；不能把两者简单相加当作不同问题使用。CSV 中的专家身份、验证反馈、Explanation、正确选项标签及其他评审字段不进入模型 prompt。作为四个候选选项之一的正确答案文本正常进入 prompt，但不会标示哪个选项是正确答案。

## 审计方法与结果

来源版本：GPQA `56686c06f5e19865c153de0fdb11be3890014df7`；Qwen tokenizer `59d61f3ce65a6d9863b86d2e96597125219dc754`。

输入是一个 user message：简短解题说明、Question 列、四个确定性打乱的选项。使用官方 `chat_template.jinja`、`add_generation_prompt=True`、`enable_thinking=True`，随后用官方 `tokenizer.json` 编码，`add_special_tokens=False`。没有 system message、few-shot 示例、拼接、padding 或截断。以下结果仅适用于这一明确模板；服务器须核对 tokenizer/template 散列，改变 prompt 后重新审计。

| 版本 | 题数 | 最短 | 中位数 | P95 | 最长 | 至少 2048 | 至少 4096 | 至少 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Main | 448 | 74 | 213 | 442.95 | 2780 | 1 | 0 | 0 |
| Diamond | 198 | 77 | 218.5 | 460 | 2780 | 1 | 0 | 0 |
| Extended | 546 | 74 | 216 | 463 | 12323 | 2 | 1 | 1 |

Main/Diamond 可以提供真实短输入的生成测速工作负载，不能提供 8K prefill 样本。Extended 只有一个自然长度超过 8K 的样本，不能支撑稳定的 8K 工作负载集合。最多生成 32K（32768）tokens 是输出限制，不会让短问题成为 8K prefill。首轮保留自然题目长度；不自动重复题目、补无意义文本或扩大数据集实验矩阵。

## 复核与服务器准备

在项目根目录执行：

```bash
python3 scripts/fetch_gpqa.py
python3 scripts/fetch_gpqa.py --verify-only
python3 -m pip install -r requirements-data.txt
python3 scripts/audit_gpqa_lengths.py --tokenizer-dir /mnt/public_data/Qwen/Qwen3.5-35B-A3B
```

本地已配置的审计环境可直接使用 `.venv-data/bin/python scripts/audit_gpqa_lengths.py`。

正式输入统一从 Main 生成，按 seed 42 固定题序；选项顺序使用 `42 + 原始 CSV 行号` 固定。两组复用同一份已生成的 prompt/token ID 清单；共享测试内容不会在 Git 中明文发布。

## 分发与来源

作者的数据卡将数据标注为 CC-BY-4.0；GitHub 代码仓库另附 MIT LICENSE，二者不能混为一个授权声明。保留原始来源与数据集引用。作者请求避免将测试题明文公开，以减少训练污染，因此原始 CSV、ZIP、准备后的 prompt 和逐题生成正文位于 `.gitignore` 中。GitHub 同步代码、元数据、校验值及下载脚本；服务器 `git pull` 后运行下载脚本得到相同版本。

- [官方仓库与归档下载说明](https://github.com/idavidrein/gpqa/tree/56686c06f5e19865c153de0fdb11be3890014df7)
- [作者 Hugging Face 数据卡](https://huggingface.co/datasets/Idavidrein/gpqa/blob/main/README.md)
- [GPQA 论文](https://arxiv.org/abs/2311.12022)
- [官方 Qwen3.5-35B-A3B tokenizer](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754)
