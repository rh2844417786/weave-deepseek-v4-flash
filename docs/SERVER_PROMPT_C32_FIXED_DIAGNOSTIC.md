# 给服务器 DeepSeek V4 Flash 的执行 Prompt

请在当前 Weave 项目中执行一次 C32 固定计划 eager 阶段诊断，并把公开证据提交回GitHub。请先阅读仓库AGENTS.md与`docs/C32_FIXED_EAGER_DIAGNOSTIC.md`。

本次目的是分析已有约46.43 token/s的`tiled + eager + fixed(1,1,0)`路径。只运行这一组诊断，不增加原生组、模型、数据集、校准扫描，不改变kernel算术或调度参数。输出是诊断证据，不宣称新的加速结果。

1. 在现有可写Git clone中执行`git pull --ff-only origin main`，记录commit。确认存在`scripts/run_concurrency_diagnostic.py`。
2. 从上一轮`job-20261014-020203-8dae7a66`的负载命令/作业日志定位实际使用的custom JSONL，设置绝对路径变量`C32_DATASET`。该文件是九条SWE-Bench输入各重复200次，共1800行；不是只有9行的prepared token-ID文件。使用原文件，不重新生成、改prompt或下载数据。仅报告路径、记录数和SHA256，勿把文本复制到公开仓库。
3. 复用已存在的`/srv/weave-work/cuda-functional.2349166e7413`。若不存在，定位另一份同CUDA源码身份95e8d6c87ab67514c2b090834d7cc43ec352acfb24f088bb749f8482e5a9d7c0、原始checkpoint全部43层通过的core目录。入口会核对身份。无需重跑374点校准，也不要提供measured profile。
4. 使用现有GPUQ方式申请四张H100，给作业预留模型启动、600秒负载和最多120秒drain时间；租期至少一小时，排队时间不计入运行时间。尊重GPUQ授予的`CUDA_VISIBLE_DEVICES`，不要硬编码或覆盖物理编号。
5. 在GPUQ实际授予四卡的作业内执行以下命令，将已核实的custom数据绝对路径放入环境变量`C32_DATASET`：

```bash
python3 scripts/run_concurrency_diagnostic.py \
  --work-root /srv/weave-work \
  --core-run-dir /srv/weave-work/cuda-functional.2349166e7413 \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --dataset "$C32_DATASET" \
  --export-root /srv/weave-work/public-reports \
  --seconds 600 \
  --stride 16 \
  --capacity 8 \
  --execute
```

脚本会创建全新目录，运行四GPU event smoke，显式启动`tiled/eager/fixed(1,1,0)`，在同一现有镜像中运行本地tokenizer的SGLang benchmark，并在warmup结束后清零探针。它会负责窗口截止、停止本次负载、abort本次服务请求、idle采集、容器清理及公开导出。不要自行再启动第二个负载，也不要在活跃请求期间手动flush。

服务器只能通过GitHub获取材料。使用现有`lmsysorg/sglang:v0.5.17`与本地只读checkpoint，不运行pip/apt/HF下载，不拉取镜像。若本地benchmark源码与锁定版本不匹配、数据文件或core缺失，保留错误与已有证据并汇报，不自动安装、替换或降低校验。

运行结束后，检查`experiment.json`是否为`diagnostic_complete`，四rank是否均有至少两个valid_rows=32且与43层对齐的采样。若失败，同样回传已有公开导出，说明阶段和具体错误；不把失败或不足的样本改写成成功。

只将程序打印的`public-reports/c32-diagnostic-<id>/`复制到可写clone的`reports/server/c32-diagnostic-<id>/`，核对manifest全部SHA256后，仅提交该公开目录并推送main。不要提交工作盘原始日志、benchmark-private文件、prompt、token IDs或生成内容；不要使用`git add .`。

请最后汇报：运行commit、GPUQ job ID、实际四卡分配、fixed plan/graph/kernel；请求started/succeeded/failed和截断情况；四rank的实际decode batch histogram及batch32调用占比；C32对齐样本中的完整forward GPU时间、host enqueue时间、43层persistent span与剩余时间；最慢层的compute work/queue、input wait、return work/wait、end wait和group fill；清理状态、公开证据路径及推送commit。

解释时注意：c=1表示1个通信CTA和131个可计算CTA；capacity2048不是KV上下文容量；CTA累计时间和重叠envelope不能直接相加当latency；完整forward减persistent的差值包含其他GPU工作，不能直接称为CPU时间。带探针结果不与原生1970 token/s计算正式加速比。
