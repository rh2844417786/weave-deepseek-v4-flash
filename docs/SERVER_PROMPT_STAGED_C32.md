# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt

请拉取rh2844417786/weave-deepseek-v4-flash的main并记录commit，阅读AGENTS.md和docs/STAGED_C32_SERVICE.md，执行本轮staged-direct真实decode与C32对照。按下面顺序完成并回传证据，不自行增加实验矩阵。

约束：仅使用已有lmsysorg/sglang:v0.5.17、原始本地混合精度模型/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash和已安装包。服务器只从GitHub拉取代码/回传材料；禁止在线pip/apt/HF下载或拉镜像。保留GPUQ给出的四卡CUDA_VISIBLE_DEVICES，勿硬编码编号。用户允许根据需要延长GPUQ运行时间，须给各阶段和cleanup留够时间。

1. 提交新源码core+全部43层checkpoint检查。旧P0/P1的72592ee源码gate不能复用，无需成本模型扫描。取得分配后执行：

```bash
git pull --ff-only origin main
export WORK_ROOT=/srv/weave-work
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
export CHECKPOINT_TESTS=1
export CHECKPOINT_LAYERS=all
export CORE_RUN_NAME="cuda-functional.staged-c32-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
bash scripts/run_cuda_functional_gpuq.sh
export STAGED_C32_CORE="$WORK_ROOT/$CORE_RUN_NAME"
```

2. 确认原有custom数据文件存在。已知路径/srv/weave-work/bench-concurrency32-dataset.jsonl，SHA256必须是7d7bcdb9996f9293b63d337f21ee977f39013107ef23c2c411de2310d64501c2。可定位同内容的新路径，不得重新生成、填充输入或替换数据。

3. 运行统一入口。它自动先做四卡online-staged数值gate，再顺序做native eager、persistent-cached eager、staged-direct eager三组。每组先32请求×8token验收，再同样的C32/1024token/600秒流式窗口：

```bash
python3 scripts/run_c32_comparison.py \
  --work-root /srv/weave-work \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --dataset /srv/weave-work/bench-concurrency32-dataset.jsonl \
  --core-run-dir "$STAGED_C32_CORE" \
  --seconds 600 \
  --setup-timeout 7200 --gate-timeout 7200 \
  --acceptance-timeout 1200 --drain-timeout 600 \
  --execute
```

可按GPUQ政策延长setup/gate/作业总预算。不要仅因旧的一小时限制杀掉正在正常执行的作业。不要把native切成fullgraph；三组必须同为eager。保留staged的在线分组、slot all-reduce和原外层TP汇总，它们是本次真实服务计时的一部分。

任一gate/验收/组失败就停止后续依赖步骤。不要降低数值门限、删失败用例、绕过实际token验收、切回native或复用旧源码证据。Logprob数值仅记录，不作为门限。若失败，读取工作目录私有日志定位具体错误，汇报阶段、错误和容器清理情况；不要上传原始请求内容。

完成时应看到comparison_complete。核对online gate每rank49项（6合成+43层B32），三组验收各32×8token，全部43层的staged/persistent执行计数与B32证据。吞吐以client-summary.json的decode_tokens_per_second为准：窗口内流式累计token增量，包括未完成请求并排除首token；不能拿完成请求token数量或scheduler日志TPS替代。

将程序打印的export_directory复制到可写clone的reports/server/下，核对manifest.json每个SHA256，只暂存该公开目录，commit并push main。禁止git add .；不得上传acceptance-private.json、raw trace、prompt、token IDs、生成文本或完整私有日志。

最后汇报：commit、GPUQ job、物理GPU、runtime/source/checkpoint身份，core与online gate结果，三组验收、窗口时长/完成失败未完成数/实际B32证据、三组decode TPS与两个比值、TTFT/可用ITL、容器清理状态。声明这是原custom负载下的C32 eager服务结果，不是正式GPQA准确率或fullgraph结果。若未完成则明确写未完成，不计算有效加速比。
