# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt

请拉取 `rh2844417786/weave-deepseek-v4-flash` main并记录commit，阅读AGENTS.md与docs/GPQA_C32_COMPARISON.md，执行GPQA Main两组对照：Native SGLang与当前Weave Hopper FP8 prefill+decode。两组均32并发，正式测量各1200秒。

## 约束

- 使用已有 `lmsysorg/sglang:v0.5.17`、Torch2.11/CUDA13/Python3.12及原始只读模型 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`。
- 服务器只通过GitHub取材料；不在线pip/apt/HF下载，不拉镜像。保留现有编译缓存和数据缓存。
- GPUQ取得4张H100后执行，保留分配的CUDA_VISIBLE_DEVICES，不硬编码物理卡号。
- 两组均TP4/EP4/eager、max-running-requests32。GPQA自然输入、Think High、temperature0、最多32768输出token、允许EOS。不能改成强制1024token或重复题目填满时间。
- 每组正式窗口1200秒，加载、检查、32请求×最多8token预热及清理在窗口外。首次GPUQ总运行预算建议6小时，可按需要延长。
- 所有数值误差和生成内容差异只记录；不恢复精度/一致率门限。该任务不评估GPQA准确率。

## 数据准备

优先使用已有prepared目录，目标main.jsonl SHA256必须为：
`cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888`

它的相邻manifest.json必须完整且匹配实际本地模型metadata。此前默认路径是仓库 `data/gpqa/prepared/deepseek-v4-flash/main.jsonl`。可使用同一内容的其他路径，必须给绝对路径。不要手工改manifest或hash以绕过检查。

如没有可用prepared文件，先在GPUQ外用主机Python从作者固定GitHub来源取得数据；既有文件会先验SHA：

```bash
export WORK_ROOT=/srv/weave-work
python3 scripts/fetch_gpqa.py --data-root "$WORK_ROOT/gpqa-source"
```

随后使用已有镜像离线准备；不需要GPU：

```bash
REPO_ROOT="$(pwd -P)"
docker run --rm --pull=never --network=none \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -v "$REPO_ROOT:/workspace:ro" \
  -v /srv/weave-work:/work \
  -v /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash:/model:ro \
  -w /workspace lmsysorg/sglang:v0.5.17 \
  python3 /workspace/scripts/prepare_gpqa.py \
  --csv /work/gpqa-source/raw/gpqa_main.csv \
  --source-manifest /workspace/data/gpqa/source_manifest.json \
  --model-manifest /workspace/data/gpqa/deepseek_model_manifest.json \
  --model-dir /model --output-dir /work/gpqa-main-c32
```

准备输出应为0664，GPUQ执行身份可读。不要将prepared token IDs或题目加入GitHub。数据已有效则无需重做。

## 作业执行

复用仓库中完整的95a27da公开证据 `reports/server/c32-prefill-63dac24bf44a`。脚本会验证原98个清单文件、GPU结果和当前模型/原生内核身份，默认不重跑未改动的CUDA检查。

获得GPUQ分配后，从仓库根目录、用已有Python3.12执行：

```bash
REPO_ROOT="$(pwd -P)"
python3 scripts/run_gpqa_c32.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --prefill-evidence-dir "$REPO_ROOT/reports/server/c32-prefill-63dac24bf44a" \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --setup-timeout 7200 --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

若复用了已有prepared文件，替换`--records`为它的绝对路径。不要调用旧benchmark_gpqa.py或custom C32入口代替本轮，也不要开启stall profiler或graph。两组顺序执行，用同一组分配的卡。

到1200秒脚本自动停止发新请求、取消和中止其拥有的未完成请求。不要另起请求干扰窗口。若worker完成后已无新题可补，保留refill_exhausted标记，即使其余请求仍运行也不输出持续C32加速比。若448题提前全部完成，还保留dataset_exhausted和空闲时间；不能循环题目。

## 汇报与推送

核对两组窗口均1200秒、客户端峰值并发32，以及Native scheduler/Weave四rank覆盖中的实际B32。Weave必须全部43层执行Hopper prefill+decode，两个collective计数分别核对。

报告两组decode tokens/s、TTFT/可用ITL、窗口token数、完成/未完成/未尝试题数、共同完成行数、是否数据耗尽、运行与清理状态。吞吐分子包含未完成请求在窗口内已输出的token，每请求排除首token；不能用仅完成题的token或每请求耗时之和代替总窗口吞吐。未完成全448题要明确说明，不把速度结果称为准确率。

成功或失败均保留证据。将脚本打印的export_directory复制到可写clone的reports/server/，核对manifest的SHA256，只暂存该公开目录并commit/push main。禁止git add .；不得上传prepared文件、题目、生成文本、token IDs或完整私有日志。最后回报推送commit、GPUQ job、卡号、状态与报告目录。
