# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt

请拉取 `rh2844417786/weave-deepseek-v4-flash` 最新main，记录完整commit，阅读AGENTS.md和docs/NATIVE_P0_PAIR.md，执行原生未采样／P0采样的同32题配对检查，定位此前开启P0后输出明显变短的现象。

## 范围与环境

- 本次仅运行两组：原生SGLang不启用P0，以及原生SGLang启用原配置P0。默认先native再p0；不自动扩展实验矩阵，不修改P0插桩、Marlin、精度或CTA配额。
- 使用已有 `lmsysorg/sglang:v0.5.17`、Torch2.11/CUDA13/Python3.12、只读模型 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`。保留现有缓存。
- 服务器只通过GitHub获取材料，不在线pip/apt/HF下载，不拉镜像。缺依赖或来源不匹配时，保留明确失败，不擅自换版本。
- GPUQ申请4张H100，使用其分配的CUDA_VISIBLE_DEVICES，不固定物理卡号。两组顺序使用同一分配，不能与别的负载混跑。总预算建议6小时，可按需要延长。

## 数据与执行

使用已有完整GPQA Main prepared文件与相邻manifest。main.jsonl SHA256必须为：
`cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888`

脚本固定选择既有seed42文件顺序的前32题。选择SHA256应为：
`a5748d54c4605d57042784ad3e2de569ae017796411239a360f90aaabc9b3018`

不要编辑prepared内容或hash，也不要从旧输出中挑题。替换下面records路径为实际已有文件的绝对路径。

在GPUQ作业内、仓库根目录，用已有Python3.12执行：

```bash
python3 scripts/run_native_p0_pair.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --setup-timeout 7200 --warmup-timeout 960 \
  --request-timeout 3600 --drain-timeout 60 \
  --execute
```

两组均为TP4/EP4、full CUDA Graph精确1..32、关闭padding、max-running-requests32。使用同32题预热，每题最多8token，然后空闲清缓存。正式检查同时发出32题，不补题；Think High已编码进输入，temperature0，正常EOS，每题最多32768token。每组等全部请求自然结束，3600秒只用于超时保护，不是20分钟吞吐窗口；必要时可增大该参数。

复用并核验仓库内上一轮成功的P0事件/CUPTI smoke；不要运行旧Hopper/persistent验收。若P0组短输出导致未达到128次B32 replay，保留0次采样的事实，不改stride、不强行延长生成或循环题目。

## 结果判读与回传

1. 检查两组完整响应、输入/请求体承诺、镜像和启动配置的一致性，以及实际full graph/B32执行。读清理状态，不把残留进程算作完成。
2. 读取comparison.json：逐题输出长度、结束原因、完整文本摘要、可观察前缀摘要、两组长度分布。所有生成内容和长度差异只记录，不恢复数值、token或文本一致率门限。即使32题全部不同，也不得因差异判失败。
3. 核对P0测量前后b32_seen、sampled_total。报告“图插桩运行”和“实际触发采样”两个层次；没有采样不等于没有P0干预。
4. 如果发现差异，报告有多少题、长度变化、结束原因变化、最早可观察的不同前缀，以及是否触发过运行时采样。前缀是累积文本摘要，不是精确token ID位置；不要用单题token序号推断B32 replay序号。
5. 本轮不报告正式吞吐加速比，也不把一次配对差异称为根因已证明。若未复现，明确只覆盖当前32题及实际采样范围。

成功或失败均将脚本打印的export_directory复制到可写clone的reports/server/，校验manifest SHA256，只暂存该新公开目录，commit/push main。禁止git add .；禁止上传原始题目、生成文本、token IDs、rid、路由数组、原始Chrome trace或私有日志。

最后回报推送commit、GPUQ job、实际GPU分配、两组状态、公开目录、长度/结束原因/摘要差异统计、P0是否实际触发采样和清理结果。运行失败时给出经过脱敏的具体原因，保留原始失败证据，不手工改报告状态。
