# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt

请拉取 `rh2844417786/weave-deepseek-v4-flash` 的 main，记录实际commit，阅读AGENTS.md与docs/HOPPER_PREFILL.md，验证新的compact多tile Hopper prefill并执行三组C32服务对照。不要自行改实验矩阵、调度参数或数值策略。

## 约束

- 使用现有 `lmsysorg/sglang:v0.5.17`（Torch2.11/CUDA13/Python3.12）、已安装包和原始只读 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`。
- 服务器仅GitHub同步；禁止在线pip/apt/HF下载、拉取镜像。缺失依赖需报告，不下载替代。
- GPUQ分配四张H100后执行，保留CUDA_VISIBLE_DEVICES，不硬编码物理卡号。
- 保留 `/srv/weave-work/torch_extensions` 与 `/srv/weave-work/cache`。新增prefill扩展具有独立编译身份，不删除旧decode缓存。
- 新工作区约2.207GiB/卡，服务仍保持既定mem-fraction-static0.80。若OOM，保留具体阶段和显存信息回传，不自行改变三组配置或切回旧prefill。
- 原custom数据 `/srv/weave-work/bench-concurrency32-dataset.jsonl` SHA256为 `7d7bcdb9996f9293b63d337f21ee977f39013107ef23c2c411de2310d64501c2`，不得改内容、填充或重排。
- 使用仓库完整证据 `reports/server/hopper-candidate-115da040aca7`，不重跑1376点decode回放。
- 所有数值误差和生成内容差异只记录，不恢复任何容差或一致率门限。非有限输出、结构错误、身份/覆盖缺失和运行错误仍需停止。

GPUQ首次总预算建议8小时，可按需延长，给编译、模型哈希、208个rank-case检查、三次加载和600秒/组测量留足时间。用户已允许根据需求延长。

## 执行

先安全拉取main，不覆盖本地未提交内容。确认驱动使用Python3.12（和镜像契约一致；不要用Python3.9运行历史浮点汇总的精确复核）。取得GPUQ四卡分配后，从仓库根目录运行：

```bash
export WORK_ROOT=/srv/weave-work
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
REPO_ROOT="$(pwd -P)"
python3 scripts/run_prefill_comparison.py \
  --work-root "$WORK_ROOT" \
  --model-path "$MODEL_PATH" \
  --hopper-evidence-dir "$REPO_ROOT/reports/server/hopper-candidate-115da040aca7" \
  --dataset /srv/weave-work/bench-concurrency32-dataset.jsonl \
  --seconds 600 \
  --setup-timeout 7200 --gate-timeout 14400 \
  --acceptance-timeout 1200 --drain-timeout 600 \
  --execute
```

如果主机`python3`不是3.12，使用已经存在的Python3.12可执行文件；不要在线安装。默认本轮不要加`--stall-diagnostic`，先取得未加观测的实际吞吐与TTFT。

统一入口依次：验证原证据 → 既有Hopper组装检查 → 新prefill四rank52例/卡检查 → Native、旧persistent prefill+Hopper decode、新Hopper prefill+同一Hopper decode三组。两自研组均为EP4/eager/C32。新prefill行数上限8192，保持chunked-prefill-size8192。

GPU检查中的旧FP8分块只用于数值参考，不是服务实现。误差较大也继续记录和测量；不要删用例或添加阈值。任一编译、运行、结构、源码身份或覆盖检查失败时保留失败目录和私有日志，回传已有公开证据，不伪装完成。需要修复后重跑时使用新目录。

## 回传

确认experiment.json为comparison_complete、三组客户端均完成600秒窗口。读取并汇报：

1. 新prefill CUDA是否编译，四rank各52例是否完整，有限数值差异记录范围，工作区资源/显存情况。
2. 两自研组全部43层的prefill/decode实际路径；新组每次prefill只有一个额外slot SUM，decode计数独立核对。
3. 三组client-observed decode TPS，prefill相对旧prefill和Native的比值，TTFT均值、可用ITL和最长输出间隔，完成/失败/未完成请求数。
4. 与5629aaf诊断提出的问题是否一致：新prefill是否降低首token等待和长停顿。未开启诊断时不要虚构GPU阶段时间。
5. GPUQ job、分配卡号、commit、source/checkpoint/image身份、退出和容器清理状态。

这是原custom负载的C32 eager结果，不是GPQA准确率或graph结果。不要用单层检查的compute_wall_seconds代替服务吞吐；它只是一次带同步检查的计算区间记录。

将程序打印的export_directory复制到可写clone的reports/server/下，核验manifest SHA256，仅暂存该公开目录并commit、push main。可另写脱敏操作记录；禁止git add .。原始请求、生成文本、token IDs、路由、激活和完整日志留在服务器。最后回报推送commit和报告目录。
