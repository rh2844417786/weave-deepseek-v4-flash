# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt

请拉取 `rh2844417786/weave-deepseek-v4-flash` 的 main，记录实际 commit，阅读 AGENTS.md 和 docs/HOPPER_C32_SERVICE.md，执行新的 Hopper C32 服务对照。不自行增加或替换实验组。

服务器只通过 GitHub 同步。仅使用已有 `lmsysorg/sglang:v0.5.17`、已安装包、原始只读模型 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`。禁止在线 pip/apt/HF 下载和拉取镜像。使用 GPUQ 分配的四张 H100，原样保留 CUDA_VISIBLE_DEVICES，勿硬编码物理卡号。用户允许按需求延长作业；三组各测 600 秒，另需模型加载、编译、预热与清理时间，建议首次总预算 6 小时，可依实际 GPUQ 策略增加。

1. 在可写 clone 拉取并记录源码；不要覆盖本地未提交内容。确认现有镜像和数据文件。数据必须是原 custom 9×200 文件，SHA256 为 `7d7bcdb9996f9293b63d337f21ee977f39013107ef23c2c411de2310d64501c2`。已知路径 `/srv/weave-work/bench-concurrency32-dataset.jsonl`，只允许使用相同内容的其他路径，不重新生成。

2. 使用仓库内完整回放包 `reports/server/hopper-candidate-115da040aca7`。其中包含四卡 core/checkpoint，统一入口会检查所有 29 个文件及全部 1376 点；不需要重新跑该回放，也不要用旧 schema1 失败报告替换。

3. 提交 GPUQ 四卡作业，在获得分配后从仓库根目录执行：

```bash
export WORK_ROOT=/srv/weave-work
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
REPO_ROOT="$(pwd -P)"
python3 scripts/run_c32_comparison.py \
  --comparison-suite hopper \
  --hopper-evidence-dir "$REPO_ROOT/reports/server/hopper-candidate-115da040aca7" \
  --work-root "$WORK_ROOT" \
  --model-path "$MODEL_PATH" \
  --dataset /srv/weave-work/bench-concurrency32-dataset.jsonl \
  --seconds 600 \
  --setup-timeout 7200 --gate-timeout 7200 \
  --acceptance-timeout 1200 --drain-timeout 600 \
  --execute
```

统一入口先执行实际 HopperRuntime 四 rank 组装 smoke，然后依次测 native eager、staged_direct BF16 eager、hopper_fp8 eager。每组完成 32×8 token 响应检查，再进行 C32/1024 token/600 秒测量。不能把 native 切成 graph，不能省掉 slot 汇总或外层 TP 汇总，不能回退为 native 专家后继续标为 Hopper。

数值误差和生成 token 内容差异均只记录，不作为停止或拒绝性能结果的门限。不要恢复 relative L2、绝对误差、allclose 或 token 一致率阈值。响应必须完整、请求身份正确；非有限输出、编译/运行错误、组装算法错误、源码身份和覆盖缺失仍需停止并报告。

4. 若失败，读取私有 controller/server 日志定位具体阶段，记录错误与容器清理状态。不要删除失败用例或重写失败报告；需要重跑时用新目录。失败时也回传程序生成的公开诊断目录。若 smoke 中 Triton 编译失败，保留完整私有报错并在公开操作记录中给脱敏错误摘要；不在线安装替代版本。

5. 完成应为 `comparison_complete`。核对四卡各 6 个 runtime case 与 4 个 ordered-finalizer case；三组各 32×8 个有效响应；两个自研组全部 43 层实际调用、B32 histogram、额外 slot collective 数。吞吐取 `client-summary.json.decode_tokens_per_second`，不能用 scheduler 日志或仅完成请求的 token 数代替。报告 native/BF16/Hopper TPS、两个 Hopper 比值、TTFT/可用 ITL、窗口/完成/失败/未完成数及 token 一致性记录。不要将本次 custom C32 eager 实验写成 GPQA 准确率或 fullgraph 结果。

6. 将程序打印的 export_directory 复制到可写 clone 的 reports/server/ 下，逐项验证 manifest SHA256，只暂存该公开目录并 commit、push main。可添加脱敏操作记录（commit、GPUQ job、物理 GPU、阶段、退出/清理）；不要 `git add .`。禁止上传 acceptance-private.json、原始 prompt、token IDs、生成文本、激活、路由、trace 或完整私有日志。

最后回报推送 commit、作业状态和报告目录；未完成三组则明确写未完成，不计算有效端到端加速比。
