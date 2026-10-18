# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt：Graph 阶段计时

请拉取 `rh2844417786/weave-deepseek-v4-flash` 最新main并记录完整commit，阅读AGENTS.md、docs/GRAPH_STAGE_TIMING.md。本轮目标是定位Weave full CUDA Graph为何慢：测真实B32重放中的专家处理、跨rank slot SUM、聚合以及整次graph剩余耗时。

这是插桩诊断。不能将诊断TPS当成新的正式加速比；现有未插桩基线为reports/server/gpqa-c32-9f1635df2afe（Native1894.69、Weave247.69 token/s）。不新增其他实验组，不修改模型、计算内核或精度政策来让结果通过。

## 环境

使用已有lmsysorg/sglang:v0.5.17（Torch2.11/CUDA13/Python3.12）和只读模型/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash。服务器只从GitHub获取材料；不pip/apt/HF下载，不拉镜像。保留/srv/weave-work的所有数据与编译缓存及历史报告。

通过现有GPUQ申请同一组4张H100，沿用分配的CUDA_VISIBLE_DEVICES，不硬编码卡号。两组顺序执行，每组正式窗口1200秒、客户端并发与服务上限32，full decode graph，prefill eager。保留Think High、temperature0、最多32768输出token、自然EOS、固定GPQA顺序。

使用已验证prepared main.jsonl，相邻manifest完整。文件SHA256必须为cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888。若在其他路径，仅替换下面--records；不要修改数据或manifest绕过校验。

## 执行

先git status --short、git pull --ff-only origin main、git rev-parse HEAD；保留本地修改，不reset/clean。确认新文件src/weave/graph_stage_profile.py和tests/gpu/check_graph_stage_events.py存在。

在仓库根目录，GPUQ作业内，用已有Python3.12执行：

```bash
REPO_ROOT="$(pwd -P)"
python3 scripts/run_gpqa_c32.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --prefill-evidence-dir "$REPO_ROOT/reports/server/c32-prefill-63dac24bf44a" \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --decode-graph full \
  --graph-stage-profile --graph-stage-stride 128 --graph-stage-capacity 64 \
  --graph-gate-timeout 7200 --graph-cleanup-timeout 120 \
  --setup-timeout 7200 --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

可先移除--execute预览，必须看到measurement_mode=instrumented_diagnostic、formal_gpqa=false、stage_event_gate_required=true，以及两组都带--graph-c32、--graph-stage-profile、--decode-graph full。

控制器先运行原graph检查，然后在另一个容器中运行新的事件计时检查，分别确认图释放/正常退出，再加载两组模型。首次作业预算建议6小时，按编译/加载需要延长。每组1200秒不包含加载、检查、预热和清理。

不要开启旧execution-profile/stall profiler，不发额外请求，不更改128/64采样参数来追求某个结果。有限数值误差和生成内容差异只记录，无精度门限。错误、非有限输出、缺失覆盖、陈旧事件、source mismatch和清理失败必须如实保留；不能退回eager或跳过检查。

## 汇报

1. GPU event smoke是否完成：每rank两张图、4次变化输入重放，事件时间是否新鲜，计数与完整数值记录，清理是否正常退出。
2. 两组是否实际full graph重放，是否出现B32、是否有eager回退；Weave是否43层全部覆盖。
3. 从stage-diagnostic-summary.json读取Weave对齐采样的critical_rank统计，表格列whole_graph_ms、expert_pipeline_ms、slot_sum_ms、finalize_ms、routed_total_ms、residual_ms的mean/p50/p95/max和平均占比。
4. 列出各rank三段总耗时、whole_graph rank极差、最慢的若干层，说明主要时间落在哪一段。如果结论不明确，明确保留不确定性。
5. Native只提供整次graph时间，不能杜撰Native内部三段；不能将不同rank的最大阶段值相加。专家阶段包含输入准备/分组/量化，通信阶段包括rank/stream等待，剩余时间包含多个未细分工作，不能直接称为纯GEMM、纯NVLink或纯attention。
6. 报告每组实际采样数、保留的replay序号范围、采样间隔和容量。环形缓冲保留末段样本，不宣称逐步覆盖整个1200秒。报告诊断TPS时明确插桩扰动，正式加速比必须为空。

成功或失败均保留证据。将脚本打印的export_directory复制到可写clone的reports/server/，核验manifest，只暂存该公开目录并commit/push main。禁止git add .；不上传题目、token IDs、生成文本、激活、路由或完整私有日志。

最后给出执行源码commit、结果推送commit、GPUQ job、卡号、报告路径，以及根据真实阶段时间得到的瓶颈结论。失败则给出最后完成阶段，不能用旧吞吐数字冒充本轮诊断结果。
