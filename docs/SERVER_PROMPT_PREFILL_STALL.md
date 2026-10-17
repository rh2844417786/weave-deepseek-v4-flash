# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt

请拉取 `rh2844417786/weave-deepseek-v4-flash` 的 main，记录实际 commit，阅读 AGENTS.md 和 docs/PREFILL_STALL_DIAGNOSTIC.md，执行 prefill、首 token 等待与输出长停顿诊断。不要修改计算实现或自行增加实验矩阵。

## 约束与准备

- 只通过 GitHub 同步；使用已有 `lmsysorg/sglang:v0.5.17` 和安装包，禁止 pip/apt/Hugging Face 下载、拉镜像。
- 模型仍为只读 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`，保留原始混合精度权重。
- 保留 GPUQ 分配的四张 H100 和 CUDA_VISIBLE_DEVICES，不硬编码卡号，不干扰别人的作业。
- 使用原 `/srv/weave-work`，保留 `/srv/weave-work/torch_extensions` 和 `/srv/weave-work/cache` 等缓存；新作业目录由脚本创建，不复用旧输出目录。
- 原 custom 数据路径为 `/srv/weave-work/bench-concurrency32-dataset.jsonl`，SHA256 必须为 `7d7bcdb9996f9293b63d337f21ee977f39013107ef23c2c411de2310d64501c2`。不得补齐输入、换数据或改排列。
- 完整回放证据使用仓库内 `reports/server/hopper-candidate-115da040aca7`，无需重跑 1376 点计算回放。
- 保留三组 native、staged_direct、hopper_fp8；统一 eager、C32、1024 token 上限，每组 600 秒。
- 这是带计时观测的诊断，不计算或宣传新的正式加速比。数值误差和生成内容差异只记录，不恢复任何误差或一致率门限。

首次 GPUQ 总运行预算建议 6 小时，按实际排队系统和阶段耗时延长；用户已允许根据需求延长。模型启动、四卡 smoke、采样和清理都需留够时间。

## 执行

在可写 clone 拉取 main，不覆盖本地未提交内容。提交四卡 GPUQ 作业，在取得分配后从仓库根目录运行：

```bash
export WORK_ROOT=/srv/weave-work
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
REPO_ROOT="$(pwd -P)"
python3 scripts/run_prefill_stall_diagnostic.py \
  --work-root "$WORK_ROOT" \
  --model-path "$MODEL_PATH" \
  --hopper-evidence-dir "$REPO_ROOT/reports/server/hopper-candidate-115da040aca7" \
  --dataset /srv/weave-work/bench-concurrency32-dataset.jsonl \
  --seconds 600 \
  --setup-timeout 7200 --gate-timeout 7200 \
  --acceptance-timeout 1200 --drain-timeout 600 \
  --port 30001 --evidence-port-base 30110 --stall-port-base 30210 \
  --execute
```

脚本会先验证原证据，执行四 rank collector CUDA smoke（控制线程 arm/snapshot、非默认 CUDA stream forward）及 Hopper 实际组装 smoke，再顺序启动三组。每组客户端预热结束后才 arm 采集；结束后先中止/排空请求、确认空闲，再读取 GPU event。不要在负载运行时手工调用 snapshot，也不要开启额外 profiler、graph 或 trace capture。

若端口被已有任务占用，可以同时指定三段互不重叠的空闲 localhost 端口，记录实际值；不要终止其他任务。若出现失败，保留私有 traceback、controller 和 server 日志，记录具体阶段；修复/重跑必须使用新输出目录，不删失败证据。

## 阅读与回传

完成后先检查 experiment.json 的状态、三组 `used_for_formal_performance:false` 和所有容器清理状态。每组重点读取：

- `client-stall.json`：匿名请求的首 token 时间、scheduler/API 时间和最长 64 个输出间隔。
- `stall-rank0.json` 至 `stall-rank3.json`：每次主机调用、所有 prefill 和每 32 次 decode 的 GPU 采样、43 层 routed expert 区间及 anchor 校准范围。
- `stall-analysis.json`：按阶段汇总、四 rank 最大/中位/差值、routed expert 与模型剩余区间、请求排队/prefill 时间、长间隔与 prefill 的重叠，以及缺失字段计数。
- 原有 coverage/client-summary/launch/runtime：确认实际路径、负载和来源。

汇报时回答：

1. Native、BF16 staged、Hopper 的首 token 等待主要落在哪些已观测区间？哪些字段仍缺失？
2. persistent prefill 的整模型和 routed-expert GPU stream 时间分别多长？最慢层和最慢 rank 是谁？
3. 最长输出停顿是否与 prefill 区间重叠？同时给出时钟 anchor 不确定范围。相关不等于因果。
4. sampled decode 的模型与 routed 区间怎样？明确它只是抽样，不能外推为全部 decode 总时间。
5. 是否出现采样容量溢出、跨 rank 序列不齐、缺层或采集失败？不能将这些缺失当作零耗时。

scheduler prefill 区间可能包含 chunk 调度；routed expert 时间包含通信与组装；CUDA stream 时间包含等待。不要把这些统称为纯 GEMM 耗时，也不要把四个并行 rank 的时间相加。可以报告诊断观察到的客户端 TPS，但不能用它替换 e8498aa 的未加观测吞吐。

将程序打印的 export_directory 复制到可写 clone 的 reports/server/ 下，核验 manifest 每项 SHA256，仅暂存该公开目录并 commit/push main。失败时也回传已有公开诊断目录，可另加脱敏操作记录。禁止 `git add .`；不要上传 prompt、生成文本、token IDs、激活、路由、原始 trace 或完整日志。最后报告推送 commit、GPUQ job、分配卡号、状态及报告目录。
