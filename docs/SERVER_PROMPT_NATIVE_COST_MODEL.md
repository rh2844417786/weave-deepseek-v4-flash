# 服务器执行 Prompt：原生实测 MoE 成本模型

请在 `rh2844417786/weave-deepseek-v4-flash` 拉取本轮交付 commit，记录完整 SHA，并在运行期间固定该源码。阅读 `AGENTS.md` 和 `docs/NATIVE_COST_MODEL.md`，完成 `native_marlin_measured_cta_v2` 的统一校准、GPU 有效性验证及原生 SGLang 吞吐对照。本轮已批准成本模型实现和服务对照，直接执行统一入口，不再增加 P0、额外 profiler 或新的实验组。

1. 使用原始只读 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`、已有 `lmsysorg/sglang:v0.5.17`、锁定 Python3.12/Torch2.11/CUDA13 环境、prepared GPQA 和原有缓存。服务器只从 GitHub 取材料，禁止在线 pip/apt/Hugging Face 下载或镜像 pull；缺少依赖或 checkpoint 文件应记录具体错误，不能自动换环境。保留所有历史结果。
2. 在 GPUQ 申请四张 H100，使用作业分配的 `CUDA_VISIBLE_DEVICES`，不写死物理卡号。外层作业预留至少12小时，并按实际需要延长；校准自身可使用12小时，额外还需覆盖独立编译验证、两次模型加载/图捕获、两组各1200秒、预热与清理。若超时不足就延长外层及校准超时，不缩短1200秒测量窗口。
3. 找到既有 GPQA Main prepared `main.jsonl` 及相邻 manifest。记录文件 SHA256 必须为 `cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888`。缺少时按既有 GPQA 文档从固定 GitHub 来源下载并校验，在已有镜像离线准备；不修改 manifest，不填充、拼接或重复题目。
4. 将下方 `WORK_ROOT` 与 records 换成实际已核验的绝对路径，先去掉 `--execute` 检查离线预览，然后在同一 GPUQ 分配内执行完整命令：

```bash
python3 scripts/run_native_cost.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --native-cost-repeats 7 \
  --calibration-timeout 43200 --gpu-timeout 7200 \
  --setup-timeout 7200 --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

控制器默认依次启动真实完整模型校准服务，等待正常 ready 和四 rank 完整 `native-cost-rankN.json`，核验每类别×batch1..32及两种独立 held-out 输入，然后停止校准服务；运行新的实际 cost GPU/graph/lock 检验；最后运行原始 `native` 与 `cost` 两组 full-graph GPQA，各1200秒。不要跳过校准而生成猜测成本曲线，缺表不得退回75%启发式。若明确复用以前的完整表，在上述命令中加 `--calibration-dir /实际/WORK_ROOT/旧实验/calibration`；所有来源、镜像、GPU、模型与表身份检查仍需通过。

本次搜索的是九个 gate/up、down active CTA 预算对，目标为完整原生 MoE 实测关键路径，包含 shared 竞争、join、串行 all-reduce 和适用的 post-add。保留原始专家算术、prefill、共享和 collective 路径以及原生寄存器相关 launch 选择。它是固定 captured grid 内的软资源控制，不能标作物理 SM 分区或完整 Weave scheduler，也不能把 isolated GEMM 最快称为完整 MoE 最优。

两组保持相同原始 mixed precision/MXFP4、TP4/EP4、精确1..32 full CUDA Graph、无 padding、自然 GPQA Think High、temperature0、max_new_tokens32768 与正常 EOS。初始并发32是本轮控制变量。校准事件和临时绑定在吞吐前释放；测量期间不启动额外请求，不访问会同步 GPU 的 `/budget-coverage`，由控制器仅在空闲前后读取。

完成后核查并报告：

- 四份表的完整类别/batch覆盖、七次重复统计、两种 held-out 输入的预测误差及 regret、表摘要、源码/镜像/GPU/checkpoint 身份。预测误差不设阈值。
- 四份 `native-cost-gpu-rankN.json` 的实际 case coverage、profile SHA、非有限/内存/routing检查和图释放/容器 cleanup。GPU 验证失败应停止后续服务对照。
- 两组 `graph-check.json` 的四 rank 实际 replay、B32与全部实际 batch 覆盖、无 eager 回退；`cost/budget-check.json` 的四 rank×43层×两角色调用差分，以及每 MoE 恰好一次 planner、九候选对直方图和非法 plan 零计数。
- 预算对是否实际变化。观察到零就报告零，不能用 enabled 配置代替运行证据。
- 两组完整1200秒实际 decode tokens/s 与 `cost_vs_native_whole_window_ratio`，以及请求/补充数据耗尽情况。若 `sustained_c32_ratio_available=false`，明确没有持续32并发比值解释。自然 GPQA 客户端未评估 token 一致率，不补写成已评估。
- 校准服务、GPU 检验、两组服务和客户端拥有的进程/容器都已清理；失败保留 failed 状态和已有证据。

所有有限数值差异和生成 token 差异均为 `record_only`，不得添加误差阈值、allclose 或替代数值门槛，也不能因为预测 regret 或吞吐下降拒绝保存结果。仍保留非有限输出、非法内存、编译/运行、malformed tensor/routing、来源、覆盖、确定性算法和 cleanup 检查。

本轮策略为 `measured_moe_surface_v2`。跨候选五项工作量特征与 `native_routing_sha256` 的一致性门限已移除：差异只记录，不停止校准、不拒绝表、不增加替代阈值。保留每个候选自己的 `workload` 和 `workload_difference`，以及实际 cost replay 的差异记录；允许 `features_match=false` 或 `native_routing_sha256_match=false`。使用新目录重新校准，保留旧失败证据。

将控制器输出的 `export_directory` 白名单目录复制到可写 clone 的新 `reports/server/` 子目录，逐文件校验 manifest SHA256。只暂存该新目录，commit/push；不要 `git add .`。禁止回传原始题目、token IDs、生成文本、路由数组、激活、权重、完整日志、private/ 或 JIT overlay。失败同样回传已存在的脱敏聚合证据，具体原因另给脱敏描述，不能改写为成功。

最终回复执行 commit、回传 commit、GPUQ job、实际分配、镜像ID、校准完成状态、预测误差/选择 regret、GPU验证状态、完整模型图与 planner/43层计数覆盖、预算对变化、两组吞吐与比值、数据耗尽、清理状态及公开目录。没有真实服务证据时不宣称吞吐提高。
