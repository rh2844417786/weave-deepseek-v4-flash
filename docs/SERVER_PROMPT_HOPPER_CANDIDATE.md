# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt

拉取rh2844417786/weave-deepseek-v4-flash的main，记录实际commit。先读AGENTS.md和docs/HOPPER_EXPERT_CANDIDATE.md。本轮只运行独立Hopper FP8专家计算候选与原BF16的真实B32 trace对照；不改生产服务、不放宽旧生成一致性门限、不跑新数据集。

服务器只使用现有lmsysorg/sglang:v0.5.17和本地原始混合精度DeepSeek-V4-Flash。CUTLASS依赖已随Git提供并有SHA256检查；禁止pip/apt、HF下载、拉镜像或git submodule update。保留GPUQ分配的四张H100 CUDA_VISIBLE_DEVICES，不硬编码。

先核对已有私有文件：
- 模型：/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
- BF16 core：/srv/weave-work/cuda-functional.staged-c32-a550a0e31585
- 真实trace：/srv/weave-work/replay-capture-a8eb705fd34f/private-trace

core必须对应25b51a34436de7494a80d739afb03ad0065f63639e4208b784270adf3632498f且模型/镜像相同。当前没有改原BF16 csrc，可以复用通过此身份校验的core；不要复用72592ee旧core。若指定路径不存在，定位同身份的已有目录；若没有，通过既有run_cuda_functional_gpuq.sh补齐all43 core。私有trace可以保留旧capture源码身份，脚本会独立校验，不能重新造合成trace代替。

取得GPUQ分配后运行：

```bash
git pull --ff-only origin main
python3 scripts/run_hopper_candidate.py \
  --work-root /srv/weave-work \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --core-run-dir /srv/weave-work/cuda-functional.staged-c32-a550a0e31585 \
  --capture-dir /srv/weave-work/replay-capture-a8eb705fd34f/private-trace \
  --warmups 3 --repetitions 20 \
  --setup-timeout 7200 --budget-seconds 7200 --execute
```

用户允许按需延长时间，GPUQ总时限覆盖setup、measurement和cleanup。预算不足时保留partial证据并在新目录重跑，不能拼接不同运行/源码计时。

入口要求两个Torch oracle单元检查实际通过，随后四卡编译和合成/graph门，再对真实43层×2样本×4变体回放：bf16_eager、fp8_eager、bf16_graph、fp8_graph。完整应为每rank344点、总1376点。不要关闭graph或跳过失败样本来填满结果。

实现正确性与FP8漂移分开：candidate对FP8 oracle、baseline对BF16 oracle先通过原门限；FP8-vs-BF16的relative-L2超过0.025时，保留诊断计时并报告precision_gate_failed，不得改门限或自动切服务默认。编译/实现错误停止并报告具体stage，读取工作盘私有日志定位，不安装缺失依赖。

完成后核对四rank身份、全部点数、资源和所有manifestSHA256；仅将打印的export_directory复制到可写clone的reports/server/下，明确暂存该目录，commit并push main。不要git add .；不要上传.pt、原始激活、路由/生成token IDs、prompt或完整私有日志。

汇报commit/GPUQ job/物理GPU、source/checkpoint/capture身份、oracle2项结果、GPU/graph门、完整点数、FP8实现误差和BF16漂移、eager/graph配对时间比及分布、空专家和有效行分布、compile资源、超时与cleanup。说明计时只覆盖本rank在线group→routed slots，不含通信及完整服务，不能称为C32服务吞吐加速。既有两次persistent_cached生成验收失败仍然保留，不能借本轮结果宣称已解决。
