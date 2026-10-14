# 服务器 DeepSeek V4 Flash：P0/P1 执行 Prompt

请拉取Weave仓库main，读取AGENTS.md、docs/P0_P1_COMPUTE_CONTROLS.md和docs/STAGED_COMPUTE_CONTRACT.md，执行Pro建议第一批P0/P1。记录运行commit。代码已提供热路径缓存、带/不带probe控制、真实trace捕获、相同M16算术的staged132/264/528/direct回放。

只执行本批，不实现local-EP服务替换、LUT、M8/GEMV、split-K、新graph buckets或成本模型扫描。不得把微基准结果称为服务吞吐加速。

服务器只用现有`lmsysorg/sglang:v0.5.17`和本地原始模型。禁止在线pip/apt/HF下载、拉镜像或换权重。GPUQ分配四张H100后保留其CUDA_VISIBLE_DEVICES，不硬编码。用户允许按实际需要延长作业时间；可把core、capture、replay分成独立作业，给cleanup留出余量。

先执行新源码core。源码身份方案已改变，不能复用旧95e8的core。此步骤含编译、绑定/graph检查和全部43层checkpoint检查，不是374点校准：

```bash
git pull --ff-only origin main
export WORK_ROOT=/srv/weave-work
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
export CHECKPOINT_TESTS=1
export CHECKPOINT_LAYERS=all
export CORE_RUN_NAME="cuda-functional.p0p1-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
bash scripts/run_cuda_functional_gpuq.sh
export P0P1_CORE="$WORK_ROOT/$CORE_RUN_NAME"
```

核对原有custom9×200数据：前次文件是`/srv/weave-work/bench-concurrency32-dataset.jsonl`，SHA256为`7d7bcdb9996f9293b63d337f21ee977f39013107ef23c2c411de2310d64501c2`。如果路径变化，从原作业记录定位同一文件，勿重新生成或用9行token-ID文件代替。设定C32_DATASET后采集：

```bash
export C32_DATASET=/srv/weave-work/bench-concurrency32-dataset.jsonl
python3 scripts/run_replay_capture.py \
  --work-root /srv/weave-work \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --dataset "$C32_DATASET" \
  --core-run-dir "$P0P1_CORE" \
  --setup-timeout 7200 --capture-timeout 7200 --execute
```

该作业从真实C32服务捕获eligible步骤64/128、全部43层、全部4rank的输入和路由。采集不测性能；数据GPU预分配、idle后落盘。记录脚本打印的private_trace_directory为P0P1_TRACE。不要将rank*.pt、激活、路由ID、原始日志或生成内容提交GitHub。

随后执行回放，P0P1_TRACE必须是上一步实际打印的私有路径，不得猜测：

```bash
python3 scripts/run_compute_replay.py \
  --work-root /srv/weave-work \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --core-run-dir "$P0P1_CORE" \
  --capture-dir "$P0P1_TRACE" \
  --warmups 3 --repetitions 20 --stage-repetitions 3 \
  --setup-timeout 7200 --budget-seconds 7200 --kernel-timeout 120 \
  --execute
```

可按GPUQ作业剩余时间增减预算。若返回partial/124，保留旧目录，在同一源码、测量参数、trace和GPU分配下增加`--resume-dir <上次work_directory>`续跑；不得覆盖旧结果或降低数值条件。完整应为每rank602点，四rank2408点。脚本先执行staged synthetic gate，每个real-trace变体数值通过后才计时。

任一阶段失败时停止后续依赖步骤，保存已有证据并汇报具体阶段、错误和清理状态。不要改门限、切回native或用合成trace伪装完成。编译或GPU测试失败不等于性能结论。

将capture公开manifest及replay打印的public export目录分别复制到可写clone的reports/server/下，校验所有manifest SHA256后仅暂存这些明确文件，提交并推送main。不要git add .。工作盘的原始trace和私有日志留在服务器。

最后汇报运行commit/GPUQ job/物理GPU、source/checkpoint/tokenizer/capture身份、完整点数与正确性、probe true/false registers/local/shared/spill信息、raw vs bound、staged各grid的配对时间及分布、最慢层、pairs/groups/fill、partial/超时及清理状态。给出哪种解释得到支持：probe扰动、host热路径、CTA并行组织，或仍需优化解包/MMA数据流。

不要把128线程×132 CTA解释为实测occupancy；不要直接增加原persistent workers；不要将staged纯计算与persistent协议总时间的差值称为E2E加速。C32 full graph及local-EP服务将在本批证据返回后决定。
