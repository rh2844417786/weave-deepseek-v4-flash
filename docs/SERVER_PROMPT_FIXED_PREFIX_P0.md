# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt

请拉取 rh2844417786/weave-deepseek-v4-flash 最新main，记录完整commit，阅读AGENTS.md和docs/FIXED_PREFIX_P0.md，执行固定前缀的原生重复/P0干预拆分诊断。

目标是比较相同输入token、batch32、前缀与KV逻辑位置下的原始logits，定位P0短输出的来源。本次不测正式吞吐，不把有限误差或top1差异变成门限。

本次重跑包含prefill页预算修复：五组统一chunked-prefill-size=16384、max-prefill-tokens=16384，32题按256-token页对齐后需要9472。入口已设置，不要手动回退到8192。核对每组及每阶段的prefill-budget.json：页大小256、实际预算足够、scheduler实际请求上限至少32。此前fixed-prefix-d21fe5a23861失败记录保留，使用脚本自动生成的新目录重跑。

使用已有 lmsysorg/sglang:v0.5.17、Torch2.11/CUDA13/Python3.12 和只读模型 /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash。保留缓存；不在线pip/apt/HF下载，不拉镜像或模型。缺依赖或源文件身份不匹配就保留具体失败，不换版本或改hash绕过。

GPUQ申请4张H100，沿用CUDA_VISIBLE_DEVICES分配，不硬编码物理卡号。五组顺序使用同一分配，总运行预算建议8小时，可按需要延长。WORK_ROOT需可写，并留出数GiB保存私有logits。

使用已有GPQA Main prepared main.jsonl及相邻manifest，文件SHA256必须为：
cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888

固定前32题的选择SHA256必须为：
a5748d54c4605d57042784ad3e2de569ae017796411239a360f90aaabc9b3018

在GPUQ作业内从仓库根目录，用已有Python3.12执行；records替换为实际已有且校验通过的绝对路径：

```bash
python3 scripts/run_fixed_prefix.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --setup-timeout 7200 --phase-timeout 360 \
  --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

入口会执行更新后的事件smoke和五组：native_a、native_b、p0_full、p0_events、p0_buffers。各组原生TP4/EP4、full graph精确1..32，统一关闭CPU/GPU重叠调度并开启固定前缀处理器。P0读回全部关闭，仅事件插入与缓冲保留的组合不同。不要自行调整stride、精度、CTA或实验组。

每个测量阶段固定8个预测位置、一次32题、不补题。native_a先生成私有teacher，再与其余组沿同一teacher执行。仅此短计算检查ignore_eos=true，正式GPQA规则不改。检查实际输入token、位置、完整32行和7次B32 decode；若未形成该轨迹，保留失败，不能填造覆盖。

重点判读：
1. 先看native_a/native_b的波动，再看三组P0的误差分布、top1匹配数及首次观察到差异的位置。所有有限差异只记录，不设门限，不因“全部top1不同”中止。
2. 强制输出与teacher不一致是控制输入未施加成功，应保留失败；它与原始logits/top1是否接近teacher是两回事。
3. 核对实际事件节点和分组引用数量：full有两者、events只有事件、buffers只有引用，且runtime readback为0。
4. 检查native_a的2位置trace是否取得完整86个Marlin记录。该trace来自未插入P0阶段事件的原生图，只用于观察kernel时间线，不计算正式加速比。
5. 记录KV池槽位摘要/行顺序是否相同；相同逻辑位置不等于证明KV数值完全一样。单次结果不直接宣称根因已修复。

成功或失败都保留证据。只将脚本输出的export_directory复制到可写clone的reports/server/，核对manifest SHA256，只暂存该新公开目录并commit/push main。禁止git add .；禁止上传teacher原始token、logits .pt、题目、生成文本、原始Chrome trace或私有日志。

最后回报推送commit、GPUQ job、GPU分配、五组状态、公开目录、原生重复及P0差异统计、首次差异位置、原生trace是否取得、清理结果。失败时给出经过脱敏的具体阶段和原因，不修改原报告状态。
