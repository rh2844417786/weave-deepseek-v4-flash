# Hopper FP8 专家计算候选

## 交付范围

本轮按已同意的顺序实现第一阶段：独立的FP8激活/MXFP4权重计算候选，用原有真实B32 trace与BF16路径做数值和时间对照。生产服务默认配置保持原样。减少服务collective、接入新内核的完整服务、B32服务graph及重新评估动态调度，要等本轮GPU证据返回后推进。

最新714af26包含两次此前C32作业：新core和online-staged门通过，native eager完成600秒测量，cached persistent在生成token一致性验收停止，staged服务吞吐未开始。本候选不会通过改门限掩盖该失败，也不声称解决了全模型生成差异。

## 实现

`experimental/hopper_fp8`是独立CUDA扩展。现有`csrc`、BF16 core源码身份与服务默认值不变。候选使用原始只读MXFP4/E8M0专家权重，保持EP4、H4096/I2048/E256/top6、43层；没有引入附件的EP1或SGLang0.5.19 overlay。

每rank的前向包含GPU在线分组、活跃专家压缩、FP8激活打包、三个专家GEMM、SwiGLU及routed slot输出。实际GEMM使用Hopper TMA搬运、M64×N64×K32 FP8 WGMMA，K方向以128列为一个stage。每个专家最多32个有效token行，因此仅需一个M64 tile；空专家退出，图中的launch容量仍是预分配上界。M64是否适合当前稀疏B32分布，需要实测。

权重不整层反量化或另复制一份；tensor map在绑定时指向原始packed存储。每层bound handle持有权重和workspace。图捕获前分配缓冲区和描述符；前向不读取GPU计数到CPU、不执行显式同步、不重新编码tensor map。

候选不是完全流水化megakernel。它仍在stage边界等待TMA和WGMMA，只把较大的分块、packed权重处理与FP8数学路径作为一个可测量候选。

## 数值契约

- 每行、每128列BF16激活使用FP32 maxabs/448作为scale；全零块scale为1。
- 量化为FP8 E4M3，round-to-nearest-even、有限值饱和。
- MXFP4 E2M1 nibble精确映射为E4M3编码，原E8M0缩放保持每32列一组。
- 每K32产生FP32 partial，再依序乘activation scale和weight scale，按K顺序累加。
- GEMM输出、SwiGLU和routed product保留既有BF16舍入边界；down输入另做一次FP8量化。

FP8激活改变了数值路径，不能只凭生成固定文本相同接受它。独立Torch oracle不调用候选扩展，以有限FP8码本、原始nibble解码和K32 partial计算期望值。

数值对照分开记录，遵循用户2026-10-17明确指定的`record_only`政策：

1. **参考实现差异**：BF16基线对BF16 oracle、候选对FP8 oracle，保存relative L2和最大绝对误差。
2. **FP8漂移**：候选对BF16输出，同样保存误差值。

以上三类有限数值差异均不设置通过/失败门限，不调用`allclose`判定，不阻断任何层/样本计时。不得以后增加替代误差上限。误差统计使用FP64累积，减少统计过程自身溢出的风险。NaN/Inf、非法访问、编译/runtime异常以及分组/地址/身份错误仍失败。

当前证据schema为2，身份中明确包含`numerical_policy: record_only`。`complete`仅表示运行与覆盖完成，不表示数值一致性已验收。schema1旧报告保持原样，不被新代码追认成功；新复跑使用全新目录。

## 离线依赖与源码身份

已在开发机准备并随Git提交62个CUTLASS/CUTE头文件，固定于附件gitlink的commit`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。`third_party/cutlass_hopper/manifest.json`包含archive及逐文件SHA256；loader检查内容、精确文件集合和include闭包，缺失或变化就报错。服务器不下载、不安装、不执行submodule更新。

来源与许可见`THIRD_PARTY_NOTICES.md`和候选目录的reference provenance。候选源码、原BF16源码、GPU回放Python闭包和外层controller各自有摘要；修改oracle或loader也会改变证据身份。

## 服务器运行

使用已通过的当前BF16 core与服务器私有trace。此次没有修改BF16 `csrc`，因此可以复用source为`25b51a34436de7494a80d739afb03ad0065f63639e4208b784270adf3632498f`且模型/镜像匹配的core。程序会校验，不能用旧72592ee的core冒充当前BF16 core。

历史capture输入可以复用：它保留原72592ee/capture源码身份，与当前候选/基线计算身份分开记录，并重新核对模型、tokenizer、镜像和所有文件哈希。这不允许复用旧计时。

已知服务器路径（须核实存在）：

```text
WORK_ROOT  /srv/weave-work
MODEL      /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
CORE       /srv/weave-work/cuda-functional.staged-c32-a550a0e31585
TRACE      /srv/weave-work/replay-capture-a8eb705fd34f/private-trace
```

取得GPUQ四张H100分配后，保留其CUDA_VISIBLE_DEVICES：

```bash
python3 scripts/run_hopper_candidate.py \
  --work-root /srv/weave-work \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --core-run-dir /srv/weave-work/cuda-functional.staged-c32-a550a0e31585 \
  --capture-dir /srv/weave-work/replay-capture-a8eb705fd34f/private-trace \
  --warmups 3 --repetitions 20 \
  --setup-timeout 7200 --budget-seconds 7200 --execute
```

入口先在现有镜像强制运行两个Torch oracle单元检查（不能skip），再torchrun4执行。包括动态B1/7/17/32、空rank、全零输入、所有FP4 nibble、稳定地址和B32核心graph输入更新检查；然后覆盖真实43层×2个捕获步骤×4种变体。四种变体为BF16/FP8各自eager和graph，总计每rank344点、四rank1376点。

主计时范围从在线分组到本rank的routed FP32 slot，包含FP8打包/量化与数学开销，不含IPC/NCCL，不含服务attention/shared/sampling。只比较同模式、同rank、同layer/sample的成对时间；不要将它换算为服务tokens/s。

setup与测量预算分开，可以延长。结果目录必须全新；本版不复用partial计时，超时后保留已完成点并用新目录重跑。

## 状态与回传

| 状态 | 含义 |
|---|---|
| `compute_candidate_complete` / exit0 | 完整1376点且运行/布局检查完成；有限误差大小不影响状态，不表示数值一致性或服务结果 |
| `partial` / exit124 | 预算不足；不形成完整结论 |
| `failed` / exit1 | 编译、非有限输出、布局、身份或运行失败；查看私有日志和公开stage |

`summary.json`给出eager/graph成对GPU event中位数比及三类误差范围，不再输出precision_passed、implementation_passed、numerical_promotion_eligible等误差判定。`service_promotion_allowed`为false仅表明本入口是局部计算实验、没有执行服务接入，不属于精度限制。当前persistent generation mismatch仍单独待定位。

仅回传程序打印的export_directory：含源码身份、四rank标量结果、compile资源、capture元数据、core证据和manifest哈希。实际激活、routes、token IDs、`.pt`、完整private日志不能上传公开GitHub。ptxas只导出资源行；缓存命中导致资源日志为空时，应据实标记，不把local_bytes等同spill。
