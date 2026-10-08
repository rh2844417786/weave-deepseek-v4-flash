# 2048 / c131 / K1 / compute 故障恢复

证据：`reports/server/full-experiment-8c14e090e30e/`（a57baec）。四rank共有362/374个合法点，首个缺失配置为2048/c131/K1/compute，iteration1在cuda_synchronize报错。rank2最先记录错误，仍有1086秒预算；其他rank约60秒后报错。完整校准和GPQA均未完成。

## 当前判断

源代码中的131个通信CTA不计算tile，只剩1个计算CTA；K1要求通信CTA等整个chunk。等待保护不观察计算进展，合法长计算也可能触发trap。已有同K的1024/c131/compute样本约27秒，2048工作量翻倍，60秒裕量需要验证。clock64保护按设备峰值时钟换算，不是精确wall timer。当前错误尚未定位到具体trap指令，不能断言根因已确认。

新增日志记录每个GPU调用的开始、结束/失败wall时间、请求guard、同步/异步模式，以及本invocation起始点数和新提交点数。保存的285..361点单是timed samples就有188.19秒，超过本invocation首个失败elapsed113.46秒；因此旧日志不能证明这77点全部在这个invocation中新执行。

CUDA源码哈希仍为 `9906ca7c29a1aca1ac74a20d2020ddb8f35d419371e0dc25bb42794b1fbc3623`。现有core和362点按原校验复用；调试样本仍不进入正式校准。GPU快照只记录调度器分配的四张卡的clock/temperature/ECC等字段，不作为验收门限。

## 1. 使用本地clone和本地工作盘

`/home/jovyan`的NAS已有Errno122记录。请在有空间的本地clone拉取GitHub，在GPUQ已分配四张H100并设置CUDA_VISIBLE_DEVICES的作业内执行。现有原版混合精度模型和镜像保持离线使用。core与校准数据已在`/srv/weave-work`。

先以原来的异步条件和60秒guard重放**真正失败的case**：

```bash
git pull --ff-only
python3 scripts/run_calibration_job.py \
  --work-root /srv/weave-work \
  --core-run-dir /srv/weave-work/cuda-functional.047ed4696f52 \
  --output-dir /srv/weave-work/diagnostic-2048-c131-K1-async60-1 \
  --lifecycle-path /srv/weave-work/diagnostic-2048-c131-K1-async60-1-lifecycle.json \
  --export-dir /srv/weave-work/public-reports/diagnostic-2048-c131-K1-async60-1 \
  --diagnose-case 2048,131,1,compute \
  --launch-blocking 0 --kernel-timeout-seconds 60
```

三个输出路径须全新，重复时换编号。读各rank的progress/failure/diagnostic、gpu-before/after和launch.json，重点看`gpu_iteration_elapsed_seconds`、`kernel_timeout_seconds`、`launch_blocking`、`initial_completed_cases`和`committed_this_invocation`。

## 2. 若60秒复测失败，只改变guard再测同case

保持所有配置不变，用全新`async120-1`三路径，把上一条命令的`--kernel-timeout-seconds`改为120。仍保持`--launch-blocking 0`，避免同步模式改变时序。两次诊断都最多1200秒；失败停止，不在坏CUDA上下文里继续运行。

若120秒仍失败，回传首次rank的原始trace、wall时间和GPU快照，继续定位设备异常/非法访问/协议问题。不要删除这12个必需点或用估算值补profile。

若120秒通过且60秒失败/调用时长支持等待裕量假设，可按下节显式用120秒校准保护恢复。60秒也通过时，可保留默认60恢复；单点成功仍需完整校准和模型gate验证。

## 3. 从362点恢复，并将导出放在本地盘

```bash
python3 scripts/run_full_experiment.py \
  --work-root /srv/weave-work \
  --export-root /srv/weave-work/public-reports \
  --records /home/jovyan/wangtonghan/weave-deepseek-v4-flash/data/gpqa/prepared/deepseek-v4-flash/main.jsonl \
  --core-run-dir /srv/weave-work/cuda-functional.047ed4696f52 \
  --calibration-dir /srv/weave-work/calibration.047ed4696f52 \
  --calibration-kernel-timeout-seconds 120
```

120仅是显式校准等待保护选择；c/K/bucket/权重与原始数学保持原契约，每次校准总预算仍为1200秒。缺失点全部实测通过、四rank完整profile发布后，脚本才进入模型eager/fullgraph gate及两组GPQA。正式GPQA仍按自然Main448、ThinkHigh、greedy、32768/normalEOS、每组ready后1200秒运行。

prepared token清单被Git忽略；新本地clone需要显式指定已有清单（上面的NAS路径只读取），或复制到本地后校验manifest/hash。不能假设git clone会带回这些私有文件。

完整入口默认导出到`WORK_ROOT/public-reports/<run-name>/`，也可使用`--export-root`选择绝对路径。启动前探测可写性；运行中若磁盘再次失败，原始错误仍保留在工作报告。

## 4. 回传

在本地clone中将实际成功或失败的**公开导出目录**复制到`reports/server/`，核对manifest哈希后只暂存该目录并commit/push。不要复制整个`/srv/weave-work`，其中包含私有GPQA输入及生成。

本轮Mac仅验证控制器/元数据/导出逻辑，尚未运行修复后的H100诊断或正式GPQA。
