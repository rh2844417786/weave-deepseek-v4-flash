# 校准 CUDA 故障定位与恢复

当前2048/c131/K1/compute故障和NAS满盘按 [最新恢复说明](CALIBRATION_2048_RECOVERY.md) 执行。下面的1024单点与权限复测已通过，保留为前一阶段记录。

适用报告：`reports/full_experiment-047ed4696f52.json`，作业 `job-20261003-095433-46cf54c3`。服务器摘要报告 core 通过、四 rank 各完成285点，随后 CUDA launch failure。原始工作 JSON 未进入 Git，具体失败调用尚未确认。

固定扫描顺序下，已完成点索引0..284之后的下一点为 `tokens=1024,c=131,K=8,mode=mixed`。这是诊断入口默认建议重放的配置，不是已证实的设备故障位置。

最新记录 `reports/calibration_diagnostic-047ed4696f52-1_operation_record.json`（d4275c0）报告：测试6ae957f的同步单点诊断四rank均通过，容器退出0；宿主GPUQ驱动因rank JSON权限拒绝而退出1，因此完整流程没有恢复。这次没有复现原CUDA错误，仍不能排除异步时序或扫描累积状态问题。

## 已修正的代码问题

原校准将 `min(60, remaining_budget)` 传入设备等待保护。CUDA 的 `wait_epoch/wait_count` 超时通过 `trap` 终止；临近测量截止时，正常工作也可能触发此保护并毒化上下文。现在设备等待保护独立配置，默认60秒；1200秒测量窗口由宿主容器管理器和完整实验控制器执行。晚完成或未完成点仍然不能进入 profile。

这修正了一个可复现的超时语义问题；是否解释原报告的 CUDA fault 仍需服务器重放确认。没有改变 CUDA 源码、权重、计算公式、bucket 或 c/K 网格。当前源码 digest 仍为 `9906ca7c29a1aca1ac74a20d2020ddb8f35d419371e0dc25bb42794b1fbc3623`，原 core 和合法285点可以通过原有身份/哈希检查继续复用。

JSON发布函数现会在fsync和原子replace前执行 `os.fchmod(fd, 0o664)`。临时文件默认0600会压掉继承ACL的mask；文件有扩展ACL时，group权限位对应mask，设置0664使named-user条目的读写权限可生效。[Linux ACL说明](https://man7.org/linux/man-pages/man5/acl.5.html) 此修复仅作用于校准JSON，不修改旧GPU记录、工作目录ACL或GPQA原始数据文件。Mac回归检查了受限umask下的新建与替换权限；Linux跨UID/ACL和真实导出链路由新诊断作业验证。

## 1. 在 GPUQ 分配4,5,6,7的作业内重放一个配置

仓库根目录运行；只使用已有镜像与本地只读模型，无在线安装或下载：

```bash
git pull --ff-only
WEAVE_REPO_ROOT="$(pwd -P)"
python3 scripts/run_calibration_job.py \
  --core-run-dir "$WEAVE_REPO_ROOT/work/cuda-functional.047ed4696f52" \
  --output-dir "$WEAVE_REPO_ROOT/work/calibration-diagnostic-047ed4696f52-2" \
  --lifecycle-path "$WEAVE_REPO_ROOT/work/calibration-diagnostic-047ed4696f52-2-lifecycle.json" \
  --diagnose-case 1024,131,8,mixed
```

输出目录、lifecycle文件和公开证据目的地均须是新的。`-1`已经使用，本轮改为`-2`；再运行应换`-3`。诊断会设置 `CUDA_LAUNCH_BLOCKING=1`，同步 CUDA 调用以缩小堆栈定位范围。[PyTorch 环境变量说明](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)

诊断仍使用完整原始第0层权重和原来的输入 fixture，最多1200秒；不会修改 `work/calibration.047ed4696f52` 的合法样本，不发布 measured profile，也不启动 GPQA。没有依赖运行时设置 `TORCH_USE_CUDA_DSA` 来改变现有 PyTorch 二进制；该选项需要构建时启用。[PyTorch 维护者说明](https://discuss.pytorch.org/t/how-to-enable-torch-use-cuda-dsa/202824)

每个 rank 的证据包含当前 case、迭代号、已完成点数、剩余/已用预算、失败阶段和首个堆栈。关键文件：

- `progress-rank<N>.json`：在每次 launch 前持久化当前配置；被硬截止终止时也能查看它。
- `failure-rank<N>-<invocation>.json`：实际异常阶段，例如 `kernel_launch`、`cuda_synchronize`。
- `diagnostic-rank<N>.json`：单点通过/不完整结果，明确 `used_for_calibration=false`。
- `calibration.log`：原始控制台日志，仅留工作盘。

诊断结束后公开证据自动导出到 `reports/server/calibration-diagnostic-047ed4696f52-2/`。GPU kernel故障后重新运行使用新的容器/上下文；不在坏上下文中继续提交计算。

## 2. 诊断命令退出0、四rank通过且导出成功后恢复完整流程

核对退出码0、四份 `diagnostic-rank<N>.json` 均为 `diagnostic_passed`、公开目录的 `manifest.json` 存在且 `experiment_status=diagnostic_passed`。仅容器退出0或四rank文件存在，不能替代宿主验收和导出通过。

```bash
python3 scripts/run_full_experiment.py \
  --core-run-dir "$WEAVE_REPO_ROOT/work/cuda-functional.047ed4696f52" \
  --calibration-dir "$WEAVE_REPO_ROOT/work/calibration.047ed4696f52"
```

恢复先校验原文件的源码、权重、镜像、设备、网格、哈希，取四 rank 共同完成点的交集，再测剩余点。完整profile通过后才进入模型eager/graph验收和正式两组GPQA。正常校准明确关闭 launch blocking，调试样本不混入性能校准。

如果同一点继续出现 CUDA fault，保留默认的失败停止，不自动把退出码1吞掉重试。请回传新的诊断证据，按具体阶段继续定位；当前还不能断言是越界、协议超时或硬件故障。

## 3. 回传证据

```bash
git add reports/server/calibration-diagnostic-047ed4696f52-2
git commit -m "Record isolated calibration fault diagnostic"
git push origin main
```

完整流程现在成功、失败或取消都会尝试导出已存在的安全证据。导出错误写入工作报告，不掩盖原始实验失败。原始题目、生成内容、token IDs继续保留工作盘，logprob仍仅记录。
