# CUDA Graph 检查退出挂起修复（2026-10-18）

## 证据与原因

失败报告 `reports/server/gpqa-c32-189654a40ed7` 的四 rank 都完成了 64 次捕获、38 次重放，但检查容器未退出。提交 `77be03b` 记录了进程组销毁阶段的 futex 等待，约 46 分钟后作业取消；Native 和 Weave 服务都未启动。

原 `tests/gpu/check_hopper_graph.py` 在 `finally` 中直接调用 `destroy_process_group`，此时 `graphs` 字典和循环局部变量仍持有 CUDA Graph。含 NCCL 操作的图持有 communicator 引用，销毁 communicator 时可能等待图先释放。该机制与 NVIDIA 的 [Process Hang](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/process-hang.html) 说明一致。另一个问题是报告在清理前已写 `passed`，无法单独证明检查流程完成。

## 改动

1. 每个图创建后立即登记所有权，包括后来捕获失败的图。
2. 清理顺序固定为：同步 GPU → 逐个 `graph.reset()` → 再次同步 → Gloo monitored barrier → 销毁 NCCL → 销毁 Gloo。图引用、输入、绑定权重和 workspace 在 reset 时仍有效，避免过早释放缓冲区。显式 reset 不依赖 Python 垃圾回收或别名是否还活着。
3. 每 rank 原子写入 `graph-cleanup-rankN.json`，记录阶段、已 reset 数、开始/完成时间与错误类型，权限沿用 0664。完整原始异常留在私有日志。
4. GPU 检查完成先写 `gpu_checks_passed_cleanup_pending`；只有清理返回成功才能写最终 `passed`。新的 rank 报告 schema 为 2，验证器要求全部 64 个图 reset 和完整清理阶段，旧报告保持原样，不充当新的完整成功证明。
5. 控制器新增 `--graph-cleanup-timeout`，默认 120 秒，从首个 rank 的清理标记开始计时。由父进程监控，即使 CUDA/NCCL C++ 调用阻塞也能进入既有 owned-container 停止与导出流程。GPU 编译/检查总超时仍为 7200 秒；不会因为 rank 文件写了通过而跳过子进程退出检查。
6. 超时或失败时保留 `stage: graph_gate` 和各 rank 清理记录，不启动两组服务；成功退出、四份完整证明校验通过后才进入两组各 1200 秒测量。

## 验证

Python 3.12 本地执行以下相关测试，98 项通过（6.364 秒）：

```bash
.venv-data/bin/python -m unittest \
  tests.cpu.test_graph_cleanup tests.cpu.test_hopper_graph \
  tests.cpu.test_hopper_decode tests.cpu.test_graph_evidence \
  tests.cpu.test_graph_hooks tests.cpu.test_gpqa_c32_controller \
  tests.cpu.test_gpqa_c32_contract tests.cpu.test_gpqa_c32_client \
  tests.cpu.test_c32_comparison_launch -q
```

新回归检查包括：图存在别名时 reset 仍先于 communicator 销毁；reset 失败不得继续销毁 communicator；失败 capture 的图仍受管理；父进程面对实际挂起的 CPU 子进程按清理期限报错；控制器超时后导出阶段证据而不启动服务；缺失、旧版或不完整清理报告不得通过。数值 fixture 继续允许任意大的有限误差，无新增精度门限。

Python 编译和 `git diff --check` 通过。独立代码审查未发现阻塞问题。此次未重复全套测试；此前完整套件中的旧 GPQA loader 失败记录见 `HOPPER_GRAPH_C32_VALIDATION.md`。

原 CUDA 代码、Hopper decode/prefill/finalizer 和 graph 计算运行时未修改。此次只调整 GPU 检查生命周期、控制器等待与证据验证。**本机没有 CUDA/H100，以上为代码及 CPU 验证；尚不能宣称服务器清理挂起已实测消失。**

## 服务器重跑

使用更新后的 `docs/SERVER_PROMPT_HOPPER_GRAPH_C32.md`，拉取 main 后从新的作业目录重新执行 full graph 流程。保留已有模型、镜像和编译缓存；保留上次失败目录。两组仍各 32 并发、1200 秒、Think High，自然 EOS；不能跳过 graph 检查或退回 eager。

新报告应同时满足：schema 2、`gpu_checks_status: passed`、`cleanup.status: complete`、`graphs_reset: 64`，检查进程正常退出，然后才产生两组真实服务 graph 覆盖与吞吐结果。如果仍失败，按清理文件中的阶段定位，不通过延长总超时或改写状态掩盖失败。
