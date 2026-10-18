# C32 Graph：本地验证记录（2026-10-18）

基线 commit：`95490050b844fc1440ed99bf550d59b5a76dc91a`。实现和审查在本机 macOS 完成；本机没有 Torch/CUDA/H100，尚未执行新 graph GPU 检查或正式 graph GPQA。

## 已完成

- 对照 SGLang 固定 commit `29481685462732237d80d86076d6563e1f658102` 的 FullCudaGraphBackend、decode/base runner、配置与 batch alignment 源码，核对 hook 的构造、warmup、capture 与 replay 生命周期。
- 三项任务分别完成独立规格与代码审查；最终整合审查通过，未发现 Critical/Important 问题。审查修正了 smoke 中 eager 参考计算预写 graph 输出的问题：重放前写入无效元数据/NaN，先重放、校验和保存输出，再运行参考计算。
- 补齐数值记录完整性校验，要求实际有限的 `relative_l2` 和 `max_abs`。无误差上限；测试接受 `1e300`，不以误差大小拒绝结果。
- `--decode-graph full` 的 GPQA preview 成功，无需 GPU 分配、Docker 检查或在线下载。
- 旧 CUDA 专家代码及原 eager decode/finalizer/prefill 文件未改动；当前 hash 与原完整 GPU 证明一致。

## CPU 测试

最终相关测试命令：

```bash
.venv-data/bin/python -m unittest \
  tests.cpu.test_hopper_graph tests.cpu.test_hopper_decode \
  tests.cpu.test_graph_hooks tests.cpu.test_graph_evidence \
  tests.cpu.test_staged_decode tests.cpu.test_native_ep_launch \
  tests.cpu.test_hopper_launch tests.cpu.test_hopper_adapter \
  tests.cpu.test_hopper_prefill_adapter \
  tests.cpu.test_gpqa_c32_controller tests.cpu.test_gpqa_c32_contract \
  tests.cpu.test_gpqa_c32_client tests.cpu.test_c32_comparison_launch -q
```

结果：**138 项通过**，Python 3.12.14，9.027 秒。覆盖假重放、缺失 capture、padding、eager 回退、跨 rank 不一致、错误设备计数、缺失数值记录、离线启动和 GPQA 客户端/控制器行为。

整套 CPU 测试在审查修正期间执行一次：484 项，7 个 failure、3 个 error、2 个 skip。失败集中在旧通用 GPQA loader 的 `test_gpqa_performance` 与 `test_service_acceptance`。将基线 `9549005` 的源码导出到独立临时目录，用同一 Python 执行这两个模块：36 项复现相同 7 个 failure 和 3 个 error。此次未修改该通用 loader；本轮专用 GPQA C32 测试已通过。完整套件并非全绿，不能表述为全部测试通过。修正后仅重跑覆盖改动的上述 138 项。

## 源码身份

| 项目 | SHA256 |
|---|---|
| 旧 Hopper CUDA candidate | `bcabd0f4d1ea9430c480f4637940b13dcb8becd13bb016fdf6fda658e3d3b2ce` |
| 旧 BF16 CUDA | `25b51a34436de7494a80d739afb03ad0065f63639e4208b784270adf3632498f` |
| 旧 decode assembly | `8ed74ed9c02e136673dc6c6aa6eea63552d5affa7fb3bdc7c7276a0d73bae311` |
| 旧 prefill CUDA | `415a9931de1c875782fcf87a59d6294d342e392a44ca60b4ae68ddf3cf2e9345` |
| 旧 prefill runtime | `2538d4394a73d9630d733f9ef7f9a2826b6f6e52005b35f277d2618b8456c25b` |
| 新 graph runtime | `ac35f468554e9c08e46ee358bfc82501d36531ac7579106e798a1ed06246599b` |
| 新 graph observer | `75b8433958fd25e59c95e86ad56023a6af3042ea7b80a891ec823b2c58b4c142` |

## 实现决策与待验证项

1. 捕获全部整数 batch 1–32，禁用 padding。代价是启动时间与 graph 内存增加；是否在现有显存预算内须由服务器验证。
2. 优化 prefill 保持 eager。当前适配只降低 decode 的主机调度/launch 开销，不提供 prefill 图加速。
3. 新 graph runtime 单独记录身份、执行新四卡 smoke；复用未变的旧数值证明。代价是多一个运行时入口与一次 GPU 检查。
4. GPQA 默认仍为 disabled；显式 `--decode-graph full` 同时开启两组。服务器必须使用新 Graph Prompt，避免误跑原 eager 命令。

服务器仍需完成：CUDA 扩展编译、NCCL graph capture/replay、真实 43 层模型加载及 graph 重放、每组 1200 秒吞吐测量、清理和公开证据导出。结果返回前不声称 graph 已在 H100 跑通或吞吐提高。
