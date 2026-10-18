# Graph 阶段计时：本地验证记录（2026-10-18）

基线commit为1f4edbb。用户授权实际graph分阶段诊断脚本、GitHub交付与服务器Prompt。本机macOS没有CUDA/H100；以下仅为源码、CPU模拟和脚本验证，不能替代服务器GPU计时结果。

## 已验证

- 对照原HopperGraphDecodeRuntime，B32诊断子类保持workspace.run→slot SUM→计数器→finalizer顺序；B1–31直接调用原实现。
- 计时事件捕获前分配/初始化，层内四个事件使用external=True；捕获期间不读GPU值、不执行主机同步。
- 采样只围绕实际B32重放，整图end事件同步完成后读数，随后才能进行下一次重放。检查时间戳属于当前外层窗口，防止旧事件通过。
- 明确区分专家流水处理与纯GEMM、slot SUM等待区间与纯通信、剩余工作与纯注意力。Native只测整图，不伪造内部阶段。
- 四rank样本按真实重放序号对齐；选整图耗时最大的rank，再取同rank三段，避免跨rank最大值相加。
- 采样环保留末64点，并按测量前后快照过滤预热样本；诊断TPS不输出正式速度比。
- 原graph smoke与新event smoke有不同容器/目录，各自执行120秒清理期限，均要求正常退出及完整证明。
- 原CUDA、hopper_graph.py、hopper_decode.py、hopper_finalize.py、hopper_prefill.py未改动；事件模块单独记录源hash，observer及serving manifest包含新代码。

## 测试结果

以下组合在Python3.12环境运行：140项通过（9.362秒）。

```bash
.venv-data/bin/python -m unittest \
  tests.cpu.test_graph_stage_profile tests.cpu.test_graph_stage_integration \
  tests.cpu.test_graph_stage_controller tests.cpu.test_graph_stage_evidence \
  tests.cpu.test_graph_cleanup tests.cpu.test_hopper_graph \
  tests.cpu.test_graph_evidence tests.cpu.test_gpqa_c32_controller \
  tests.cpu.test_gpqa_c32_contract tests.cpu.test_gpqa_c32_client \
  tests.cpu.test_c32_comparison_launch tests.cpu.test_hopper_adapter \
  tests.cpu.test_hopper_prefill_adapter -q
```

新增回归覆盖事件覆盖写与读数顺序、采样环与序号、陈旧时间戳、异常采样清理、43层注册、默认关闭、两组launcher参数、独立检查容器、来源/缺层/错序/非有限计时拒绝、同rank阶段汇总和完整GPU证明格式。数值记录fixture接受1e300，未恢复精度门限。

相关Python文件py_compile与git diff --check通过。独立代码审查未发现Important/Critical问题，并核对了固定SGLang源码的构造、warmup与replay顺序。未重复完整套件；此前旧通用GPQA loader失败记录保持在HOPPER_GRAPH_C32_VALIDATION.md中。

## 服务器仍须验证

1. 原有四卡graph smoke正常通过并退出。
2. 新事件smoke在固定Torch2.11/CUDA13镜像内完成四rank、两张图、4次变化输入重放；事件时间有效、来源/数值记录完整、两张图reset后退出。
3. 真实Native与Weave服务各1200秒，实际B32图重放、无eager回退，Weave43层完整阶段样本及跨rank一致性。
4. 返回专家处理、slot SUM、finalize和剩余时间的真实占比，之后才判断主瓶颈。此次交付不声称性能提升，也不声称已取得GPU阶段耗时。
