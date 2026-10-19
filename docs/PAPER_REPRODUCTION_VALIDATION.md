# Hopper persistent 集成本地验证

基线b999f17，目标是在真实persistent IPC grid内接入Hopper设备tile和论文调度机制。原Hopper+NCCL算子候选的历史记录保持不变。

## 代码验证范围

- Hopper设备tile：M64N64的fragment输出覆盖与唯一性、Shared对齐、MXFP4映射/量化契约的便携C++检查。
- 调度器：M64成本计算、ready优先领取、输入未ready跳过、gate完成后down才可领取；8个并发CPU claimant的release/acquire依赖及exactly-once任务领取。该测试使用address/undefined-behavior sanitizer另行通过。
- Runtime/adapter：默认factory保持旧路径，新后端source closure离线验证；43层prefill/decode实际分支与独立计数；新成本表source/image/checkpoint/device/shape/bucket绑定；graph launcher明确选择hopper_persistent。
- Controller/evidence：新入口阶段/容器输出路径；四rank证据、正常退出、成本实测推导、原权重检查绑定同一服务profile；CUDA及Python源码前后身份；受控案例各机制必须实际出现；oracle两行范围显式记录；服务43层device/scheduler/graph计数对应；无routed collective或operator fallback；公开白名单。

## 测试

Python3.12相关组合87项通过，10.347秒：

```bash
.venv-data/bin/python -m unittest \
  tests.cpu.test_paper_evidence tests.cpu.test_paper_reproduction \
  tests.cpu.test_hopper_persistent_runtime tests.cpu.test_hopper_persistent_scheduler \
  tests.cpu.test_hopper_persistent_tile tests.cpu.test_hopper_contract \
  tests.cpu.test_graph_cleanup tests.cpu.test_gpqa_c32_controller \
  tests.cpu.test_graph_stage_controller tests.cpu.test_hopper_adapter \
  tests.cpu.test_hopper_prefill_adapter tests.cpu.test_native_ep_launch \
  tests.cpu.test_graph_hooks tests.cpu.test_hopper_launch -q
```

Python编译、git diff --check与无需GPU的两个preview通过。没有重跑无关完整套件；旧通用GPQA loader的历史失败见HOPPER_GRAPH_C32_VALIDATION.md。

## 独立审查

CUDA审查未发现确定的Critical/Important缺陷，核对了grid-constant tensor maps、TMA/WGMMA顺序、所有producer lane fence、tile依赖、epoch复用与one-CTA-per-SM证据。

集成审查发现并修正：checkpoint profile未绑定、Python源码证明未绑定、受控机制活动验收不完整、独立oracle两行范围未注明、reference局部失败可能让其他rank进入NCCL。针对前四项新增回归检查；reference准备通过Gloo协商后再进入collective，并复用原有协调包装器且保持clamp10。修正的限定复审通过。

## 尚未验证

本机没有CUDA/H100。**没有进行NVCC/PTXAS编译、四rank IPC/TMA/WGMMA执行、graph重放、真实重叠观察或新服务吞吐测量。** 源码中有机制与CPU测试通过，不能替代这些GPU证据。

第一版设备tile仍有每输出tile重复A量化的成本；成本模型使用声明的efficiency1/overlap0保守常数。不同于论文原始BF16实验，本项目适配原DeepSeek混合精度权重与FP8激活计算。后续应验证这些适配的数值记录、模型预测和实际性能，不能无条件声称与原论文完全等价。

统一服务器入口、失败保留与公开导出规则见SERVER_PROMPT_PAPER_REPRODUCTION.md。性能数字描述当前实现，包含机制证据计数开销；不得直接作为纯调度收益或完整论文复现结论。
