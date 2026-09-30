# DeepSeek EP 路由参考实现计划

**Goal:** 为 EP=4 的 GPU dispatch/combine 提供可执行 CPU 判据：token 按目标 rank 去重，专家任务保持独立，合并时每个 top-k 槽恰好贡献一次。

**Architecture:** 接收原生 router 已选出的 expert IDs 和最终路由权重，以 `(origin_rank, token_index)` 标识唯一 token。CPU 参考不重新选专家或归一化权重；共享专家由原生 runtime 处理。此模块不是 CUDA 通信或 Weave megakernel。

**Tech Stack:** Python 标准库、unittest；服务器 SGLang 包版本 0.5.16。

**Spec:** `docs/DEEPSEEK_V4_FLASH_MIGRATION.md`。

## 约束

- 原版 DeepSeek-V4-Flash 混合精度模型，256 个 routed experts、top-6、GPU 4–7、EP=4。
- 连续均分专家，每 rank 64 个；不同 origin rank 的相同 token index 不得混淆。
- 路由权重包含上游实际缩放，参考模块不二次缩放。专家输出输入到 combine 时尚未乘该路由权重。
- GPQA Main、Think High、输出上限 32768、每组 1200 秒的正式实验约束继续有效。

## Task 1：dispatch/combine CPU 合同

Files: `src/weave/routing.py`, `tests/cpu/test_routing.py`。

接口：`build_dispatch_plan(routes, num_experts=256, ep_size=4, top_k=6)` 生成逐 rank 唯一 token 列表和逐专家任务；`combine_results(plan, results, hidden_size)` 校验槽完整性并合并。

- [x] 先写失败用例：两个目标专家在同一远程 rank 时只传一个 token，仍保留两个专家任务；两个 origin 的 token index 相同；三个空 rank；重复/缺失回传；路由权重和不等于 1。
- [x] 执行 `python3 -m unittest tests.cpu.test_routing -v` 确认缺少模块时失败。
- [x] 实现不可变 token/task/plan 数据结构、输入边界检查、dispatch 去重、按 slot 顺序的加权合并。
- [x] 执行上述用例与完整 CPU suite。GPU 通信状态保持未执行。

## Task 2：锁定版本源码核对

Files: `docs/SGLANG_V0516_INTEGRATION.md`。

- [x] 检查服务器来源说明中的 commit `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`，记录 DeepSeek MoE 入口、共享专家边界、quant method 与 runner 分派。
- [x] 对文档中的推断标记静态来源范围；包版本与镜像来源说明不能证明镜像内源码未被修改。
- [x] 记录下一步 GPU adapter 必须满足的输入输出合同，不把 CPU 参考注册为正式 Weave 后端。
- [ ] 提交并推送本轮可核验代码及文档。

## Task 3：服务器原生 EP4 启动候选

Files: `scripts/run_native_ep.py`, `tests/cpu/test_native_ep_launch.py`。

- [x] 先用 CLI 测试证明脚本缺失时失败。
- [x] 默认 JSON 预览；显式 `--execute` 才调用 Docker，拒绝非 4,5,6,7 的分配。
- [x] 解析实际 image ID，运行标准 dispatcher/Marlin 的 EP4 候选并记录日志、GPUQ job ID 和退出码。
- [x] 用模拟 Docker 进程核验命令参数和退出证据；完整 CPU suite 21 项通过。
- [ ] 在服务器确认 native EP 数值、生成和 CUDA Graph。模拟 Docker 测试不覆盖此项。

本计划只交付 CPU 合同、静态接入核对和原生启动入口；完整 CUDA Weave 实现仍未完成。
