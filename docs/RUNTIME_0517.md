# 当前服务器运行契约：SGLang 0.5.17

根据用户在当前服务器逐镜像实测的清单，默认选用本地已有的 `lmsysorg/sglang:v0.5.17`。历史 `sangfor-inference-...:v2.4.0.2` 镜像在当前 Docker 环境不可获取，原始预检文件仅作为各自运行的历史记录。当前每次作业重新采集证据，不从旧报告推定镜像可用。

配置统一保存于 `configs/runtime.json`：Python 3.12.3、PyTorch 2.11.0+cu130、torch CUDA 13.0、nvcc release 13.0、SGLang 0.5.17、Transformers 5.12.1。镜像 tag 每次执行前解析为实际 image ID，Docker 以该 ID 启动并禁止自动拉取。

默认 `WORK_ROOT=<repo>/work`，使用已有权限与继承 ACL，不修改目录所有权/权限；要求可写和可进入。`WORK_ROOT` 环境变量或 `--work-root` 可指定其他绝对路径。编译缓存、运行日志、JSON 证据均落入该目录，它已被 Git 忽略；需要回传的单份报告可另行 force-add 或复制到 `reports/`。

## 分阶段检查

- Core 功能作业：检查 Python、Torch、torch CUDA、nvcc 和四张 SM90 GPU。SGLang/Transformers 版本只记录，不作为核心 kernel 检查的执行条件；该阶段不 import SGLang。PyTorch 2.13 与当前选择的 ABI 不同，会在编译前产生明确的失败报告。
- Native/adapter 完整模型作业：还要求 SGLang 0.5.17 和 Transformers 5.12.1，并检查关键源码文件 SHA256。adapter 内在替换专家模块前重复同一检查，防止只有版本标签相同但实现发生变化。
- CUDA 扩展缓存名包含源码、Torch、CUDA、Python 与目标架构标识，隔离不同 Torch ABI 的缓存产物。

`scripts/check_runtime.py` 将当前 host 名称、container 名称、请求镜像、解析出的 image ID、实际 Python/包版本、CUDA/nvcc 路径和版本、GPU 枚举写入每次作业的 `runtime.json`。检查失败时仍保存报告，并返回非零退出码；核心测试成功还需四个 rank 结果齐全。

## 0.5.17 API 核对

官方 `v0.5.17` tag 对应 commit `29481685462732237d80d86076d6563e1f658102`，与用户实测的短 commit 一致。核对了 external model registry、DeepSeek-V4 类和 forward 参数、DeepseekV2MoE 中的 routed-expert/共享专家边界、runtime parallel context、MXFP4 Marlin method、缩放 helper 与 packed top-k 解包入口。文件哈希锁定在 `configs/sglang_adapter_sources.json`。

adapter 的 `get_server_args` 导入已改为该版本的 `sglang.srt.runtime_context`。`is_sbo_enabled`/`is_tbo_enabled` 仍由 MoE utils 提供；这两个模块也纳入源码哈希检查。

[0.5.17 的 MXFP4 Marlin 源码](https://github.com/sgl-project/sglang/blob/29481685462732237d80d86076d6563e1f658102/python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py) 新增 `build_marlin_moe_quant_info`，传入 dispatcher 的 `expert_map` 和 `global_num_experts`；这是对历史 0.5.16 源码核对中所见映射缺口的修复。此静态核对允许将 adapter 的版本目标更新为 0.5.17，数值/生成结果仍由当前服务器的新作业验证。

## 执行入口

这些命令在 GPUQ 已分配 GPU 4–7 的作业中运行：

```bash
git pull --ff-only
bash scripts/run_cuda_functional_gpuq.sh
```

默认生成 `<repo>/work/cuda-functional.XXXXXX/`：`runtime.json`、`build-and-test.log`、image ID、分配信息、退出码以及成功执行后的 `rank0.json`–`rank3.json`。

完整模型启动可先预览：

```bash
python3 scripts/run_native_ep.py --backend weave_functional
python3 scripts/run_native_ep.py --backend weave_functional --execute
```

原生组使用 `--backend native`。两个后端读取同一运行契约和镜像选择。功能 adapter 仍为显式 eager 验证阶段，正式性能比较尚须完成 graph 与数值验收。

如需当前主机的新模型清单报告：

```bash
bash scripts/run_preflight_gpuq.sh
```

默认输出 `<repo>/work/preflight.XXXXXX/` 下的 runtime 检查、模型预检 JSON、日志及退出码。此入口也使用当前 0.5.17 镜像。

脚本保持 GPU 分配硬约束 `4,5,6,7`。它们只在 GPUQ 开始执行已分配的作业后工作，不设置排队期间的计时器，也不占用 0–3 卡。GPUQ 的 1 小时作业时限从实际启动起算；原先预计的排队时长属于用户提供的调度快照，脚本不硬编码该快照中的到期时间。

## 取消与结果回传

三个 Docker 入口共用命名容器的信号清理逻辑，处理 SIGTERM/SIGINT、限时停止和 force-remove fallback。`container_lifecycle.json` 记录终止信号及清理结果。GPUQ 取消后的实际显存释放验证见 `docs/GPUQ_CANCEL_CONTAINER_CLEANUP.md`。

`work/` 保持忽略。已完成的 core 结果可直接导出至可追踪目录，无须重新占用 GPU：

```bash
python3 scripts/export_core_evidence.py --run-dir work/cuda-functional.Uzo4T0 --include-build-log
git add reports/server/cuda-functional.Uzo4T0
git commit -m "Record four-rank CUDA functional evidence"
git push
```

导出脚本复制 runtime、rank0–3、退出码、镜像/分配记录及可选 core 编译日志，附上原文件哈希。它不递归复制目录，不复制原生服务生成日志。缺少 rank 或状态不一致时，导出摘要不会标记通过。上面的路径来自服务器此次提交的运行说明。
