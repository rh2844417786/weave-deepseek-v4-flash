# CUDA 功能核心与 SGLang 接入

这批代码实现了可供服务器编译的功能路径。CUDA 13 clock-rate API 已修复；服务器已回传两轮原始报告 `reports/server/cuda-functional.Uzo4T0` 与 `cuda-functional.YfA3yp`。runtime 检查、四 rank 小张量数值和 core graph replay 均通过，导出清单列出的 19 个文件哈希已核对。测试范围及验收记录见 `reports/core_acceptance.json`。对外后端名称为 `weave_functional`，真实模型尺寸、SGLang adapter/graph、实测 profile、非零 stealing 和正式性能仍待验收。

## 实现内容

- `csrc/megakernel.cu`：每 rank 独立 cudaMalloc 工作区、IPC handle 导入导出、system-scope release/acquire 和递增 epoch；检查 Hopper、peer access 与 native peer atomics。每层是一次准备 kernel 和一次 persistent execution kernel，路由统计在 GPU 上读取。
- persistent execution：按目标 rank 去重输入；保留全部 token-expert 任务；执行 packed E2M1/E8M0 专家 gate/up、受限 SiLU-mul、down；按 top-k slot 返回，origin 乘路由权重并合并。包含空 rank 和零输入分支。
- `csrc/scheduler.h`：论文第 3 节式 (1)–(16) 的 `c/K` 联合搜索和残余工作 stealing 预算。成本模型由 persistent kernel 的 GPU prologue 执行。无实测 profile 时仅允许显式固定计划进行功能测试，不填充虚构吞吐曲线。
- Dispatch 不分 chunk；计算按 token 所属 chunk 入队，Combine 等待对应 chunk 的计算全部完成；comm workers 在 Dispatch 完成后执行一段连续 stealing，再进入 Combine。计算 workers 耗尽队列后协助尾部回传。
- `src/weave/weights.py`：从原版 safetensors 逐层读取本 rank 的完整专家、packed bytes 和 E8M0 scales。按服务器索引中的 `layers.L.ffn.experts.E.w1/w3/w2` 命名定位，验证形状与存储 dtype。
- `src/weave/sglang_backend.py`：保留原生 gate/HashTopK、共享专家、attention/mHC；按连续 token 分片指定唯一 origin，返回 owner rows 的 routed contribution，再由原生外层归约恢复复制输出。当前目标为源码核对后的 SGLang 0.5.17，启动前校验关键文件哈希；限定 TP4/EP4、expert TP1、attention DP1、标准外层 runtime 和 Marlin 的缩放合同。所有 43 层替换完成后才开始 forward，记录 prefill/decode 调用次数。
- 外部模型包 `weave.sglang_ext.models`：使用 SGLang 官方 external model registry 的覆盖入口，在第一次 eager forward 前安装 adapter。unsupported top-k、graph capture、EPLB/MTP/CP/TBO/SBO/torch.compile 等路径报错，无 native expert fallback。

## 与最终论文复现的差距

当前专家计算采用 warp 归约的标量 MXFP4 GEMV。stealing work unit 是一个 token/expert forward，尚不是论文中的 Tensor Core GEMM tile；因此此版侧重正确性、同步和接入，不作为最终速度结论。输入传输为 BF16，返回槽为 FP32，成本模型分别使用 2 和 4 字节，未套用论文相同 B 的简化。

core 使用 device valid-count 与 epoch，提供独立 CUDA Graph replay 检查。SGLang adapter 的变长 graph 生命周期尚未验收，功能入口显式选择 eager；不能用它与启用 graph 的原生组产出正式加速比。

还需完成：Tensor Core tile 内核及对应 tile stealing、实际校准曲线、SGLang graph 集成、原生与本后端逐层/完整模型差分，以及两组同配置的正式计时。

## 服务器验证顺序

在 GPUQ 分配四张 H100 的作业中，先运行小张量核心检查：

```bash
git pull --ff-only
bash scripts/run_cuda_functional_gpuq.sh
```

默认镜像与版本检查见 `docs/RUNTIME_0517.md`，默认输出与缓存目录为 `<repo>/work`。该入口记录实际 runtime、image ID、GPUQ job ID、编译/运行日志和退出码。`tests/gpu/check_functional.py` 实际编译扩展并使用四个进程，检查去重远程任务、空 rank、BF16 参考数值、改变路由与 valid count 的 graph replay，以及设备端选计划。profile 数值仅为明确标注的测试夹具，不能拿来正式测速。只有四个 rank 的 JSON 均报告成功，才证明这组小张量检查通过。

核心检查通过后，再做完整模型 eager 集成：

```bash
python3 scripts/run_native_ep.py --backend weave_functional
python3 scripts/run_native_ep.py --backend weave_functional --execute
```

第一条预览，第二条执行。输入模型只读挂载，源码只读挂载；Torch 扩展编译产物写到 WORK_ROOT。无 profile 时默认 `(c=1,K=1,steal=0)`，日志明确为固定计划功能阶段。原生路由给出的缩放状态由 Marlin 外层 helper 保留，核心只乘传入的 top-k 权重；需要差分验证比例因子与共享专家恰好应用一次。

## 动态 profile 合同

`--profile` 必须位于 WORK_ROOT，JSON 必须包含 `status="measured"`、`kernel="scalar_mxfp4_token_v1"`、匹配设备名称/SM 版本/SM 数的 `device_key`、`hidden_size`、`intermediate_size` 和 `candidates`。每行是 `[c,K,compute_FLOP_per_second,bandwidth_byte_per_second,chunk_efficiency,overlap]`。compute rate 对应 `N-c` workers，overlap 在 `[0,1)`，chunk efficiency 在 `(0,1]`；不合法或硬件不匹配时拒绝启动。

CPU 公式依据本地 Weave v3 PDF 第 6–7 页第 3 节；DeepSeek packed 专家格式依据原版模型 config、服务器分片头和官方 `inference/kernel.py`（修订 `60d8d70770c6776ff598c94bb586a859a38244f1`）。
