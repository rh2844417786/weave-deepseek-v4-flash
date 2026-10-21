# 原生 SGLang：动态 CTA 预算

本轮实现 `native_marlin_dynamic_cta_v1`，目标是提高原生 SGLang 的模型吞吐。2026-10-21 用户批准从 P0 诊断进入实现和服务器对照。它借鉴 Weave 的资源调度思路，控制 Marlin 内参与工作的 CTA 数量；它不具备论文 persistent 通信调度器的完整机制，不能标作完整 Weave 复现。

## 方法与适用范围

保留原生 MXFP4 Marlin 的权重、算术、资源约束下的配置选择算法和 workspace 容量，保留 shared experts、collective 和原始 prefill。仅在 decode 的 routed gate/up 与 down 中使用私有 JIT overlay。虚拟 worker 数变化可能改变 split-K 归约顺序，有限数值差异继续只记录。已有 SGLang 安装文件不修改，安装源码必须符合 `configs/native_budget_sources.json` 的固定 SHA256；不符合时立即报错。

物理 grid 在 CUDA Graph replay 之间保持不变。kernel 使用当前 GPU grouping 中的有效本地 routed slots 与全部 routed slots 计算预算：默认平衡 EP4 负载对应原生 grid 的 75%，按本地占比相对均衡值缩放，限制在 25%–100%，向上取整到八分之一 grid 桶，再受可执行 K/N/block 迭代数约束。本地工作为空时为零。远端 `-1` grouping 会合并多个远端专家，因此不能用本地/全局 grouping block 比例当作路由负载比例；报告同时保留路由数量和 block 几何。

保留原生按寄存器与共享内存约束选择配置的算法。补丁可能改变编译后的寄存器数，因此具体 tile、grid、shared-memory 选择可能与原始二进制不同；保证的是同一张捕获图的物理 grid 固定，不能宣称二进制 launch geometry 完全相同。证据字段 `native_grid` 指补丁沿用原生选型算法后得到的物理 launch grid，`last_n_tiles`/`last_k_tiles` 记录实际配置。

所有 stripe 划分与 lock 索引使用同一虚拟 worker 数，停用的 CTA 在访问 lock 前退出。预算计算在已有 Marlin kernel 内完成，不插入 planner kernel，不做测量窗口内的 CPU 读回，也不重建图。该预算约束的是活跃 CTA 数量，无法保证固定的物理 SM 分区。默认 75% 是未校准的启发式参数；它是否提高吞吐必须由本轮对照回答。

## 启动与实验契约

| 项目 | 本轮设置 |
| --- | --- |
| 模型 | 原始只读 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash` |
| 环境 | 已有 `lmsysorg/sglang:v0.5.17`，Python3.12、Torch2.11、CUDA13 |
| 分配 | GPUQ 分配四张 H100，使用实际分配，不写死物理卡号 |
| 并行 | TP4 / EP4，原生 Marlin，原生 shared experts |
| 两组 | `native` 原始实现；`adaptive` 动态 CTA 预算 |
| 图 | 两组 full CUDA Graph，捕获精确 1..32 batch，关闭 padding |
| Prefill | 两组均走原始 native |
| 负载 | GPQA Main 自然题长，Think High，temperature0，max_new_tokens32768，正常 EOS |
| 窗口 | 每组 1200 秒，加载、预热、排空、读证据和清理另计 |
| 初始并发 | 客户端与服务上限32；它是本轮控制变量，实际 decode batch 可为1..32 |
| 数值 | `numerical_policy: record_only`，不设误差或 token 一致率门限 |

`run_native_ep.py` 新增 `--native-budget off|adaptive|full`，默认 `off`，以及 `--native-budget-target-percent`，默认75。`off` 不传入预算环境变量，是原始 native 基线。`full` 是保留原生虚拟 worker 数的补丁控制选项，只用于独立验证；正式控制器固定执行 `native`、`adaptive` 两组。

启用预算必须同时指定原生 backend、`--graph-c32 --decode-graph full --request-timing`，并设置 `WEAVE_MAX_RUNNING_REQUESTS=32` 与 `WEAVE_CUDA_GRAPH_MAX_BS_DECODE=32`。P0、stage/execution profiler、stall diagnostic、fixed-prefix、replay capture 与自定义算子路径均不能混用。运行身份写入 `/work/<run_name>/native-budget`；overlay 使用 `/work/cache/weave-native-budget/rank*` 下的源码哈希目录，现有 `/work/cache` 编译缓存继续复用。

## 运行

先在 GPUQ 作业内完成 `tests/gpu/check_native_budget.py` 的真实 Marlin 验证，随后执行两组服务。GPU 测试应覆盖不同本地路由负载、空本地工作和同一张图内改变路由，并记录原始/补丁输出差异。CPU 测试通过仅证明本地契约和代码路径，不能证明 CUDA 编译、图 replay 或整模型吞吐已经成功。

```bash
python3 scripts/run_native_budget.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --native-budget-target-percent 75 \
  --setup-timeout 7200 --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

省略 `--execute` 只输出计划，不调用 Docker 或 GPU。每次执行创建新的 `native-budget-*` 实验目录、两组 `native-ep-*` 服务目录与公开导出目录，保留所有旧结果。控制器复用现有 GPQA streaming 客户端、请求身份检查、窗口标记、排空和 OwnedChild 清理。

默认 `--native-baseline-dir` 为仓库已有 `reports/server/gpqa-c32-9f1635df2afe`。只复用其中原生模型的 checkpoint 内容证明：先核对 attestation 哈希、各文件 stat 与 config/index 哈希，必要时重新读取全部本地权重校验。历史候选算子的数值状态不是本实验的前置门限。

服务器只允许从 GitHub 取材料。优先使用已有 prepared GPQA `main.jsonl` 与相邻 manifest，记录文件 SHA256 必须为 `cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888`。缺少时按既有 GPQA 文档从固定 GitHub 来源下载并校验，在已有镜像内离线准备。缺少模型、包、编译器或镜像时报告具体错误，不在线 pip/apt/HF 下载，也不拉取镜像。

## 证据与解释

控制器在客户端预热排空后、发布 `measurement-go.json` 前读取证据；窗口结束且客户端清理、服务排空后再读取一次。`/budget-coverage` 复用 graph evidence 的每 rank 端口，会同步 GPU 并复制聚合计数，因此仅允许在上述空闲边界访问。原生基线不调用预算端点。测量过程中不轮询预算快照。

- `native/graph-check.json` 和 `adaptive/graph-check.json`：实际 full graph replay、1..32 实际 batch 直方图与 B32 覆盖，检查四 rank 一致且窗口内无 recapture、失败 replay 或 eager decode 回退。
- `adaptive/budget-before.json`、`budget-after.json`：4 rank ×43层 ×2个 GEMM 的设备计数、空本地调用数、9个预算桶计数、最后一次 grid/worker/block/tile 和 routed-slot 数量，绑定补丁与上游来源哈希。
- `adaptive/budget-check.json`：每个 layer/role 的调用差分必须等于该 rank 实际 graph replay 总次数（包含所有1..32 batch）；桶计数之和必须等于调用差分。空本地工作仍计入一次调用和 bucket0，不视作缺失覆盖。
- `comparison.json`：两个1200秒完整窗口的 decode tokens/s 与 `adaptive_vs_native_whole_window_ratio`，同时保留请求完成/未完成统计、自然 EOS、数据/补充队列耗尽情况。
- `experiment.json`、`group.json`、`launch.json`、`runtime.json` 与 lifecycle：代码、镜像、checkpoint、数据身份、阶段、错误类型与清理状态。

`rows_with_budget_variation=0` 是允许且必须公开的观察。它说明该窗口内每个 layer/role 只观察到一个预算桶，不能据此声称“动态切换已在真实服务中出现”。观察到多个桶也可能同时受路由和实际 batch/shape 影响，仅凭直方图不能声称已隔离因果。

完整窗口吞吐比始终反映当前负载运行。若某组提前耗尽数据或可补充请求，`sustained_c32_ratio_available=false`，持续 C32 比值置空，并给出原因；不能把尾部低并发解释成持续32并发加速。该自然 GPQA 客户端不评估答案准确率或生成 token 内容一致率，报告明确 `generation_agreement_evaluated=false`。

非有限输出、无效内存访问、编译/运行失败、畸形 routing/tensor、来源不匹配、覆盖缺失和清理未完成仍是有效性失败。有限的数值差异只记录。失败保留原始状态和私有日志，不改写成成功，也不加入新的诊断实验。

## 回传

控制器打印 `work_directory` 与 `export_directory`。只回传 `export_directory` 中的静态白名单 JSON 与其 SHA256 manifest。原题、输入/输出 token IDs、生成文本、路由数组、权重、激活、overlay、私有日志均不公开。GPU 独立测试的聚合报告和容器 lifecycle 单独保存并审查后回传，不能把整个私有工作目录上传。

## GPU 验证命令

以下代码从仓库根目录、GPUQ 作业内执行，`WORK_ROOT` 使用现有可写工作根目录。它沿用调度器分配的四张卡，使用已有镜像和 compiler，为本次验证创建新目录；外层 owner 对容器设置总超时并在失败/取消时清理。该测试是独立算子与图 replay 验证，不是43层服务或性能结论。

```bash
export WORK_ROOT=/srv/weave-work
python3 - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from src.weave.container_lifecycle import run_container
from src.weave.gpu_allocation import allocated_gpus
from src.weave.runtime_contract import load_contract

root = Path.cwd()
work = Path(os.environ['WORK_ROOT']).resolve()
contract = load_contract()
allocation = allocated_gpus()
run = 'native-budget-gpu-' + uuid.uuid4().hex[:12]
out = work / run
out.mkdir(parents=True, exist_ok=False)
image = subprocess.run(['docker', 'image', 'inspect', contract['image'], '--format', '{{.Id}}'],
                       check=True, capture_output=True, text=True, timeout=30).stdout.strip()
if not image.startswith('sha256:'):
    raise RuntimeError('local content-addressed image required')
name = 'weave-' + run
command = ['docker', 'run', '--rm', '--pull=never', '--name', name,
           '--gpus', allocation.docker_argument, '--network=host', '--shm-size=8g',
           '-v', f'{root}:/workspace:ro', '-v', f'{work}:/work',
           '-v', '/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash:/model:ro',
           '-w', '/workspace', '-e', f'CUDA_VISIBLE_DEVICES={allocation.container_visible}',
           '-e', f'WEAVE_HOST_ALLOCATION={allocation.csv}', '-e', f'WEAVE_IMAGE_ID={image}',
           '-e', 'GLOO_SOCKET_IFNAME=lo', '-e', 'HF_HUB_OFFLINE=1',
           '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_DATASETS_OFFLINE=1',
           '-e', 'XDG_CACHE_HOME=/work/cache', '-e', 'HF_HOME=/work/cache/huggingface',
           image, contract['python_executable'], '-m', 'torch.distributed.run',
           '--standalone', '--nproc-per-node=4', '/workspace/tests/gpu/check_native_budget.py',
           '--model-path', '/model', '--output-dir', f'/work/{run}',
           '--cache-root', f'/work/{run}/jit', '--timeout-seconds', '7200']
deadline = time.monotonic() + 7320
with (out / 'gpu-private.log').open('w') as log:
    lifecycle = run_container(command, name, out / 'container-lifecycle.json', stdout=log,
                              expired=lambda: time.monotonic() >= deadline)
print(json.dumps({'gpu_work_directory': str(out), 'exit_code': lifecycle['exit_code']}))
if lifecycle['exit_code']:
    raise SystemExit(lifecycle['exit_code'])
for rank in range(4):
    report = json.loads((out / f'native-budget-gpu-rank{rank}.json').read_text())
    if report['status'] != 'complete' or report['cleanup']['status'] != 'complete':
        raise RuntimeError(f'incomplete GPU validation rank {rank}')
PY
```

GPU 验证的公开文件仅限 `native-budget-gpu-rank0.json` 至 `native-budget-gpu-rank3.json` 与 `container-lifecycle.json`，附上逐文件 SHA256 manifest；上传前使用 `scripts.run_gpqa_c32._safe` 检查每个 JSON。`gpu-private.log`、`private/`、`jit/` 不上传。即使验证失败也保留并回传现有的脱敏聚合报告，停止后续服务对照，定位真实编译/运行/覆盖/清理错误；有限数值差异不属于停止原因。
