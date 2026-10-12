# GPQA 设备端阶段诊断

用户已同意在当前 GPQA 流程中记录专家计算、dispatch、combine、结束等待和实际 c/K/steal。入口增加 `--execution-profile`，不增加模型、数据集、人工输入长度或基线组。

## 服务器执行

在现有 GitHub checkout 拉取 main，使用已安装的 `lmsysorg/sglang:v0.5.17`、本地只读模型和已准备的 Main448 token ID。GPUQ 分配四张 H100，脚本读取它设置的 `CUDA_VISIBLE_DEVICES`。不要手工覆盖分配编号。

本轮 CUDA 源码身份已改变。让统一入口创建新 core、校准和模型验收目录，原有9906ca7c源码的374点保留为历史证据。新源码不接受旧bank或旧模型验收。

```bash
git pull --ff-only origin main
python3 scripts/run_full_experiment.py \
  --work-root /srv/weave-work \
  --export-root /srv/weave-work/public-reports \
  --records /home/jovyan/wangtonghan/weave-deepseek-v4-flash/data/gpqa/prepared/deepseek-v4-flash/main.jsonl \
  --calibration-kernel-timeout-seconds 120 \
  --execution-profile \
  --execution-profile-stride 128 \
  --execution-profile-capacity 8
```

上述 Python 命令在 GPUQ 实际授予四卡的作业内执行。沿用既有 GPUQ 提交方式；提交时为 core/校准/加载及两组各1200秒预留足够作业时间。每次校准调用仍最多1200秒，并使用新目录中的合法点恢复。

服务器不安装依赖、不拉容器镜像、不下载 Hugging Face 文件。缺失的镜像、checkpoint、数据清单或包会报错。服务和rank通信继续使用本机网络。

## 收集什么

每个 rank、43层、prefill/decode 分别维护计数。每128次调用采样一次；每层每阶段保留最近8个采样的时间段，累计 worker 时间和 c/K 直方图保留所有采样。缓冲区在 graph capture 前分配，replay 通过GPU计数选择新槽，地址保持不变。

| 字段 | 含义 |
|---|---|
| `compute_work` | 专家tile执行，包括其内部反量化、WMMA、激活、写结果及完成同步 |
| `input_wait` | tile输入就绪及对应CTA同步 |
| `compute_queue` | tile领取、依赖查询和pending轮询 |
| `dispatch_work` / `dispatch_wait` | 输入搬运协议及等待全部dispatch完成 |
| `return_work` / `return_wait` / `return_queue` | 专家结果回传、依赖等待和回传任务领取 |
| `reduce_work` / `reduce_wait` | origin加权合并及等待回传结果 |
| `end_wait` | 本卡CTA完成及peer完成等待 |
| `plan_wait` / `plan_work` | 到达planned状态的耗时；CTA0的planner工作包含在plan_wait中 |
| `schedule` | 该采样实际选择的c、K、steal预算 |
| `worker_counts` | 实际tile、stolen tile、回传task、合并row计数 |

设备时间戳使用H100的 `%globaltimer`，单位为纳秒。跨SM时间段只在同一GPU内比较，不把不同rank的时钟拼接。该寄存器的语义受目标GPU约束，当前GPU功能gate会检查实测记录。[NVIDIA PTX定义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)

`worker_totals_ns` 是CTA耗时之和，可能大于整层耗时。多个阶段的 `phase_intervals_ns` / `phase_envelopes_ns` 可以重叠，不能直接相加，也不代表NVLink利用率或Tensor Core Active。`kernel_span_ns` 从最早worker进入到最晚完成屏障退出，排除prepare、trace记录尾部写入、adapter的外部copy和原生outer all-reduce。

## 采样窗口与身份

开始Weave组之前通过localhost端点清零计数，排除模型加载和graph初始化记录。之后的计数包含warmup、正式请求和截止时abort/drain完成。现有1200秒截止规则和生成设置保持原值。

完成测量后，控制器等独占服务 `/flush_cache` 成功，再读取四个rank的 `/execution-profile`，然后停止服务。读取时比较前后的调用计数，发现并发forward或不完整记录会失败，不能发布为通过。

每份报告绑定CUDA源码、profiling Python源码、原始checkpoint内容、校准bank、镜像、设备与物理GPU分配。聚合器检查计数、环形槽序号、阶段顺序、stolen预算，以及tile/task/row计数守恒。

诊断只在显式开启时启用。关闭时编译到无计时数组和probe的kernel specialization。启用会影响寄存器和执行成本，所有摘要及comparison均标记：

```json
{
  "measurement_mode": "instrumented_diagnostic",
  "used_for_formal_performance": false
}
```

正式运行不能复用带探针的模型验收。正式结果仍以关闭诊断的独立记录为准。

## 回传文件

自动导出到 `/srv/weave-work/public-reports/full-experiment-<run-id>/`：

- `weave/execution-profile-rank0.json` 至 `rank3.json`。
- `experiment.json`、`comparison.json`、两组 `summary.json`。
- 新源码的core、完整校准bank和模型验收证明。
- `manifest.json`，包含逐文件SHA256。

只把该公开导出目录复制到可写Git clone的 `reports/server/` 后提交、推送。问题文本、token ID、生成内容继续留在工作盘。新增profile使用0664原子写入，保持继承ACL可读。

服务器执行者完成后汇报：运行commit和GPUQ job ID、四个rank的profile状态、各层decode的c/K分布、计算/领取/等待时间及近期时间段。若gate或收集失败，回传已有公开证据与具体错误；不要修改门限、替换权重或把带探针TPS当作正式加速比。

## 验证范围

CPU测试覆盖实际聚合、localhost接口、配置门限、损坏记录拒绝、身份绑定和导出。GPU core新增数值对照、graph replay计数、ring wrap与prefill/decode分离检查。

本机没有nvcc/H100。CUDA编译、寄存器/occupancy变化、timer行为以及真实43层诊断数据需要这次服务器作业确认。
