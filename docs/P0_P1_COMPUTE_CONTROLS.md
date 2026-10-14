# P0/P1：热路径缓存与相同算术的计算组织对照

本轮按Pro意见实现第一批P0/P1。现有C32诊断显示低填充率的专家计算与跨rank等待值得优先处理。本轮代码提供对照工具和候选执行组织；吞吐收益、CUDA编译和真实数值结果仍以服务器回传为准。

第一批保留原版混合精度DeepSeek-V4-Flash、4×H100、TP4/EP4、全部43层、M16×N64、MXFP4 decode4及原BF16舍入/SwiGLU/routing语义。没有把staged local partial接入正式SGLang decode，也没有加入LUT、M8/GEMV、split-K、新graph buckets或旧成本模型重校准。

P0把workspace视图缓存到Python runtime，把device静态属性、occupancy、peer布局及权重指针/shape验证移到构造或绑定阶段。`BoundWeights`持有workspace和六个原始张量的强引用；绑定期间shape/storage必须保持不变。动态输入、profile/plan、trace范围、device guard和当前stream检查仍在每次调用执行。缓存视图减少host包装工作，不减少模型权重分配。

P0保留原始`Workspace.run(raw,...)`和`cache_views=False`作为未缓存host控制。`PeerRuntime.bind_weights(raw)`及`run(bound,...)`使用缓存路径。两条路径使用本版本相同的persistent GPU数学kernel，这样可以单独比较host缓存影响；它们不是重新运行旧95e8构建产物。数学提取可能改变编译资源，因此所有比较都记录当前构建资源，不能直接与历史服务TPS相除。

P1把共用数学逻辑抽入`csrc/tile_math.cuh`，persistent保留原有IPC、计数和完成协议。新增staged路径通过同stream的三个独立launch满足依赖：grouped gate/up → grouped down → 按原top-k slot顺序汇总本rank贡献。该路径不使用peer memory，不设跨CTA自旋屏障。只在staged中测试132/264/528 CTA与直接tile映射，persistent仍为132 workers。

直接映射的grid取冻结metadata的容量，固定grid通过GPU group_count限制实际任务。group metadata由离线CPU planner根据真实trace构造，按原token→slot首次遇见顺序分组，每组最多16行。计时内没有GPU `.item()`。该接口要求合法的完整冻结metadata；不能据此宣称已支持任意在线重分组、padding或连续批处理。

完整接口与算术提取说明见`docs/STAGED_COMPUTE_CONTRACT.md`。

## 身份和验收变化

CUDA身份现在是所有`csrc/**/*.cu/.cuh/.h`的相对路径、逐文件SHA256和编译flags/arch的canonical manifest摘要。构建源列表、扩展名与摘要来自同一快照，构建后重新比较文件清单。编译增加`-Xptxas=-v`以取得资源信息，没有启用fast_math。

因此旧95e8的core不能用于本版本。服务器先运行一次新源码的core与全部43层checkpoint检查。原functional GPU检查新增绑定、storage生命周期、错误shape/device、foreign-workspace拒绝、eager/graph及两种probe specialization资源检查。后续replay还有独立staged小形状GPU gate，以及全部43层真实输入的数值对照。无需重跑374点校准。

`resources()`分别记录persistent `<false>`和`<true>`的numRegs、localSizeBytes、sharedSizeBytes、occupancy与launch参数。带探针specialization始终是`<true>`，stride16只控制实际写入采样，并不让其余调用变成`<false>`。`localSizeBytes`不能单独当成spill字节；同时检查导出的ptxas stack/spill报告。资源数值均需服务器编译后获取。

## 阶段一：新源码core

在GPUQ已授予四卡的作业内，使用现有镜像和原始只读模型：

```bash
export WORK_ROOT=/srv/weave-work
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
export CHECKPOINT_TESTS=1
export CHECKPOINT_LAYERS=all
export CORE_RUN_NAME="cuda-functional.p0p1-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
bash scripts/run_cuda_functional_gpuq.sh
```

记录`$WORK_ROOT/$CORE_RUN_NAME`作为后续`P0P1_CORE`。不要覆盖GPUQ设置的`CUDA_VISIBLE_DEVICES`。编译或core失败时回传公开错误与证据，不复用旧source的结果。

## 阶段二：私有真实trace采集

`scripts/run_replay_capture.py`沿用已有custom JSONL，即九条SWE-Bench各重复200次。捕获eligible完整B32 decode的第64、128次，在每层origin slicing之前复制全量replicated输入。每rank预分配：

```text
x       [2,43,32,4096]  BF16
ids     [2,43,32,6]     int32
routing [2,43,32,6]     FP32
```

采集期间只enqueue GPU-to-GPU copies，状态端点只读CPU计数。采满后停止本次负载、abort独占服务并等待idle，再同步、下载和持久化。每个rank的`.pt`只含tensor及基本类型，用`torch.load(weights_only=True,map_location='cpu')`读取。文件使用0664原子写入以保留gpuq-runner的可读ACL。

采集身份绑定kernel、checkpoint、dataset文本SHA、镜像、tokenizer/config/encoding文件内容摘要以及共享capture UUID。四rank必须有相同UUID和完整64/128×43层覆盖；后续replay还会逐项验证四rank的x/ids/routing完全一致。raw tensors与路由ID只留服务器工作盘，GitHub只接收严格schema的manifest、描述、计数和哈希。

```bash
python3 scripts/run_replay_capture.py \
  --work-root /srv/weave-work \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --dataset "$C32_DATASET" \
  --core-run-dir "$P0P1_CORE" \
  --setup-timeout 7200 \
  --capture-timeout 7200 \
  --execute
```

`C32_DATASET`使用前次真实文件；已知上一轮路径为`/srv/weave-work/bench-concurrency32-dataset.jsonl`，SHA256为`7d7bcdb9996f9293b63d337f21ee977f39013107ef23c2c411de2310d64501c2`，仍需现场核对。入口验证1800行、9种输入各200次，不自动重建或下载。

成功时输出`private_trace_directory`和`public_manifest`。将前者记为`P0P1_TRACE`。可以用`--reuse-capture-dir`验证并复用同一采集版本的完整trace；该选项只继承历史UUID，其他当前身份及全部文件哈希必须匹配。不完整trace不能用合成输入补齐。

## 阶段三：冻结trace回放

`scripts/run_compute_replay.py`先验证新core、原checkpoint与私有trace，再在现有镜像中执行staged synthetic gate和四rank replay。无新增包或外部下载。

```bash
python3 scripts/run_compute_replay.py \
  --work-root /srv/weave-work \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --core-run-dir "$P0P1_CORE" \
  --capture-dir "$P0P1_TRACE" \
  --warmups 3 \
  --repetitions 20 \
  --stage-repetitions 3 \
  --setup-timeout 7200 \
  --budget-seconds 7200 \
  --kernel-timeout 120 \
  --execute
```

预算按实际需要配置，用户允许延长。setup与measurement分别计时；超出measurement预算时，完整点保存，当前未完成点不进入结果。若GPUQ政策限制单次时长，分开提交core、capture和replay；replay可增加`--resume-dir /srv/weave-work/compute-replay-...`续跑。capture/source/模型/镜像/测量参数或rank设备身份变化时不能复用计时点。恢复时只保留四rank共同完成的点，其余保留为orphaned证据后重测。新的invocation不会使用旧progress的deadline。

冻结trace可以用于后续不同计算源码的对照：保留其原capture源身份，重新核对checkpoint/tokenizer/image和文件哈希，并为新replay源码创建全新结果目录。复用输入与复用旧测量结果是不同操作。

每层、每个捕获样本包含以下7种控制，总计每rank602点、四rank2408点：

| 名称 | 范围 |
|---|---|
| `persistent_raw_no_probe` | 未缓存Tensor视图和raw host metadata路径，无probe specialization |
| `persistent_raw_probe` | 同样raw路径，带probe specialization，stride16 |
| `persistent_bound_no_probe` | 缓存视图和绑定权重/静态metadata，无probe |
| `staged_132` | 三阶段、132 CTA strided gate/up与down |
| `staged_264` | 同上、264 CTA |
| `staged_528` | 同上、528 CTA |
| `staged_direct` | 每个输出tile直接映射CTA |

所有staged变体使用同一reduce grid。每个case的变体顺序按固定seed轮换；各rank执行一致。全部输入、metadata、权重绑定在计时外准备；事件先初始化，warmup后才记录正式样本。每个variant先与独立checkpoint tensor oracle比较，再计时。原始BF16边界、relative-L2≤0.025及rtol0.03/atol0.02数值条件保留。

Replay先检查四rank的全部冻结replica相等。每层top6所有权总量必须恰为192 pairs；persistent与staged的local pairs/groups/tile jobs须相同。全局reference按原slot顺序组合各rank已舍入的贡献，persistent检查origin输出；staged检查本rank确定性FP32 partial。staged partial尚未接入外围all-reduce。

总时间使用单对CUDA events包围整个variant，host enqueue单独记录。Persistent总时间含输入workspace更新、prepare、IPC/协议与计算；staged总时间含gate/up、down、reduce，不含IPC。staged阶段时间另做3次辅助测量，不把额外阶段event开销加入主比较。不同范围的差值不能直接称为端到端收益。

控制器输出`compute_controls_complete`表示完整点和数值检查已完成；`partial`返回124表示可续跑；`failed`保存已有证据并停止。完整suite旧GPQA provenance/order失败继续单列，不删除测试或改Logprob门限。

## 回传与判读

只提交程序打印的public export目录。Replay公开包包括四rank点记录、资源、progress、source manifests、capture metadata、core证据、摘要和逐文件SHA256。`compilation-resources.txt`与`core/compilation-resources.txt`仅保留各自日志里的ptxas资源行；回放命中编译缓存时应查看core文件。如果两处都没有资源行，须明确报告编译日志缺失，不能从localSizeBytes推断spill。完整编译日志、`.pt`和实际x/ids/routing不公开。capture成功仅公开capture_manifest，失败详情在工作盘私有report，服务器执行者应提取阶段和错误类型回报。

摘要按相同layer/sample配对，提供每rank中位数与其最大值；后者不是跨GPU统一时钟的耗时。首先看：

1. Probe true/false资源和时间差，判定此前诊断的扰动。
2. Raw与bound无探针对照，判定host缓存是否减轻开销或抖动。
3. 相同M16算术下staged132/264/528/direct的完整时间，判定更多可调度CTA是否有效。
4. 真实pairs/groups/fill与时间变化，区分数学底座和执行组织问题。

若staged更快，下一轮再讨论将胜出组织接入local-EP服务。若更多CTA无效，再针对MXFP4解包/MMA数据流推进。当前未给出优化收益承诺，也不将本轮microbenchmark与原生日志1970 token/s换算正式加速比。
