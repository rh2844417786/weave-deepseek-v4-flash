# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt

以下为可直接粘贴的完整指令。源码执行提交固定为
`df3211186bbfbd4d604f5ff0a745402447b95fcb`；文档后续提交不会改变该源码。

---

请执行 `rh2844417786/weave-deepseek-v4-flash` 的 native peer-tail 验证与正式吞吐流程。
方法标识必须保持 `native_marlin_peer_tail_v1`，不是完整 Weave 论文复现。
本轮代码已在开发机完成源码/CPU检查；CUDA 编译、四卡机制、完整模型 Graph 和吞吐均尚未验证。
请直接执行已授权的常规工作，失败时保留真实证据，不用 enabled 标志或预览替代实际结果。

## 1. 固定源码并保留现有数据

使用本机已有可写 clone（通常 `/srv/weave-push-repo`），先打印 `git status --short`、origin URL
和 HEAD。origin 应是 `rh2844417786/weave-deepseek-v4-flash`。保留本地修改、缓存和历史报告，
不执行 reset --hard、git clean 或强推。

获取 GitHub 最新 main，然后为以下准确提交建立独立执行 worktree。若可写 clone 干净且在 main，
可先 `git pull --ff-only origin main`；存在修改时保留它们，用 fetch + 独立 worktree 完成执行。

```bash
REPO=/srv/weave-push-repo
WORK_ROOT=/srv/weave-work
CODE_SHA=df3211186bbfbd4d604f5ff0a745402447b95fcb

git -C "$REPO" status --short
git -C "$REPO" fetch origin main
git -C "$REPO" cat-file -e "$CODE_SHA^{commit}"
mkdir -p "$WORK_ROOT/checkouts"
git -C "$REPO" worktree add --detach "$WORK_ROOT/checkouts/native-tail-$CODE_SHA" "$CODE_SHA"
cd "$WORK_ROOT/checkouts/native-tail-$CODE_SHA"
git rev-parse HEAD
```

若这个 checkout 已存在，先核对 HEAD 和工作树内容；符合固定提交且干净可复用，
否则创建新的带唯一后缀目录，不能重置原目录。

阅读该执行 checkout 的 `AGENTS.md`、`docs/NATIVE_TAIL_EXECUTOR.md` 和各检查脚本 `--help`。
S1/S2 的独立提交为 `5b4b7dc`、`9b0668c`，本次完整执行仍使用上面的最终源码提交。

## 2. 环境与数据

只使用服务器已有的：

- `lmsysorg/sglang:v0.5.17`，记录实际 image ID；
- Linux x86_64、Python3.12.3、Torch2.11.0+cu130、CUDA13/nvcc、既有 SGLang/TVM-FFI；
- 原始只读 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`；
- `/srv/weave-work`、现有编译缓存和已准备 GPQA Main。

服务器不在线 pip、apt、Hugging Face 下载，也不拉 Docker image。缺少工具或安装文件时报告
具体阻塞，不静默替换环境。memcheck 必须调用该环境已有的 compute-sanitizer；缺失即停止该阶段。

records 优先用 `/srv/weave-work/gpqa-main-c32/main.jsonl`；先核对文件、相邻 manifest 和已有
客户端校验。该 prepared 数据的 SHA256 应为：

`cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888`

若实际已验证文件在别处，使用其绝对路径并记录，不改变内容或题目顺序。若需要恢复数据，
只使用仓库现有固定 GitHub 来源与摘要校验，随后在已有镜像中离线准备。

## 3. GPUQ 作业与统一入口

通过本机既有 GPUQ 入口申请四张 H100，使用实际 allocation，不能写死物理卡号。
作业运行时限须覆盖首次编译、独立 GPU 检查、memcheck、校准和服务加载；建议至少24小时，
按平台限制和实际需要延长。不要缩短正式的两个1200秒窗口来迁就超时。

在同一 GPUQ 分配内，从固定 execution checkout 运行：

```bash
python3 scripts/run_native_tail.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --native-tail-repeats 7 \
  --gpu-timeout 28800 --calibration-timeout 43200 \
  --setup-timeout 28800 --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

不要把不带 `--execute` 的 preview 当作已运行。控制器按依赖完成：

1. S1 真权重 native down 与提取 down：全部原生 M block、B32及更大batch、图重放与scratch复用。
2. S2 四卡 IPC、系统同步、cooperative Graph、ready/ack/epoch/slot、延迟/取消与清理。
3. S3–S5 统一执行器：精确任务集合、C1/C2、serial、未注入与受控 stealing、真实同卡区间。
4. 实际 compute-sanitizer memcheck，针对 B32 的同一执行器；错误退出码不能忽略。
5. 真实模型原生 capture_one warmup 内的所有结构组、精确B1..32校准与留出验证。
6. 使用生成表的 GPU 当前-workload planner/graph 验证。
7. 原始 native 与 peer_tail 正式服务窗口。

只需要单独验收基础版本时可以使用 `--stage foundations`，但该结果不能标为整套流程完成。
默认完整流程不要跳阶段。只有已有完整匹配表并显式给 `--calibration-dir` 时才复用校准，
仍必须执行新源码的 GPU 检查。旧 CTA/native_cost 表不能用于此方法。

## 4. 正式窗口约束

正式只运行 native、peer_tail 两组；两组 full CUDA Graph、精确 batch1..32、关闭 padding，
初始并发32，各完整1200秒。沿用自然题长、自然 EOS、Think High、temperature0、
max_new_tokens32768 和现有 streaming 客户端，不重复题目、不强行等长输出、不提前停窗口。

正式窗口不启用 P0、阶段事件、逐任务 timeline 或额外轮询。不在窗口内访问
`/tail-coverage` 或 `/budget-coverage`。预热排空后与窗口结束排空后由控制器读取聚合证据。
真实协议、front marker、统计、GPU搜索、计划交换和结束确认成本必须保留。

有限数值差异、生成差异、跨候选 routing/workload 差异、预测误差和吞吐下降全部
`record_only`。不新增或恢复 relative-L2、absolute error、allclose、logprob、token一致率
或性能门限。保留非有限、非法访存、畸形tensor/routing、精确覆盖、来源、协议、编译运行、
graph replay、取消超时与清理检查。

任何硬失败均保留脱敏阶段报告及 lifecycle，停止后续服务对照。不能修改代码或表格伪造成功，
也不能静默回退到原生 down、NCCL、eager、75%预算或关闭 stealing/搜索后仍沿用原方法标识。
若 GPU 暴露实现缺陷，报告具体阶段和脱敏原因，保留私有日志供后续修复。

## 5. 清理与回传

控制器在完成后并发请求四个 `/tail-close`，核对实际 graph reset 与 collective IPC close，
然后关闭 owned container。启动失败或 CUDA context 中止时如实记录未完成的清理，不把进程
被终止解释为 graph 已 reset。

记录程序打印的 `work_directory` 与 `export_directory`。只把后者的白名单目录复制到可写
main clone 的 `reports/server/` 下。逐文件核对 manifest SHA256，并调用
`scripts.run_gpqa_c32._safe` 检查 JSON；不要自行扩展文件清单。

禁止上传原题、token IDs、生成文本、原始路由、激活、权重、逐任务事件/位图、完整日志、
private/ 或 JIT overlay。原始 GPU rank 报告留在私有目录，公开采用控制器生成的缩减 check.json。

回传前安全同步 main，保留现有修改。只 `git add reports/server/<这次新目录>`，
不使用 `git add .`。提交并非强制推送 main，记录回传 commit SHA。

## 6. 最终汇报

请按实际结果报告：

- 执行 commit、回传 commit、GPUQ job、实际四卡 allocation、image ID、公开目录。
- S1/S2/统一执行器/memcheck/measured验证的完整 case 数、四rank状态、graph replay、清理。
- 真正执行的 task/worker/SM 证据；chunk overlap、受控 stealing 与自然观察分开。
- 校准表与策略身份、当前-workload CPU/GPU 选择、预测偏差、regret，以及测量开销。
- 原生和新路径的 shared join 变化、静态 down/通信代价；不能将未隔离因素强行归因。
- 两组实际 full Graph、B32覆盖、四rank一致、无 eager 回退与无窗口内 recapture。
- 4 rank ×43层的 gate/up、tail、planner、protocol commit 调用差分和 replay 对齐。
- 实际计划分布、chunk/down/stolen数量、通信字节；全部同一计划或stealing为零就写零。
- 两组完整1200秒 decode tokens/s、peer_tail/native吞吐比、实际batch分布、请求完成情况、
  数据和补充队列耗尽、sustained_c32_ratio_available、所有容器清理状态。
- 分开列出 source_implemented、gpu_mechanisms_verified、
  full_model_graph_execution_verified、throughput_result_observed；失败停在哪一级就写哪一级。

不要将本方法写成完整 Weave 论文复现，也不要在没有两个完整窗口时给出加速结论。
