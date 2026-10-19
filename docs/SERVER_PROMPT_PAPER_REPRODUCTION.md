# 服务器 Claude Code + DeepSeek V4 Flash：恢复论文机制的首轮验证

请拉取rh2844417786/weave-deepseek-v4-flash最新main，记录完整commit，阅读AGENTS.md和docs/PAPER_REPRODUCTION.md。

目标是验证新 `hopper_persistent_paper_v1` 的真实论文机制集成：同一persistent kernel内的Hopper设备tile、路由驱动SM/chunk选择、去重dispatch、chunk流水、bubble stealing。旧hopper_fp8+NCCL路径只叫operator_candidate，不把旧结果或本地CPU测试称为完整Weave。

## 环境

沿用已有lmsysorg/sglang:v0.5.17、Torch2.11/CUDA13/Python3.12、只读模型/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash、工作盘/srv/weave-work。服务器仅从GitHub取材料，不在线pip/apt/HF下载、不拉镜像。保留现有编译缓存、数据和历史报告。

GPUQ分配4张H100，沿用CUDA_VISIBLE_DEVICES，不硬编码物理卡号。首次建议作业预算12小时，可按校准和编译实际需要延长。不要为了跑完而删除校准桶、减少真实层数或绕过失败检查。

prepared GPQA Main使用/srv/weave-work/gpqa-main-c32/main.jsonl（如实际在别处仅替换绝对路径），SHA256必须为cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888，保留相邻manifest。

## 拉取与执行

先git status --short、git pull --ff-only origin main、git rev-parse HEAD；不reset/clean，保留本地修改。

确认存在experimental/hopper_persistent/megakernel.cu、hopper_tile.cuh及scripts/run_paper_reproduction.py。

在仓库根目录、GPUQ分配的作业内用已有Python3.12执行：

```bash
REPO_ROOT="$(pwd -P)"
python3 scripts/run_paper_reproduction.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --prefill-evidence-dir "$REPO_ROOT/reports/server/c32-prefill-63dac24bf44a" \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --stage-timeout 14400 \
  --execute
```

可先去掉--execute预览。本入口自动按顺序运行机制smoke→真实尺寸校准→43层B32及L0偏斜B128原权重检查→Native与paper persistent各1200秒GPQA。两组full decode graph、客户端并发与服务上限32；paper组prefill和decode都必须是hopper_persistent，成本表必须来自本次新后端校准。

既有prefill-evidence只提供checkpoint/image的历史证明，不替代新内核检查。不能把旧Hopper成本表改名后复用，不能改hash/状态绕过检查，不增加NCCL slot SUM补救或退回hopper_fp8。

## 成功与失败的判断

- 核心验证应证明每SM一个resident CTA、实际SM ID唯一、所有任务/tile守恒。
- 在指定受控案例中，dispatch_compute_overlap、K>1且return_compute_overlap、实际stolen_tiles分别必须被观察到。动态案例由GPU读取实测表选c/K，不能用固定1,1,0冒充。
- 数值误差全部record_only；原Hopper参考覆盖完整batch，独立FP8/BF16 oracle明确只覆盖前两行。不要恢复L2/allclose/token一致率门限。
- 报告和成本表必须绑定同一CUDA/Python源码、镜像、checkpoint、GPU及shape；checkpoint检查必须使用服务将使用的同一成本表。
- 所有GPU检查正常退出，图先reset再清理进程组；保留每rank清理阶段。若编译/TMA/非法访存/非有限/死锁/活动证据不足失败，保留失败报告并停止，不静默降级。
- 真实服务应覆盖四rank全部43层，GPU动态调用计数匹配forward/graph replay，额外routed NCCL slot collective为0。报告实际c/K直方图、偷取和重叠计数；服务阶段某机制未被选中或未观察到时如实写false，不强行改参数制造“完整”结论。

## 汇报与推送

分别汇报：
1. NVCC/PTXAS编译是否成功、资源占用与实际SM驻留。
2. 各机制受控案例和动态路由案例是否通过，实际c/K、stolen、dispatch/return overlap数。
3. 实测成本表桶/候选覆盖及数据来源，efficiency=1、overlap=0是声明的保守常数，不能称为拟合结果。
4. 原权重43层与B128检查、完整参考范围及数值记录。
5. 若服务进入正式窗口：两组各1200秒实际graph重放、B32次数、43层paper路径覆盖、计划直方图、观测机制、decode TPS、TTFT/ITL、请求完成/未完成/失败及数据耗尽状态。
6. 明确区分源码已实现、kernel机制已验证、服务执行已验证和完整论文实验复现。不要因为implementation名字带paper就宣称完整Weave；complete_weave_claim保留false。

成功或失败均保留证据。统一入口最后输出主export_directory及service_result中的服务export_directory（若已生成）。将这两个公开目录复制到可写clone的reports/server/，核验manifest，只暂存这些目录并commit/push main。禁止git add .；不上传原始路由、激活、题目、生成文本、token IDs和完整私有日志。

最后提供执行源码commit、推送commit、GPUQ job、卡号、报告路径及最后完成阶段。未运行性能时明确写未运行，不用旧operator候选的结果替代。
