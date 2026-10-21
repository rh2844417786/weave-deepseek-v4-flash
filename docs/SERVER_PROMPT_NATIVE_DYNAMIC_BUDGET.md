# 服务器执行 Prompt：原生动态 CTA 预算

请在 `rh2844417786/weave-deepseek-v4-flash` 拉取本轮交付 commit（记录完整 SHA，后续运行期间不要更新代码），阅读 `AGENTS.md` 与 `docs/NATIVE_DYNAMIC_BUDGET.md`，完成 `native_marlin_dynamic_cta_v1` 的 GPU 验证与原生 SGLang 吞吐对照。2026-10-21 已批准实现和对照；本轮直接使用动态预算代码，不再增加 P0 或其他诊断实验。

1. 保留原始只读 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`、已有 `lmsysorg/sglang:v0.5.17`、Python3.12/Torch2.11/CUDA13、prepared GPQA、既有缓存与所有历史结果。服务器只从 GitHub 取材料；不在线 pip/apt/HF 下载，不拉取镜像。缺少文件或依赖时记录错误，不能自动换环境。
2. 在 GPUQ 中申请四张 H100，使用作业分配的 `CUDA_VISIBLE_DEVICES`，不写死物理卡号。预留至少8小时作业时限：包含一次 GPU 编译/验证、两次模型加载与图捕获、两组各1200秒、预热及清理。需要延长时延长外层作业时限，不改1200秒测量窗口。
3. 在仓库根目录先执行 `docs/NATIVE_DYNAMIC_BUDGET.md` 的“GPU 验证命令”完整代码块，使用现有可写 `WORK_ROOT`。它创建新目录并拥有/清理 Docker 容器。确认四份 `native-budget-gpu-rank*.json` 都完成 case coverage 和 cleanup；保留 actual adaptive/full/native 的数值差异记录。有限误差不设阈值。若编译、运行、非有限、内存、routing、source、coverage 或 cleanup 失败，回传脱敏证据，停止服务对照。
4. 查找已有 GPQA Main prepared `main.jsonl` 及相邻 manifest。记录文件 SHA256 必须为 `cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888`；缺少时按既有 GPQA 文档从固定 GitHub 来源获取并校验，在已有镜像离线准备。不要修改 manifest 以迎合检查，也不要填充、重复或拼接题目。
5. GPU 验证通过后，在同一 GPUQ 分配内执行下面命令。将 records 替换为实际已核验路径；控制器自动生成新的实验、服务及导出目录。

```bash
python3 scripts/run_native_budget.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --native-budget-target-percent 75 \
  --setup-timeout 7200 --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

两组是原始 `native` 和 `adaptive`；两组 prefill 均为原始 native，使用相同 full CUDA Graph、精确1..32 capture、无 padding、原始 mixed precision/MXFP4、TP4/EP4、自然 GPQA Think High、temperature0、max_new_tokens32768 和正常 EOS。初始并发32只是本轮控制变量。`full` 是独立补丁控制模式，不添加第三组服务实验。

不要在1200秒窗口内请求 `/budget-coverage`；该端点会同步 GPU 并复制计数。控制器仅在预热排空后、发布开始标记前，以及窗口结束并排空后读取它。不要启动额外请求或 profiler 干扰测量。

核查以下结果：

- 两组 `graph-check.json` 必须记录实际 graph replay、B32覆盖、四 rank 一致性及窗口内无 eager decode 回退。
- `adaptive/budget-check.json` 必须覆盖四 rank ×43层 ×gate/up、down；每项调用差分等于全部 batch 的实际 replay 总次数，包含空本地调用。报告实际桶直方图和 `rows_with_budget_variation`。观察到零变化就写零，不能把 enabled 标记写成动态切换证据。
- `comparison.json` 报告完整1200秒实际 decode tokens/s 和 `adaptive_vs_native_whole_window_ratio`，保留耗尽与未完成请求统计。若 `sustained_c32_ratio_available=false`，明确不具备持续32并发比值解释。报告不评估答案准确率或 token 内容一致率。
- 成功和失败都检查拥有的服务/客户端容器已停止；GPU 独立测试还必须完成图释放。数值与 token 差异为 record_only，保留有效性、来源和覆盖检查。

将控制器打印的 `export_directory` 白名单目录复制到可写 clone 的 `reports/server/`，逐文件核对 manifest SHA256。GPU 独立测试只回传四份聚合 rank JSON 与 lifecycle，按文档执行公共字段检查并生成 SHA256 manifest。只暂存这两个新公开目录，commit/push；不要使用 `git add .`。禁止上传原题、tokens、生成文本、原始路由、激活、权重、完整日志、private/ 或 JIT overlay。失败也回传已存在的脱敏证据并保持 failed 状态。

最后报告执行 commit、回传 commit、GPUQ job、实际分配、镜像ID、GPU验证状态、两组图和43层计数覆盖、预算变化是否实际出现、两组完整窗口吞吐和比值、数据耗尽情况、所有清理状态与公开目录。若失败，给出具体阶段和脱敏原因。没有真实服务结果时不宣称吞吐提高；本方法不标作完整 Weave 论文复现。
