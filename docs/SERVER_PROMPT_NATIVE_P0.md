# 服务器 Claude Code + DeepSeek V4 Flash 执行 Prompt

请在 `rh2844417786/weave-deepseek-v4-flash` 拉取最新 main，记录完整 commit，阅读 `AGENTS.md` 和 `docs/NATIVE_P0.md`，执行本轮原生 SGLang P0 诊断。

目标是确定原生 DeepSeek-V4-Flash 的实际 MoE 执行路径、routed/shared 时间关系和 Marlin 启动资源，为后续提高原生吞吐选择优化方向。此次只运行原生组，不调整 CTA/SM 配额，不运行旧 Hopper/persistent 对照，也不宣称已提高吞吐。

## 环境和执行

1. 保留现有模型、prepared 数据、编译缓存与工作目录。使用已有 `lmsysorg/sglang:v0.5.17`、Torch2.11/CUDA13/Python3.12；只读模型为 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`。服务器只能从 GitHub 取材料，不在线 pip/apt/HF 下载，不拉镜像，不临时替换依赖。
2. 查找已有 GPQA Main prepared `main.jsonl` 及相邻 manifest。文件 SHA256 必须为 `cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888`；不要上传准备文件或手改 hash。若缺失，按既有 GPQA 文档用固定 GitHub 来源及已有镜像离线准备。
3. 使用 GPUQ 申请4张 H100，沿用分配的 `CUDA_VISIBLE_DEVICES`，不写死物理卡号。首次运行总预算6小时，可按需要延长；正式负载窗口固定1200秒，其他时间用于检查、模型加载、图捕获和清理。
4. 在 GPUQ 作业内，从仓库根目录用已有 Python3.12 执行以下命令。替换 prepared 路径为实际存在且哈希一致的位置，WORK_ROOT 必须可写。

```bash
python3 scripts/run_native_p0.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --setup-timeout 7200 --warmup-timeout 960 --drain-timeout 60 \
  --graph-gate-timeout 7200 --graph-cleanup-timeout 120 \
  --execute
```

入口已固定：原生 backend、full CUDA Graph、精确1..32 bucket、关闭padding、客户端并发32和max-running-requests32、TP4/EP4、Think High、temperature0、max_new_tokens32768、正常EOS。保留自然题长和数据耗尽标记，不循环题目或填充请求。Batch32只用于本轮控制变量，不是后续优化限制。

脚本先跑四rank双流事件/CUPTI smoke，再跑原生模型；不要绕过 smoke、来源核验或覆盖检查，也不要另开服务/请求干扰。错误时保留结果和具体阶段，不用在线安装尝试修补环境。所有数值差异只记录，不新增精度或token一致率门限。

## 阅读结果

- 检查 `experiment.json` 最终状态、所有拥有的容器是否清理，以及四rank smoke是否完成图释放。
- 检查 `native/graph-check.json`：full graph实际 replay、B32覆盖，而不只是启动参数开启。
- 阅读 `native-p0-summary.json`：实际捕获路径、shared_tp1；43层每层gate-up/激活/down、shared、分组、归约与all-reduce的时间区间；两分支重叠与暴露尾部；每层分组统计；四rank整图跨度。
- 确认每rank实际86个Marlin kernel及grid/block/threads/registers/shared memory/tile记录，与Python调用元数据匹配。缺失字段不能自行估算或把“启用”当作执行证据。
- 使用同一样本同一critical rank的阶段值。不要把不同rank各阶段最大值拼接，不把重叠分支时间相加，不把all-reduce区间直接解释成纯通信带宽。
- 可以记录本轮带采样的吞吐及请求统计，但不得与历史未采样吞吐计算正式加速比。暴露尾部也不能单独证明CTA配额能带来收益。

## 回传

成功或失败都回传脚本打印的 `export_directory`。将该白名单公开目录复制至可写clone的 `reports/server/`，核对manifest文件SHA256，只暂存新公开目录，commit/push main。禁止 `git add .`，禁止上传原始Chrome trace、private目录、完整日志、题目、token IDs、生成文本、路由数组或激活。

最后报告：推送commit、GPUQ job、实际GPU分配、镜像ID、运行状态、公开目录、图是否实际跑通、采样覆盖、主要阶段和逐层暴露尾部、是否存在rank不平衡。如果失败，给出具体阶段与经过脱敏的错误原因；不要把未完成诊断写成已定位瓶颈。
