# 服务器 Claude Code + DeepSeek V4 Flash：CUDA Graph C32

请拉取 `rh2844417786/weave-deepseek-v4-flash` 的 main，记录完整 commit，阅读 `AGENTS.md`、`docs/HOPPER_GRAPH_C32.md` 和 `docs/GPQA_C32_COMPARISON.md`。本轮执行 Native SGLang 与 Weave Hopper prefill+decode 的 **full decode CUDA Graph 对照**：两组并发均 32，正式窗口各 1200 秒。

## 环境与数据

- 使用已有 `lmsysorg/sglang:v0.5.17`（Torch 2.11/CUDA 13/Python 3.12），不拉镜像、不在线 pip/apt/HF 下载。
- 只读模型 `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash`。保留 `/srv/weave-work` 内的编译缓存、数据缓存和旧结果。
- 通过现有 GPUQ 申请同一组 4 张 H100，保留调度器提供的 `CUDA_VISIBLE_DEVICES`，不硬编码物理编号。作业预算建议先给 6 小时；排队时间与作业时间分开，按需要延长启动/编译预算。
- 使用已有 prepared GPQA Main；内容 SHA256 必须为 `cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888`，相邻 manifest 完整且与本地模型匹配。如尚未准备，仅按 `docs/SERVER_PROMPT_GPQA_C32.md` 的“数据准备”节取得固定 GitHub 数据并在已有镜像离线编码，不执行该文档的 eager 作业命令。
- prepared 数据、token IDs、题目、生成内容和私有日志均不得推送 GitHub。

## 执行

在 GPUQ 分配的作业内，从仓库根目录用已有 Python 3.12 运行下列命令。`--records` 如需替换，仅换成已验证 prepared 文件的绝对路径。

```bash
REPO_ROOT="$(pwd -P)"
python3 scripts/run_gpqa_c32.py \
  --work-root /srv/weave-work \
  --records /srv/weave-work/gpqa-main-c32/main.jsonl \
  --prefill-evidence-dir "$REPO_ROOT/reports/server/c32-prefill-63dac24bf44a" \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --decode-graph full \
  --graph-gate-timeout 7200 \
  --setup-timeout 7200 --warmup-timeout 960 --drain-timeout 60 \
  --execute
```

控制器会设置两组相同的精确 batch 1–32 图配置，关闭 graph padding，保留 eager prefill，并在加载模型前运行新的四卡 graph smoke。它自动使用分配的 GPU、已有镜像和 localhost/inter-rank 网络。不要另起请求、开启 profiler 或改变图 bucket、并发、Think High、EOS、输出上限及正式窗口。

预热、加载、编译和图捕获不计入每组 1200 秒。先确认两组全部 rank 完成 32 个 bucket 的捕获，再依次测量。若出现编译、NCCL、非法地址、非有限输出、源码不匹配、内存不足或缺失覆盖，保留真实错误和公开失败证据；不要降级 eager 后继续冒充 graph，也不要修改 hash 绕过检查。有限数值差异、生成内容差异只记录，不设置误差门限。

## 汇报和推送

核对以下内容并汇报：

1. 完整 commit、GPUQ job、分配卡、镜像 ID、运行/清理状态和公开目录。
2. 四 rank graph smoke 的 64 次捕获与 38 次重放；两个 handle、动态输入/路由及 prefill 后重放是否完整。
3. 两组正式窗口真实 graph 重放次数和 batch 分布，实际 B32 次数；raw/captured 是否一致，有无 eager decode 回退、重捕获或失败重放。
4. Weave 43 层设备 decode 次数、实际 slot collective 次数、prefill 单独计数是否满足报告检查。
5. 两组窗口 decode tokens/s、TTFT/可用 ITL、token 数、完成/未完成/未尝试题数、数据是否耗尽、可否报告持续 C32 加速比。若补充请求耗尽，不循环题目填时间；保留观测吞吐与耗尽标记。

成功或失败均保留证据。只将脚本打印的 `export_directory` 复制到可写 clone 的 `reports/server/`，验证导出 manifest 的 SHA256，然后仅暂存该目录、commit/push main。禁止 `git add .`，不上传完整日志或私有请求数据。最后给出推送 commit 与报告路径；不要把 CPU 测试或“参数开启”当作 GPU graph 成功。
