# 缓存清理失败修复与复测

本轮修复针对每条生成请求前的缓存清理。旧驱动只请求一次 `/flush_cache`，2秒内未成功就记录 `cache_flush_error` 并跳到下一条。锁定的SGLang版本在调度器仍忙时拒绝清理，因此这种处理可能跳过本来可以继续执行的题目。

SGLang源码显示，默认 `timeout=0` 是即时检查；忙状态经HTTP400返回，常见正文为 `Flush cache failed.`。已有服务器汇总没有保存HTTP详情，无法把所有历史失败都确定归为这个原因。

- [锁定版本HTTP入口](https://github.com/sgl-project/sglang/blob/29481685462732237d80d86076d6563e1f658102/python/sglang/srt/entrypoints/http_server.py)
- [锁定版本flush状态处理](https://github.com/sgl-project/sglang/blob/29481685462732237d80d86076d6563e1f658102/python/sglang/srt/managers/scheduler_components/flush_wrapper.py)

驱动现在对已知忙状态和暂时性传输错误重试。每次HTTP请求最多2秒，间隔0.1秒；同一条记录最多等待30秒，并受原有warmup/整组截止时间约束。等待消耗原有预算，只有收到明确的成功回复才提交生成请求。

服务器端仍使用即时flush，避免客户端超时后留下延迟执行的清理操作。未知HTTP错误、无效成功回复或30秒仍未恢复会停止本组；控制器保存部分结果和容器清理记录，并将实验标为失败，不再继续下一组或生成成功比较。

每条公开记录新增 `cache_flush`，包含尝试次数、等待时间、HTTP状态、固定原因分类、响应摘要及最近16次尝试。错误正文前4096字节保存在工作盘私有逐请求JSON中的 `cache_flush_private_errors`，恢复成功后的错误也会保留。这些正文不进入公开导出。

## 服务器复测

CUDA源码摘要仍为 `95e8d6c87ab67514c2b090834d7cc43ec352acfb24f088bb749f8482e5a9d7c0`，本次修改不要求重新做374点校准。复用上一轮通过的core、校准和未加探针的模型验收即可复查同一组SWE-Bench请求。

在现有checkout拉取main。以下Python命令放入GPUQ实际授予四卡的作业中执行；保留调度器设置的 `CUDA_VISIBLE_DEVICES`。准备文件通过上一轮记录的SHA256定位，避免猜测文件名。

```bash
git pull --ff-only origin main
python3 - <<'PY'
import hashlib
from pathlib import Path
import subprocess
import sys

prepared = Path('/srv/weave-work/swe-bench-prepared')
expected = 'd3928f28e645ae716f0e4a63d06b8fa1a27103b2d8d602975f7384ada289918a'
matches = [p for p in prepared.glob('*.jsonl')
           if hashlib.sha256(p.read_bytes()).hexdigest() == expected]
if len(matches) != 1:
    raise SystemExit('没有唯一匹配上一轮SHA256的prepared JSONL，请核对原作业记录路径')
subprocess.run([
    sys.executable, 'scripts/run_full_experiment.py',
    '--work-root', '/srv/weave-work',
    '--export-root', '/srv/weave-work/public-reports',
    '--records', str(matches[0]),
    '--core-run-dir', '/srv/weave-work/cuda-functional.2349166e7413',
    '--calibration-dir', '/srv/weave-work/calibration.2349166e7413',
    '--model-acceptance-dir', '/srv/weave-work/model-acceptance-963da6f9a789',
    '--calibration-kernel-timeout-seconds', '120',
], check=True)
PY
```

沿用当前服务器提交中的9条输入和每组600秒；不安装依赖、不拉镜像、不下载模型。只回传自动生成的公开导出目录。检查重试是否恢复、是否仍有 `cache_flush_failed`、两组完成记录的顺序与数量，并保留manifest。

若同轮收集之前已批准的设备阶段数据，去掉上面 `--model-acceptance-dir` 及其值，增加 `--execution-profile`；控制器会重新做带探针的模型验收。带探针的速度按诊断数据处理。

## 验证范围

本地真实localhost HTTP回归覆盖忙后恢复、永久忙、整组截止、阻塞读取、未知HTTP错误、无效成功回复、私有错误保存和warmup失败停止。原有流式请求的取消及禁止截止后发送机制继续适用。

服务器 `5ea648f` 的数据读取通用化导致部分原GPQA来源/顺序校验测试在修复前已失败。本次保留这些测试并单独报告结果，不把全套测试描述为全绿；缓存修复不改变该数据读取行为。
