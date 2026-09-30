# GPUQ 取消 ≠ 容器停止：bring-up launcher 的 SIGTERM 清理缺口

2026-09-30 于本机（8×H100）实测。结论：`gpuq cancel`（以及作业超时、daemon 停止）只会终止 GPUQ 托管的进程组；由 launcher 通过 `docker run` 启动的容器**不会**随之退出，物理卡会继续被占用，直到有人显式 `docker stop`。任何在 GPUQ 作业里 spawn `docker run` 的 launcher 都必须自行处理 SIGTERM 清理。

## 现象（job-20260930-143712-a465d4ee）

- adapter bring-up 作业本身运行正常：14:50:57Z 服务 ready，短生成 `POST /generate 200 OK`（证据见 `work/native-ep-26fad58797ce4c9ebdbd3dbd6b926310/server.log`）。
- ~15:15Z 执行 `gpuq cancel job-20260930-143712-a465d4ee`（用户显式要求提前释放）。作业进入 `cancelled`（exit_code `-15`，error `cancel requested`）。
- 但 20 秒后检查：容器 `weave-native-ep-1941abb0dc4a` 仍是 `Up 42 minutes`；`nvidia-smi` 显示 GPU 4–7 仍占 `68072 / 76326 / 76326 / 75030 MiB`，4 个 `sglang::scheduler_TP0_EP0 … TP3_EP3` 进程继续驻留。
- 只有显式 `docker stop weave-native-ep-1941abb0dc4a` 之后，四张卡显存才回落到 4 MiB。

## 机制

- GPUQ 的终止路径（`src/gpuq/scheduler.py::_terminate`）对托管命令的**进程组**发 `SIGTERM`，10 秒后再 `SIGKILL`。
- 托管命令是 `python3 scripts/run_native_ep.py …`，它用 `subprocess.Popen` 启动 `docker run` 前台阻塞。容器内进程的父进程是 containerd-shim，**不属于该进程组**；`docker run` CLI 被 `SIGKILL` 后没有任何东西会去停止容器（`SIGKILL` 也不可被转发）。
- `run_native_ep.py` 只处理 `KeyboardInterrupt`（会执行 `docker stop`），**不处理 `SIGTERM`**，因此 GPUQ 取消时没有清理路径。

## 影响

- 取消/超时/daemon 重启后，GPUQ 的 advisory 锁会释放，但卡上仍有容器进程；此后调度器把这几张卡判为 `external_busy`（有进程 / 显存高于空闲阈值），排队中的作业会一直等待，直到人工清理。
- 对锁定 4 卡的作业尤其严重：本机 4–7 只有一组，一张泄漏的容器就能把整条 4 卡队列堵死。

## 建议修复（launcher 侧）

在 `scripts/run_native_ep.py` 注册 `SIGTERM`/`SIGINT` 处理器，执行与现有 `KeyboardInterrupt` 分支相同的清理后再退出（`name` 在启动前生成，容器尚不存在时 `docker stop` 失败也无害）：

```python
import signal

def _stop_container(_signum, _frame):
    subprocess.run(['docker', 'stop', '--timeout', '10', name],
                   capture_output=True, timeout=30)
    raise SystemExit(143)

signal.signal(signal.SIGTERM, _stop_container)
signal.signal(signal.SIGINT, _stop_container)
```

同一约定适用于任何在 GPUQ 作业内 `docker run` 的 launcher：`scripts/run_cuda_functional_gpuq.sh` 同样是「docker run 前台阻塞」结构，作业正常运行会自然退出，但**被取消时也会留下容器**。

## 验证方法

1. 提交一个 bring-up 作业，等服务 ready；
2. `gpuq cancel <job>`；
3. 15 秒内 `docker ps | grep -E 'weave-native-ep|cuda-functional'` 应为空，`nvidia-smi -i 4,5,6,7 --query-gpu=memory.used` 应回到接近 0；
4. GPUQ 队列中后续 4 卡作业应能立即被调度，而不是报 `external_busy`。

## 运维临时办法（修复前）

`gpuq cancel`、作业超时或 daemon 停止后，检查并清理残留容器：

```bash
docker ps | grep -E 'weave-native-ep|cuda-functional'
docker stop --timeout 20 <容器名>
```

否则 GPUQ 会一直把对应 GPU 报为 `external_busy`。
