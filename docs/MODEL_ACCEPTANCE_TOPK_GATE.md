# 模型验收门失败分析：weave_eager 首 token top-k 支撑集与尾部数值漂移

2026-10-02，作业 `job-20261002-092219-69d97c32`（`scripts/run_model_acceptance.py`，GPU 4–7，运行 32 分 54 秒）。四步中 **checkpoint ✅、native ✅、weave_eager ❌、weave_graph 未运行**。

服务侧完全正常：两级运行契约 `runtime_status: passed`、两次 `POST /generate 200 OK`、无 traceback；容器生命周期（含 SIGTERM 强制清理）全部达标。失败点是**数值验收门**，不是 bring-up 或崩溃。

## 门的确切语义

`src/weave/service_acceptance.py::compare_first_token`（`logprob_atol=.1`）按顺序做两级判定：

1. **支撑集必须完全相同**（刻意严格，注释写明不做静默交集）：
   ```python
   # An altered top-k support is evidence of a potentially changed next-token
   # distribution. Fail explicitly; do not silently compare the intersection.
   if set(atop) != set(btop) or not atop:
       raise ValueError('first token top-logprob support differs or is empty')
   ```
2. 公共支撑（含首 token 自身 logprob）上 `|Δlogprob| ≤ logprob_atol = 0.1`。

本次在**第 1 级**抛出：`ValueError: first token top-logprob support differs or is empty`。

## 离线比对结果（`native-request.json` vs `weave_eager-request.json`）

首 token 完全一致，且其 logprob 也一致：

| | 首 token | 首 token logprob | Δ |
|---|---|---|---|
| native | `2581` | `-0.0000` | — |
| weave_eager | `2581` | `-0.0000` | `0.0000` |

top-10 列表（logprob, token id）：

| 名次 | native | weave_eager |
|---:|---|---|
| 1 | -0.000 `2581` | -0.000 `2581` |
| 2 | -13.375 `10318` | -13.125 `10318` |
| 3 | -14.250 `671` | -13.875 `671` |
| 4 | -14.750 `5718` | -14.875 `5718` |
| 5 | -15.875 `1350` | -15.625 `1350` |
| 6 | -17.125 `201` | -16.625 `201` |
| 7 | -17.875 `11428` | -17.750 `11428` |
| 8 | -18.125 `666` | **-18.250 `92341`** |
| 9 | **-18.625 `9832`** | **-18.250 `666`** |
| 10 | **-18.625 `2337`** | **-18.625 `1124`** |

- 仅 native 有：`9832`(-18.625)、`2337`(-18.625)
- 仅 weave_eager 有：`92341`(-18.250)、`1124`(-18.625)
- 交集 8 个，公共支撑上的 |Δlogprob|（按差值降序）：

| token | native | weave_eager | \|Δ\| |
|---|---|---|---|
| `201` | -17.125 | -16.625 | **0.500** |
| `671` | -14.250 | -13.875 | **0.375** |
| `1350` | -15.875 | -15.625 | **0.250** |
| `10318` | -13.375 | -13.125 | **0.250** |
| `666` | -18.125 | -18.250 | **0.125** |
| `5718` | -14.750 | -14.875 | **0.125** |

**关键结论**：交集上最大 Δ = 0.500，**8 个公共 token 中有 7 个超过门的 `atol=0.1`**。因此本次失败**不能只归因于支撑集边界翻转**——即使把支撑集判定放宽为近并列容差，第 2 级数值判定仍然会失败。两处都需要处理：或者降低端到端数值差，或者重新论证门限。

## 事实与推测（推断部分尚未验证）

**已确证的事实**
- 首 token 预测一致，首 token logprob 一致（Δ=0）。
- 差值集中在深尾部（logprob < -13 的 token），且随名次加深总体变大；weave_eager 的 logprob 系统性地比 native 略高（更接近 0）。
- 服务与契约健康：`runtime_status: passed`（native/adapter 两级）、`POST /generate 200 OK`、无 traceback、checkpoint 门与 native 门均通过。

**待验证的推测**（不要当作结论）
- 尾部近常数偏移可能来自 **top-k logprob 的报账口径差异**（例如是否对 top-k 子集重新归一化）——这是最便宜可验证的假设：单后端自比或打印原始 softmax 尾部即可判定。
- 也可能是真实数值差：标量 MXFP4 专家路径 vs Marlin fp8 路径、MoE 归约顺序、fp8 KV cache 舍入等。
- 定位顺序建议：先排除报账口径（便宜）→ 再做逐层/单层差分（kernel 数值）。

## 建议

1. **门的语义**：第 1 级支撑集判定建议按"与第 k 名 logprob 相差 < ε 的近并列 token 允许换位"实现，或改为支持集无关度量（交集覆盖 + 全变差 / KL 上界），避免 k 截断处的近并列翻转直接判死。
2. **门限论证**：`logprob_atol=0.1` 需要针对**端到端 MoE** 给出实测基线（当前数据表明端到端尾部漂移可达 0.5），否则该门会长期失败。
3. **数值来源排查**：先验证报账口径假设，再查专家/kernel 数值。
4. **覆盖缺口**：`weave_graph`（decode graph full）本轮未运行，修复后必须补跑。

## 本次顺带验证：容器生命周期（SIGTERM 强制清理）✅

| 容器 | 信号 | exit_code | cleanup_status |
|---|---|---|---|
| `weave-cuda-functional-d4773fb0c91f` | null | 0 | `not_required`（干净退出） |
| `weave-native-ep-84f87f727345`（native） | 15 | 143 | `absent` |
| `weave-native-ep-377e4dacf8c0`（weave_eager） | 15 | 143 | `absent` |

两个服务都实际走了强制清理链路：`docker stop --timeout 2` 超时 → `docker rm --force` 成功 → `docker inspect` 返回 `no such object`；作业结束 `docker ps -a` 无任何 `weave-*` 容器。**「取消/终止后容器残留占用 GPU」的缺口修复已被真实作业端到端验证。**

## 复现与证据

```bash
# 复现本次验收（需 GPUQ 分配 4,5,6,7）
python3 scripts/run_model_acceptance.py
```

失败由门抛出，不产生额外日志：`weave_eager-launcher.log` 只有一行（指向 `server.log`），异常在验收驱动内抛出。

证据目录：`work/model-acceptance-d4773fb0c91f/`

- `acceptance.json`：总状态、`steps`、`server_runs`、`error`
- `checkpoint_identity.json`：48 个文件哈希 + `verified_stats`（`sha256=237f6543…`）
- `native-request.json` / `weave_eager-request.json`：两份服务响应证据（本文档比对的数据源）
- `core-launcher.log` / `native-launcher.log` / `weave_eager-launcher.log`

离线比对（本文档数据来源）：

```python
import json
n = json.load(open('native-request.json'))['response']['meta_info']
w = json.load(open('weave_eager-request.json'))['response']['meta_info']
atop = {int(v[1]): float(v[0]) for v in n['output_top_logprobs'][0]}
btop = {int(v[1]): float(v[0]) for v in w['output_top_logprobs'][0]}
print('only native:', sorted(set(atop) - set(btop)))
print('only weave :', sorted(set(btop) - set(atop)))
print('max delta  :', max(abs(atop[k] - btop[k]) for k in set(atop) & set(btop)))
```
