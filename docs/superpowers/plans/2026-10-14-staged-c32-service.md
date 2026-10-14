# Staged C32 service implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development task-by-task with independent reviews. User already approved implementation and GitHub delivery.

**Goal:** Serve decode through direct tile staged experts and measure comparable C32 output throughput.

**Architecture:** GPU online grouping feeds existing staged math. Immutable weight binding and fixed workspaces remove hot-path inspection. Slot contributions are summed across the TP device group before deterministic BF16 origin assembly; native outer shared/TP handling remains intact. An offline streaming client measures the same C32 load for native eager, cached persistent eager and staged eager.

**Tech Stack:** Existing CUDA13/Torch2.11/SGLang0.5.17 image, Python3.12, stdlib host drivers and existing aiohttp client dependency.

**Spec:** docs/superpowers/specs/2026-10-14-staged-c32-service-design.md

## Global constraints

- Original mixed weights read-only, four dynamic H100s, TP4/EP4, all43layers, no new downloads/packages.
- No model/route changes, no native expert fallback, no graph service integration in this batch.
- Keep P1 frozen API and existing persistent mode. Current source needs fresh GPU gates.
- Public output is counts/timings/hashes only. Raw text/token IDs/tensors/logs remain private.
- All dependent server steps stop on failure; cleanup owns only launched containers.
- Local CPU tests cannot establish GPU correctness or throughput.

## Task1: online CUDA grouping and assembly

Files: csrc/staged_compute.cu, new csrc/online_staged.h, tests/cpp/test_online_staged.cpp, tests/cpu/test_online_staged.py, tests/gpu/check_online_staged.py.

Interfaces on existing extension.StagedWorkspace(rank,ep,max_batch,h,i,experts,topk):
`bind_online_weights(raw) -> StagedBoundWeights`; `run_online(x,ids,routing,bound,clamp=10) -> None`; `finalize_online(batch) -> None`.
Properties: `online_slot_outputs` FP32[max_batch*topk,H], `online_output` BF16[max_batch,H], `online_group_meta`, `online_group_slots`, `online_group_count`. Bound handle owns tensors/workspace and rejects foreign workspace. Runtime caches prefix views before serving. Existing frozen run/bind API is unchanged.

- [x] Write CPU/C++ bound/group capacity and deterministic grouping reference tests before implementation; run them and observe missing API/helper failure.
- [x] Implement capacity-bound group planner and online CUDA group/reset, existing direct gate/up/down, routed slot fill, final origin assembly; expose stable views and bound handles.
- [x] GPU gate uses torchrun4, batches1/7/17/32, changing routing and empty local rank, compares online metadata with build_staged_groups and assembled output with existing oracle/persistent on all43 raw layers. Use Gloo CPU+NCCL device groups and guarded failure agreement. JSON output is atomic0664.
- [x] Run available CPU tests and syntax check. Report unrun GPU gate explicitly. Independent source review.

## Task2: adapter and launcher

Files: new src/weave/staged_decode.py, src/weave/sglang_backend.py, src/weave/sglang_ext/models/deepseek_v4.py, scripts/run_native_ep.py, focused CPU tests.

Consumes Task1 API. `StagedDecodeRuntime` creates one maxB32 workspace, caches output/slot prefix views, creates per-layer online bound handles; `run(x,ids,routing,bound,clamp)` calls run_online, torch.distributed.all_reduce on slot prefix using existing TP device_group, then finalize_online; returns cached BF16 origin-masked output. Online mode only decode1..32; existing cached persistent handles prefill. Add `--decode-implementation persistent|staged_direct`, envWEAVE_DECODE_IMPLEMENTATION; require tiled/hot-cache/eager/fixed(1,1,0), no measured profile/execution-profile/capture combination for staged.

- [x] Add failing mock orchestration tests for one slot collective per call, exact active prefix, cache/bind counts and invalid configurations.
- [x] Integrate explicit branch after native router unpack; keep origin masking only on persistent branch because staged already masks. Preserve shared handling in audited upstream. Add source identity closure if new runtime dependency requires it.
- [x] Add per-layer implementation counts and model CPU batch histogram without GPU timing probe; coverage remains idle-only. Reject unsupported modes rather than silently switching decode backend.
- [x] Run relevant CPU tests, review cached state lifetime and numeric boundaries, independent review.

## Task3: C32 client and evidence metrics

Files: new src/weave/c32_serving.py, scripts/bench_c32_serving.py, tests/cpu/test_c32_serving.py.

Client uses existing aiohttp in image, local JSONL validated by validate_dataset, /generate streaming with text and sampling_params(max_new_tokens1024,temperature0,ignore_eosTrue). Fixed32 workers consume same deterministic request order. Warmup outside window; shared monotonic start/end; client records cumulative completion_tokens deltas received before deadline, excluding first token per request for decode TPS. Keep aggregate token counters with unfinished requests, completed/failed/unfinished, TTFT/ITL summaries, clear denominator. Missing/regressing/noninteger token counts or early exhausted load fail measurement. Publish progress at bounded cadence, not per token disk writes.

CLI `--dataset --base-url --output-dir --seconds --concurrency 32 --output-tokens 1024 --execute` runs inside image. Final `client-summary.json` public schema, `client-private.json` only private if needed. Separate acceptance mode may use same dataset32 distinct requests and8token nonstream output proof, with raw responses private and hashes/IDs commitments public; do not gate logprob drift. Parent driver will orchestrate cross-group proof comparison and lifecycle.

- [x] Write failing tests for chunk splitting, cumulative counts, deadline/incomplete inclusion, first-token exclusion, malformed/regressing counts, low concurrency/early exhaustion and redaction.
- [x] Implement streaming parser/metrics/client with local-only HTTP, no downloads. Pin workload SHA/deterministic ordering and generation settings in output.
- [x] Test with fake/in-process HTTP streams, including split SSE and cancellation. Independent review.

## Task4: controller, gates and handoff

Files: scripts/run_c32_comparison.py, scripts/run_online_staged_gate.py if needed, tests/cpu/test_c32_comparison.py, docs/STAGED_C32_SERVICE.md, docs/SERVER_PROMPT_STAGED_C32.md, validation report/README.

- [x] Build fresh-core+online-staged gate driver using existing check_runtime, OwnedChild, image inspect and allocation helpers. Bind source/model/image evidence; safe fresh run directories and bounded setup/run/cleanup.
- [x] Sequential native eager, persistent-cached eager, staged-direct eager withsame image/model/dataset/config and measurement window. Run acceptance before throughput and require all43 staged decode coverage. Snapshot before/after only at idle; collect batch evidence and launch/runtime metadata. Incomplete/failed groups do not produce a valid ratio.
- [x] Export explicit allowlisted public files and SHA256 manifest, source identities, numerical/acceptance proof, comparable TPS summary and diagnostics. Keep raw outputs private.
- [x] Write tests for launch matrix, no graph mismatch, guard failures, exact comparison identities, no incomplete-only token bias, public privacy and owned cleanup.
- [x] Run focused/fullCPU checks accounting for known baseline7failures3errors, source review and final integration review. Pushmain and provide server prompt with configurable GPUQ budgets and three sequential groups. No GPU performance claim until results return.

## Local completion record

Implementation and independent reviews complete.49 focused tests pass; full268-test CPU suite retains the pre-existing7 failure/3 error subcases. GPU checks, service acceptance and throughput remain server tasks. Delivery source identities are in reports/staged_c32_service_validation.json.
