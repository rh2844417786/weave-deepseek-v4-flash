# Hopper EP4 serving implementation plan

Use `superpowers:subagent-driven-development` with task and final reviews. The user approved implementation and GitHub handoff. Work in the primary project on `codex/hopper-ep4-serving`; preserve the unrelated untracked handoff file.

**Spec:** [Hopper serving design](../specs/2026-10-17-hopper-serving-design.md)

## Constraints

Original mixed precision checkpoint, EP4, H100, all 43 layers, scheduler-assigned four GPUs. Existing Torch 2.11/CUDA 13/Triton 3.6/SGLang 0.5.17 image; no downloads. Candidate and BF16 CUDA remain unchanged. No native fallback, numerical gates or public raw content. Use fresh output directories and owned cleanup.

## Task 1: Adapter and finalizer

- Add `hopper_decode.py` and `hopper_finalize.py`.
- Extend configuration, adapter and launcher with explicit `hopper_fp8` mode.
- API: `HopperDecodeRuntime(peer_runtime, device_group, experts=256)`, `bind_weights(raw)`, `run(x, ids, routing, bound, clamp)`.
- Preserve prefill; compute full replicated input, SUM active slots, finalize ordered slots into cached BF16 origin-masked output.
- Record candidate/assembly identity, resources and actual per-layer calls.
- Test order, caching, failure propagation, unsupported modes and CPU-safe imports; independent review.

## Task 2: Suite, controller and evidence

- Add `hopper_serving_evidence.py`; extend existing C32 controller and summaries.
- CLI: `--comparison-suite hopper --hopper-evidence-dir <complete public bundle>`.
- Validate immutable 29-file, 1376-point schema2 replay; current native/model/image/tokenizer identity; retain historical Python identity.
- New suite uses assembly smoke and record-only 32×8 generation comparisons.
- Validate complete actual Hopper coverage, B32 and exact collective count.
- Compare native, BF16 staged and Hopper eager under the same load/timer.
- Export public proofs and statistics only; preserve legacy default suite; independent review.

## Task 3: Four-rank assembly smoke

- Add `tests/gpu/check_hopper_serving.py --output-dir ...` for torchrun4.
- Actual runtime with H128/I128/E32/top6 raw MXFP4/E8M0 weights.
- B1/7/17/32/7/1, empty ranks, changed inputs, stable buffers and retained weights.
- Snapshot actual local slots before NCCL; exact ordered reference checks deterministic assembly only.
- Adversarial known-slot Triton fixture and atomic 0664 public reports.
- CPU contract tests, syntax checks and independent review; GPU execution deferred to server.

## Task 4: Integration and handoff

- Bind new modules in source closures and persist generation record-only scope in AGENTS.
- Audit native outer shared/scaling/reduction boundaries.
- Run focused and full CPU suites; distinguish known baseline failures.
- Write design/runbook/server prompt and local validation record.
- Review whole branch, resolve findings, commit and push main.
- Server executes the new suite and returns public evidence; no end-to-end speed claim before that return.
