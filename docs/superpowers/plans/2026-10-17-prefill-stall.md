# Prefill stall diagnostic implementation plan

Use superpowers:subagent-driven-development; user already authorized implementation and push.

**Goal:** Separate first-token waiting, persistent prefill and decode output stalls on matched C32 runs.
**Spec:** docs/superpowers/specs/2026-10-17-prefill-stall-design.md
**Architecture:** shared source-pinned model/request hooks, bounded CUDA/host timeline, anonymous client observations, existing controller and offline aggregate report.
**Constraints:** Existing0.5.17 image/local mixed precision model; GitHub only; EP4 scheduler allocation; no native CUDA or schedule changes; all numerical/generation differences record-only; no raw data public; no formal TPS claims from diagnostics.

## Task1 server collector and hooks

Files: new src/weave/stall_timeline.py, stall_hooks.py; modify request_timing.py, timing_hooks/sitecustomize.py; CPU tests; tests/gpu/check_stall_timeline.py.
- Test event/call lifecycle and injected errors before implementation.
- Hook native model source verified against configs, wrap all43 experts after lazy installation. CPU-safe imports.
- Collector every host call, all prefill/every32decode CUDA samples, bounded256 samples/16000calls, preallocated88 events/sample plusanchor. Endpoint /arm POST{}, /snapshot GET, /status GET on30210+rank. Idle lock and device context required.
- Report rank/source/config/anchor/calls/routed intervals/failures/overflow. No GPU sync/reads in forward.
- Add allowlisted `weave_stall_timing` API/scheduler timestamps only when envWEAVE_STALL_DIAGNOSTIC=1.
- Source hash helper `stall_source_sha256()`; schema/API coordinate with Task2/3. No commits.

## Task2 client diagnostics and analysis

Files: new src/weave/stall_analysis.py, stall_client.py; modify c32_serving.py and bench_c32_serving.py; focused CPU tests.
- Opt-in run_measurement(...,stall_port_base=None), arm rank endpoints afterwarmup/beforewindow, collect anonymous request interval/timing records andtop64gaps, save client-stall.json atomic0664 on success/failure.
- Do not change normal client behavior. Snapshot controller owns server collection.
- `analyze_stalls(rank_reports,client_report,window)` validates identities/completealignedsamplecoverage, returns instrumented phase totals, max/median rank latency/spread, perlayer costs, gap overlap using interval union and anchor uncertainty, request queue/prefill durations or missing counts. No rank-duration sums, no missing-as-zero. Bounded strict schemas/privacy.
- Test synthetic prefills covering a gap, crossrank drift, missing timestamps, no double count and no raw data. Coordinate exact schema with Task1/3. No commits.

## Task3 launcher and controller

Files: run_native_ep.py, run_c32_comparison.py; new run_prefill_stall_diagnostic.py; CPU launch/export tests.
- Add --stall-diagnostic, --stall-port-base default30210; diagnostic only eager/nootherprofile/capture, opt-in request timing.
- Use same three Hopper groups. Launcher common env/hook for native/custom. Client passes arm port only for measurement. Ports nonoverlapping.
- Require collector GPU smoke in same existing gate container before actual Hopper smoke; runtime pin preserved. Main controller fetch /snapshot on4ranks afteridle, persist stall-rankN.json then analyze client-stall with window. Mark report/group/client/comparison instrumented and used_for_formal_performance false. No formal speedup claim.
- Source checks, all43 coverage, diagnostics completeness, owned cleanup/export retained. Strict allowlist includes diagnostic artifacts. Legacy CLI behavior unchanged. No commits.

## Task4 root delivery

- Source audit pinned request stats/model, document approved scope and schema rulings.
- Focused tests + baseline comparison, independent task/final reviews.
- Docs/server prompt with exact new entrypoint, existing dataset/proof/cache, six-hour GPUQ budget adjustable, no server downloads.
- Commit/push main after review. Preserve unrelated untracked handoff. GPU measurements pending server return.
