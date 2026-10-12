# Decode execution profiling implementation plan

> **For agentic workers:** Execute the user-approved diagnostic in the existing experiment flow. Use test-driven-development and review the CUDA and export changes before delivery.

**Goal:** Measure real per-layer worker computation, dispatch, combine and completion waits, and record sampled c/K/steal plans during the existing GPQA run.

**Architecture:** Optional device instrumentation uses the H100 global nanosecond timer. A compile-time uninstrumented kernel specialization avoids timer probes when disabled. Preallocated per-layer/per-phase counters, worker totals and a bounded sample ring survive CUDA graph replay. Idle localhost snapshots are validated and explicitly exported as diagnostic evidence.

**Tech stack:** Existing CUDA13/Torch2.11/SGLang0.5.17 image; Python standard library for aggregation, HTTP and export. No new downloads or packages.

**Approved scope:** User's 2026-10-08 annotation approval: expert computation, dispatch, combine and final waits with actual c/K/steal, under the current GPQA requirements.

## Constraints

- Four scheduler-assigned H100s; TP4/EP4; Main448, seed42, natural Think High input, greedy, output cap32768, full decode graph and1200s per group.
- Profiling changes CUDA source identity. Fresh source-bound core/calibration/model gates are required; never bypass old profile provenance.
- Profiling is an instrumented diagnostic. Its timing is excluded from formal speedup claims.
- Timer deltas are within one device only. Worker totals and concurrent phase envelopes are not additive layer latency or hardware utilization.
- Fixed ring defaults: stride128, capacity8, prefill/decode planes. Allocate before capture. No host reads in forward or replay.
- JSON exports contain counts/timing/identity only. Preserve existing ACL-safe JSON and local export disk rules.

## Task 1: Trace ABI and aggregation

Files: csrc/execution_trace.h; src/weave/execution_profile.py; tests/cpp/test_execution_trace.cpp; tests/cpu/test_execution_profile.py.

- [x] Write tests for sampling/ring wrap, timestamp-free totals, overlapping worker envelopes, malformed samples and explicit safe export.
- [x] Run `.venv-data/bin/python -m unittest tests.cpu.test_execution_profile` and observe the missing feature failures.
- [x] Implement the shared32-field ABI, counters and pure aggregation. Keep nanoseconds as integers and carry source/profile/checkpoint/device identities.
- [x] Re-run focused tests. Reject invalid sample counts, sequence/slot mismatch, bad c/K, nonmonotonic timers and changed identities.

## Task 2: CUDA and graph binding

Files: csrc/megakernel.cu; src/weave/ops.py; src/weave/sglang_backend.py; src/weave/sglang_ext/models/deepseek_v4.py; tests/gpu/check_functional.py.

- [x] Add per-worker timer records to dispatch work/wait, expert tile work/input/queue wait, return work/dependency wait, local reduction and completion wait.
- [x] Record planned c/K/steal, workload and actual tile/return counts. Use stream-ordered counters with one writer per worker; ring updates are bounded.
- [x] Configure buffers before graph capture and pass layer/phase metadata in the captured kernel arguments.
- [x] Add GPU functional checks: enabled versus disabled numeric output; capture/replay produces new samples; wrap keeps latest samples; prefill/decode planes separate. Server core gate runs these checks.
- [x] Include the new header in extension/source identity and run existing host ABI tests.

## Task 3: Existing service and experiment flow

Files: src/weave/evidence_server.py; scripts/run_native_ep.py; scripts/run_model_acceptance.py; scripts/run_full_experiment.py; src/weave/experiment_export.py.

- [x] Add explicit profile CLI flags and `/execution-profile` snapshots. Native launch rejects Weave-only probes; missing/invalid configuration fails before Docker.
- [x] Full experiment propagates options to model acceptance and the Weave group. Collect four rank profiles after measurement while the service is still owned.
- [x] Validate source/checkpoint/profile/graph identity; write public-safe profiles using mode-preserving atomic JSON.
- [x] Mark report, comparison and summaries as instrumented diagnostics, including partial/deadline output. Keep default formal behavior unchanged.
- [x] Test real CLI previews, localhost endpoint behavior and explicit export hashes with CPU fixtures.

## Task 4: Verification and GitHub handoff

- [x] Run all CPU tests, host C++ checks, Python compilation, shell syntax and diff checks.
- [x] Review CUDA safety, sampling semantics, graph allocation and diagnostics-only result labels. Repair concrete findings.
- [x] Document one GitHub-only server command with fresh work paths and guard120; archive four profiles, source-bound gates and sanitized GPQA summaries.
- [ ] Commit, fast-forward main, verify merged tests, push GitHub and confirm remote SHA. GPU verification remains pending until server reports return.
