# Tiled calibrated GPQA implementation plan

> Execute using superpowers:subagent-driven-development; tasks use continuous execution under user authorization.

Goal: grouped Tensor Core tile scheduling, measured profiles, cancellable matched GPQA runner and server orchestration.
Architecture: preserve native runtime around tiled routed experts, load measured profile banks, record shared server output timing and require GPU gates before benchmarking.
Spec: docs/superpowers/specs/2026-10-03-tiled-calibrated-gpqa.md
Tech stack: C++17/CUDA13 WMMA, Torch2.11, existing SGLang image, Python stdlib.
Global constraints: exact runtime, GPU, dataset, generation and deadline contract in the spec applies to every task.

### Task 1: server timing and formal GPQA (worker)
Files: src/weave/performance.py,src/weave/request_timing.py,timing_hooks/sitecustomize.py,scripts/benchmark_gpqa.py,tests/cpu/test_gpqa_performance.py.
Interfaces: benchmark_group(url,records_path,output_dir,*,ready_monotonic,clock=time.monotonic) returns group summary; compare_groups(native,weave) uses common completed row IDs. meta_info.weave_timing supplies request_received_s/first_token_ready_s/last_token_ready_s/output_tokens/boundary/source identity.
- [x] Fail tests for wrong token counts/client timing/partial samples.
- [x] Lazy pinned native timing hook; no installed source modifications or early Torch import.
- [x] Implement streaming1200second deadline, <=120second/2 warmups, own-rid abort,32768/normalEOS and atomic records.
- [x] Verify common-ID metrics and honest empty/incomplete status.

### Task 2: CUDA tile core (root)
Files: csrc/tile_scheduler.h,csrc/megakernel.cu,src/weave/ops.py,tests/cpp/test_tile_scheduler.cpp,tests/cpu/test_tile_scheduler.py,tests/gpu/check_checkpoint.py,tests/gpu/check_functional.py,scripts/check_runtime.py.
Interfaces: PeerRuntime(kernel='tiled'); run(...,calibration_mode=None|'compute'|'communication'); ws.workload seven int32 [local_pairs,remote_pairs,remote_unique,npayloads,ntasks,ngroups,tile_jobs]; ws.tile_counters [completed_tiles,stolen_tiles]. work_counters [stolen_units,completed_expert_tasks] retained.
- [x] Host-test actual tile claims, dependency and group-capacity helpers before production use.
- [x] Implement grouping, M16/N64 WMMA tiles, producer fences, tile stealing and counters.
- [x] Extend source identity/IPC layout/calibration modes and all43layer tiled tests.

### Task 3: measured calibration (worker after Task1)
Files: src/weave/calibration.py,scripts/calibrate_weave.py,tests/cpu/test_calibration.py.
Interfaces: validate_profile_bundle(bundle,*,kernel,source_sha256,device_key,hidden_size,intermediate_size,rank,workers) normalizes bucket rows. profile-bank-rank0..3.json, kernel=tiled_mxfp4_tile_v1, each bucket max_local_tokens/candidates. Parent loads tensors before graph capture.
- [x] Reject stale, incomplete, nonfinite or wrong-rank measured data.
- [x] Implement real compute/communication/mixed CUDA-event measurement and c/K profile derivation with original layer0 weights.
- [x] Enforce1200seconds and identity-safe resume.

### Task 4: integrate and deliver (root)
Files: scripts/run_full_experiment.py,scripts/run_native_ep.py,src/weave/sglang_backend.py,src/weave/sglang_ext/models/deepseek_v4.py,scripts/export_experiment_evidence.py,docs/FORMAL_GPQA_EXPERIMENT.md,tests/cpu/test_full_experiment.py.
- [x] Require new source core correctness, calibrated profiles, actual tiled eager/fullgraph acceptance before formal groups.
- [x] Match image/model/input/profile/timing identities; no fixed-plan Weave formal run.
- [x] Reuse owned group cleanup and bounded per-group watchdog; export only aggregate/provenance evidence.
- [x] CPU/syntax/hostC++ verification, per-task and final review; integrate and push authorized repository. Never label H100/performance as executed locally.

## Delivery verification

Local CPU suite:115 passed,0 skipped. Python compileall, launch shell syntax, new CLI parsers and shared C++ numerical/scheduler/tile helpers passed. Final source/CPU review approved after closing token-ID export and prepared-data provenance defects.

These checks establish implementation delivery only. Current tiled CUDA compilation, four-H100 all43layer/graph/steal gates, real calibration event samples, actual cancellation GPU release and formal GPQA performance remain server execution gates. The server operator pulls main and runs scripts/run_full_experiment.py using the existing offline runtime.
