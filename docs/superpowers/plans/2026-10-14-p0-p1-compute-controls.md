# P0/P1 compute controls implementation plan

User approved the Pro review on2026-10-14 and allowed GPUQ duration to grow as needed. The first delivery implements P0/P1 only. The review is source material, not measured speedup. This plan is the implementation spec for this delivery.

## Fixed scope and invariants

- Existing original mixed-precision DeepSeek-V4-Flash checkpoint; H4096/I2048/E256/top6/43layers, four H100, TP4/EP4. Respect GPUQ allocation. GitHub-only delivery, existing image/dependencies, no online installation.
- Keep current M16xN64 arithmetic and decode4/BF16/SwiGLU/routed-product rounding. No local-EP service replacement, LUT, M8, GEMV, async pipeline, split-K, new graph service buckets or recalibrated cost model yet.
- Baseline remains current persistent scheduling/IPC; never increase its132-worker grid. P1 staged controls may use132/264/528 CTA grids or direct output-tile mapping because separate kernel launches enforce stage dependencies.
- Preserve unprofiled and profiled specializations and report their compiled attributes. Sampling stride does not imply the non-sampled calls run the unprofiled specialization.
- Real replay inputs/route IDs/route weights remain private on server. GitHub receives counts, hashes, resource/timing/correctness reports only.
- Default capture: two full B32 decode steps (eligible B32 sequence64 and128), all43 layers and four ranks, from the existing9-prompt×200/1024-token stress workload. Capture is excluded from performance measurement. Capture and replay are separate jobs, with configurable setup/run timeouts and validated reuse of completed stages.
- CPU and GPU correctness gates remain explicit. Do not claim improvement from compilation or local tests. Logprob remains record-only.

## Task1: P0 static binding and compiler resources

Files: host portions of csrc/megakernel.cu, src/weave/ops.py, focused CPU tests and GPU verification additions.

Interface: Workspace.bind_weights(list[Tensor]) returns a lifetime-owning bound object; Workspace.run_bound(bound, profile,c,k,steal,timeout,clamp,tensor_core,calibration,trace_layer,trace_phase) uses cached device properties/layout. Workspace.run(raw,...) retains the uncached host-control path. Workspace.resources() reports both persistent specializations' numRegs/localSize/sharedSize/occupancy and launch parameters.

PeerRuntime.bind_weights(raw) forwards binding; PeerRuntime.run accepts either raw list/tuple (uncached control) or a bound object. Cache x/ids/routing/n/output views once in PeerRuntime initialization. Provide an explicit cache_views=False control for the uncached replay baseline. A bound object holds strong tensor references and rejects use with a different workspace. Stream lookup and dynamic input validation stay per invocation.

Do not change GPU arithmetic/protocol sections while implementing Task1. Extension source hashing must include every runtime .cu/.cuh/.h and compile flags. Keep a deterministic sorted manifest with relative names and per-file hashes for new identities. Compile with existing optimization settings plus ptxas resource reporting; no fast_math.

Tests: view accessor calls once vs repeated control; binding/device/dimension validation; source manifest includes new headers; numerical eager and graph behavior remains covered server-side. Local host tests cannot validate CUDA.

## Task2: private real-trace capture

Files: new src/weave/replay_trace.py, adapter/model wrapper/evidence server, launcher flags, capture driver and tests.

Preallocate per-rank BF16 x[2,43,32,4096], int32 ids[2,43,32,6], FP32 route[2,43,32,6]. Count eligible full-B32 forwards on CPU; sample64,128 before each routed module's origin slicing. Copy GPU-to-GPU only during capture. Complete snapshot/download after owned-service abort/idle. Export private tensors via torch.save with weights_only-safe schema, atomic readable permissions, manifest hashes and complete layer/sample bitmaps. Public metadata never contains tensors or IDs. All ranks must refer to same steps, kernel/checkpoint/input workload identity.

Capture driver reuses existing installed benchmark and dataset, existing image/ports/OwnedChild cleanup, verifies all43 checkpoint evidence for the new source, and stores a bounded capture without claiming throughput. Missing complete trace is a hard failure; no synthetic substitute.

Adapter binds each layer's raw weights outside graph capture. A runtime switch preserves raw/uncached service execution for controls; production correctness gates still apply.

## Task3: P1 staged math kernels

Files: new csrc/staged_compute.cu/.cuh and shared tile-math header, GPU/CPU planner tests. Task3 owns new files first; integration with megakernel GPU math occurs after Task1's host changes finish.

Extract shared weight decoding, group MMA, gate/up math and down math without changing operation/rounding order. Original persistent compute calls the shared math and retains its own protocol. Staged execution has independent gate/up, down and deterministic local reduce launches, no peer memory or cross-CTA spin waits.

Interface exposed by bind_staged(module): StagedWorkspace(rank,ep,batch,h,i,experts,topk), bind_weights(raw), run(x,ids,routing,group_meta,group_slots,group_count,grid), output, slot_outputs, resources. x is replicated logical input[32,H]; IDs global, route FP32. group_meta int32[G,2]=(local expert,valid rows); group_slots int32[G,16]=token*topk+slot; group_count int32[1] lives on GPU. Groups are built from frozen trace outside timing, preserve token/slot order and split at16. grid132/264/528 uses a fixed-grid strided tile loop; grid0 uses direct mapping for the frozen metadata capacity (no GPU .item in timed/captured path).

slot_outputs is float[B*topk,H], storing BF16-rounded down output as float. Local reduce walks slots0..5, selects only the rank's experts, uses original routed_product and deterministic FP32 summation, outputs FP32 local contribution. Replay validates final global slot-order output outside timing; local partials are not yet wired into SGLang.

Tests include ownership/count conservation, empty local tasks, partial/full groups, padding, exact tile coverage, unchanged expert rounding, grid variants vs original real-checkpoint reference on all43 layers. No speed claim until H100 results.

## Task4: replay harness and handoff

Replay each frozen layer/rank/step with the same raw checkpoint and buffers. P0 controls: uncached metadata/views without probe; same with probe specialization; cached bound/view path without probe. P1 controls: staged132/264/528/direct with identical arithmetic, full gate/up+down+reduce totals and per-stage events. Compare correctness before timing. Warmup/repetitions, variant order, host elapsed and CUDA-event distributions are recorded; do not fit old bandwidth curves to rank arrival delays.

Export resource attributes, compilation flags/log summaries, source/checkpoint/trace hashes, per-layer/rank pairs/groups/fill, timing samples and correctness. Kernel event time can include protocol/launch wait; staged totals exclude IPC and cannot alone be called E2E speedup. Public exports use explicit allowlists and SHA256. Support validated restart at completed capture/replay stages; do not reuse results with a different source/configuration.

Deliver scripts, focused tests, docs and server prompt through the established GitHub main handoff. Run local CPU tests, independent review and source checks; identify inherited GPQA test failures separately. Server owns compilation, numerical validation and measurement.

## Progress

- [x] Task1 P0 binding/resources; source/binding review passed, GPU gate pending server.
- [x] Task2 private capture; tokenizer/UUID provenance review passed, actual capture pending server.
- [x] Task3 staged kernels; arithmetic/planner source review passed, NVCC/numerical gate pending server.
- [x] Task4 replay and server handoff implemented; scoped and final integration reviews passed. GPU execution remains a server task.

Local checks:47 focused tests passed; the full219-test CPU suite retains the baseline7 failures/3 errors in GPQA provenance/order checks. This delivery does not claim a CUDA build, H100 numerical pass or performance improvement.
