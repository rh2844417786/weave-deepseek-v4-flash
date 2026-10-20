# Native SGLang P0 diagnosis

## Scope

The user changed the main goal to native SGLang throughput and approved P0: establish actual execution paths and routed/shared behavior before choosing an optimization. C32 controls the first experiment; it does not constrain future batches.

Keep native Marlin, weights, shared experts and collectives unchanged. Do not introduce CTA caps, custom expert kernels or IPC. Preserve offline server delivery, numerical record-only policy, historical results, and existing GPUQ allocation handling.

## Execution

Provide an opt-in `--native-p0` mode and standalone `scripts/run_native_p0.py`. Reuse the existing GPQA window, request accounting and owned-container lifecycle. Run native only, with full graph buckets1..32, no padding, client concurrency32, and a1200-second measurement window. Exclude candidate-compute prerequisites. Verify local checkpoint identity using the existing native baseline attestation; rehash when file stats change.

Before model loading, run a four-rank Torch-only smoke for external events, two-stream fork/join, captured NCCL and CUPTI launch metadata. Explicitly separate synthetic smoke, real native-model execution, and performance claims. Raw traces stay private. Instrumented throughput is not a formal speedup result.

## Interfaces

`NativeP0Profiler(torch, rank, stride=128, capacity=64, layers=43, trace=None)` allocates external events before capture. Its methods are `register_layer`, `span`, `set_metadata`, `bind_grouping`, `begin_replay`, `finish_replay`, `abort_replay` and `snapshot`.

Capture B32 events for `moe`, `routed`, `shared`, `gate_up`, `activation`, `down`, `routed_finalize`, `shared_merge`, `all_reduce`, `router`, `topk` and `grouping`. Preserve the current stream at every boundary. Unobserved stages remain absent. Do not read GPU values or synchronize the host during capture.

After each selected replay, synchronize the outer end event and freeze timestamps before another replay can overwrite them. Sample every128 actual B32 replays and retain64 samples. Store global replay sequence and B32 ordinal. Read grouping buffers only after completion, then publish aggregate local pairs/groups, active experts, maximum rows and row-count histograms; never publish IDs or arrays.

## Native hooks and source identity

Install source-pinned wrappers before graph capture on all43 layers. Use a context-local current layer only during B32 capture. Wrap original native functions without changing arguments, outputs or branch selection. Record the actual selected normal/dual-stream/DeepEP method, shared TP configuration, module shapes/dtypes and both Marlin call contracts.

Pin official SGLang commit `29481685462732237d80d86076d6563e1f658102`. Verify loaded methods and imported aliases as well as their source files. The native profiler uses the existing graph observer replay hook and idle snapshot lock; other profilers and custom backends are incompatible with this mode.

## Launch trace

`LaunchTrace(torch, rank, private_dir)` provides `begin`, `finish` and `abort`. Collect exactly one replay with Torch/CUPTI, starting before the outer event and stopping after replay completion. Disable input shapes, stacks and memory profiling. Publish only kernel names, actual grid/block/registers/shared-memory fields, scalar timestamps and parsed template parameters.

Require86 routed Marlin kernel records, map timestamp order to43 gate-up/down pairs, and check compatibility with captured Python call metadata. Parse balanced template arguments. Do not invent static/dynamic shared-memory splits or SM allocation from launch dimensions. The CUPTI replay is retained as launch evidence but excluded from ordinary event statistics.

## Evidence summary

Validate four source-bound idle snapshots and actual full graph replay deltas. Bind sampler counts to the observer B32 histogram. Select samples after the measurement baseline and require matching sequences across ranks.

Summarize each layer's interval offsets, durations, branch overlap, exposed tails, grouping and actual Marlin resources. For each replay, select the rank with the longest whole-graph interval and take every stage value from that same rank. Do not add overlapping branch times or independent rank maxima.

Record the native shared-TP1 post-reduce addition boundary explicitly. All-reduce includes waiting; outside-MoE time is not exclusively attention. Trace/event/grouping instrumentation can perturb execution. A visible tail or overlap is a hypothesis for a later controlled contention experiment, not proof of a throughput gain.
