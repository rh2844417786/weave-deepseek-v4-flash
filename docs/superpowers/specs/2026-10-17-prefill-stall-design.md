# C32 prefill and output-stall diagnostic

The user authorized diagnosis, code, GitHub push and server handoff after e8498aa. Existing C32 results: native 235.47 TPS, BF16 staged 75.68, Hopper 166.40; Hopper mean TTFT42.69s versus native0.857s and mean client ITL137.57ms versus134.92ms. These do not locate GPU causality.

## Scope

Keep the same three C32 eager groups and existing 600-second custom workload. Add opt-in diagnostic observation only; do not change compute, scheduling, graph, prefill chunk size or generation policy. All timing is instrumented diagnostic, not formal performance evidence. No numerical-error or token-content gates. Preserve original model, existing image, dynamic four GPU allocation, GitHub-only delivery, and no raw public inputs or outputs.

## Observations

A source-pinned lazy import hook wraps native DeepseekV4ForCausalLM.forward, which is also called by the custom adapter. Bootstrap at first forward after custom replacement, wrap each of 43 routed expert instances. Separate rank-local localhost endpoints (base30210) arm after client warmup, snapshot only after abort/drain/idle. No device synchronize or event reads in measured forwards. Record every model host invocation (bounded16000); collect CUDA current-stream intervals for every prefill and every32nd decode, plus all43 routed-expert intervals within sampled calls (bounded256 samples, preallocated events). Report overflow/failure explicitly; do not silently overwrite. Preserve outer native execution. CPU enqueue duration is not GPU completion. GPU stream intervals include stream waits/launch starvation; routed spans include dispatch/combine and cannot claim pure GEMM timing.

Model/layer costs use direct start.elapsed_time(end), not subtraction of large anchor-relative float offsets. Validate sequential routed events before subtracting summed direct routed durations from direct model duration; anchor offsets remain approximate alignment coordinates.

Arm calibrates each rank's anchor event to a host perf_counter bracket while idle. Report both host bracket and CUDA-relative timestamps so approximate client overlap has an uncertainty bound. Linux client monotonic/perf_counter and server perf_counter must identify the same CLOCK_MONOTONIC clock domain. Across ranks aggregate max/median/spread by aligned sequence/phase/rows, never sum parallel rank durations.

The existing pinned APIServerReqTimeStats conversion receives scheduler stats. Add a diagnostic-only numeric timing object with API created/tokenized/dispatched/first and scheduler wait_queue/forward_entry/prefill_finished timestamps. Missing timestamps are reported missing, never zero-duration stages. Scheduler prefill interval includes scheduling/chunks and is not GPU pure compute.

The existing streaming client records bounded anonymous request indices, dispatch/first/last times, counts and copied allowlisted numeric server timestamps, plus top64 longest inter-output gaps. It must never retain prompt/text/token IDs in diagnostic output. Arm all four endpoints after existing client warmup and before the measured window. Controller snapshots all four while idle after measurement, joins client intervals with forward intervals, and outputs phase/routed costs and overlap evidence, not causal claims.

## Integration

Add `--stall-diagnostic` only to the Hopper comparison suite; existing default unaffected. Add convenience `scripts/run_prefill_stall_diagnostic.py`. Existing core/replay/assembly proofs retained, new source manifest covers instrumentation. Launcher enables the common hook for native/staged/Hopper, separate ports validated. Controller preserves owned children/timeouts/cleanup, export allowlist and failure artifacts; diagnostics explicitly `used_for_formal_performance:false`. Retain one fresh directory/run and existing cache paths.

## Verification

CPU fake-event tests prove sample selection, event order, bounded behavior, no forward synchronization, all-layer completeness and idle rules. Inject timestamp and gap fixtures to prove queue/prefill calculations, missing-data behavior, interval union overlap and rank aggregation. Launcher/client/controller tests prove opt-in plumbing and public export. A synthetic four-rank GPU smoke verifies the actual collector/event ordering before full-model runs; no native CUDA edits. Actual GPU execution remains server work.
