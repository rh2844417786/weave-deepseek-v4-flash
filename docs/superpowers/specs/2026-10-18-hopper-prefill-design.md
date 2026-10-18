# Compact multi-tile Hopper prefill

User requested modifying prefill after verified5629aaf evidence: 8035row persistent prefill≈37.8s, >99% routed expert spans; Hopper decode already≈134ms/modelstep. Implement an independent FP8 prefill compute path while preserving existing BF16/decode candidate CUDA and decode assembly hashes.

## Compute and memory

New experimental/hopper_prefill CUDA extension with PrefillWorkspace(rank,ep,max_batch,h,i,experts,topk), B1..8192. Original raw6 MXFP4/E8M0 weights and existing FP8 numeric contract retained. Bind weights once/layer. GPU validates original-width IDs/range/no duplicate pertoken and finite routing, groups slots deterministically per expert, generates device-only compact tile metadata. Use per-expert stable parallel compaction or an equivalently bounded implementation, never CPU routing or host reads of active counts.

Each active expert gets ceil(count/64) M64 tiles. Compact buffers are tile-major [max_tiles,64,width], max_tiles=ceil(max_batch*topk/64)+local_experts. tile_expert and tile_row_base link to the original expert slot list [local_experts,max_batch]. This bound covers concentrated routes and empty experts without padding every expert to8192. Reuse audited TMA/packed MXFP4/WGMMA arithmetic adapted to tile offsets and original expert weight maps. All capacity-launched blocks guard inactive tiles uniformly before barriers. Zero only active routed slot prefix. No scalar/native fallback or arbitrary chunk32 loop.

Expose slot_outputs FP32 [maxB*topk,H], group_counts, expert_slots, tile_experts,tile_row_bases,tile_count, resources. `run(x,ids,routing,bound,clamp)` writes rank-local slots. `finalize(batch)` returns preallocated BF16 origin-masked output after an externally enqueued SUM. Ordered six-slot FP32 sum, BF16 RNE, contiguous ceil(B/EP) ownership. One workspace sequentially reused, raw handles retain weights. New source manifest validates vendored headers/build flags before compilation with existing CUDA13 image. Original decode source stays unchanged.

## Service

Add explicit --prefill-implementation persistent|hopper_fp8, defaultpersistent. New path only weave_functional, hopper_fp8 decode, tiled/hotcache/fixed1,1,0/eager, no existing profilebank/capture. Stall diagnostics opt-in remain supported. `HopperPrefillRuntime` runs compact workspace on replicated fullinputs, one active-slot SUM on TP device group, finalize, cached BF16 output. Supports1..8192; reject larger input with actionable error, no fallback. Existing HopperDecodeRuntime unchanged. Perlayer implementation_calls.hopper_prefill and distinct prefill collective counter/source/resources recorded. Existing decode collective counts remain separately accountable.

## Verification and comparison

New suite prefill: native, hopper_fp8 (old persistent prefill+current Hopper decode), hopper_prefill (new prefill+same Hopper decode). Same existing dataset, EP4,eager,C32,1024output,600s/group. Existing complete1376 decode replay/core evidence reused with unchanged hashes. Add GPU prefill checks before serving: synthetic changing/empty/skewed/multi-tile cases through8192; all43realweights at B129; real layer0 B8035 and8192; metadata exact, output finite, stable cache/lifetime and one slot collective. Compare finite output differences against unchanged FP8 candidate run in <=32 chunks as record-only numeric reference; metadata/assembly deterministic invariants are correctness tests, not numerical accuracy cutoffs. Source-bind new CUDA/runtime evidence and actual all43 service coverage. Compute time recorded separately from checks; no speed claim before server evidence.

Provide normal and --stall-diagnostic versions using existing controller. Instrumented results do not become formalperformance ratios. No new numerical/token-equality gates. Raw checkpoints/routes/activations/outputs stay private; only aggregate public JSON with0664 permissions, immutable manifests and owned cleanup.

## Constraints

Existing lmsysorg/sglang:v0.5.17 (Torch2.11/CUDA13/Python3.12/Triton3.6), original readonly model, scheduler-assigned fourH100. Server GitHub-only, no pip/apt/HF/image downloads. Preserve caches. Independent reviews and CPU tests; actual GPU compilation/service measurements deferred to server. User authorized existing GitHub handoff and requested no repeated permission.
