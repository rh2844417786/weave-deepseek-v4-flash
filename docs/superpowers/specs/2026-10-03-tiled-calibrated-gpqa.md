# Tiled Weave, measured calibration and formal GPQA contract

Authorization: the user explicitly requested completion of the annotated tile/calibration/GPQA deliverables. Continue implementation and existing GitHub delivery autonomously.

## Runtime and experiment
Original mixed-precision DeepSeek-V4-Flash at /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash; routed MXFP4/E8M0 retained. H4096/I2048/E256/top6/43 layers. GPU4,5,6,7 only; TP4/EP4, expertTP1, attentionDP1, no MTP/PP/EPLB/SBO/TBO. Existing lmsysorg/sglang:v0.5.17 and pinned commit29481685462732237d80d86076d6563e1f658102, Torch2.11.0+cu130, CUDA/nvcc13.0, Python3.12.3, Transformers5.12.1. Capture actual image/runtime/source/checkpoint identities. Server can access GitHub only: no pip/apt/HF/image pull.

Formal groups: native and Weave, both full decode graph batch1, shared native attention/KV/shared-expert/sampling path. GPQA Main448, seed42, natural Think High inputs, greedy, max_new_tokens32768, normalEOS. Ready observation starts a1200second absolute group deadline; warmup at most120seconds and at most2 natural records. No added baseline or artificial length. Timeout/abort rows never count as completed. Common completed original row IDs define comparison. Include reasoning and all actual output IDs. Decode TPS=sum(n-1)/sum(last-first), n<2 null; TTFT, TPOT, output counts, finish reason and input length recorded. Logprob remains record-only.

Shared timing uses native APIServerReqTimeStats first_token_time/finished_time perf_counter timestamps at TokenizerManager output handling with per-token streaming. Client SSE arrival timestamps are not decode timing. Empty/aborted/missing timing is invalid for performance.

## CUDA tiles
Add kernel=tiled (ID2) beside legacy scalar/WMMA controls. Device planner groups up to16 same-expert/same-chunk tasks. A CTA computes64 output columns using four BF16 WMMA warps; separate gate/up/down tile claims use per-group completion dependencies. Global mid buffer bounded by ceil(max_tasks/16)+local_experts*MAX_K groups. No worker may wait for a dependency while holding uncomputed gate/up work. Communication workers steal actual tiles in one post-dispatch window. Keep IPC epochs, dispatch dedup/chunk combine/BF16 boundaries; no fallback. Calibration-only communication mode zeros arithmetic output; compute mode requires no remote pairs.

Calibration: actual CUDA-event compute/communication/mixed timings with original layer0 raw weights; source/device/runtime/checkpoint/rank/token-bucket bound profiles, raw points retained. Logical FLOP and actual dispatch/return byte widths explicit. Derive rate/efficiency/overlap from measured data.1200second calibration invocation limit with validated resume. Incomplete data never claims measured.

## Gates and delivery
New CUDA must pass compile, all43layer scalar/WMMA/tiled reference/graph/steal tests, then actual43layer SGLang eager/full graph gates before formal performance. One server orchestrator runs gates, calibration, native formal, Weave formal and sanitized export. Raw questions/token IDs/generations stay ignored work. Mac has no CUDA and no direct GPUQ server connection; implement and verify all local work, mark actual H100/performance results pending until server execution.

Ruling: tiled compute profiles use executed M16 padded FLOPs (logical expert FLOPs also saved). GPU planner derives actual expert/chunk group counts for candidate K, so partially occupied expert tiles do not inherit a falsely optimistic logical throughput from a densely packed calibration fixture. Legacy scalar/WMMA profiles retain logical FLOPs.
