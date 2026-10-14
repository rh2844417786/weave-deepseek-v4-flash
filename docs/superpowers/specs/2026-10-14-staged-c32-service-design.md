# Staged direct decode service design

User approved on2026-10-14: keep cached host path, integrate direct tile staged computation into real decode, implement online grouping and cross-rank assembly, provide a C32 comparison and GitHub/server prompt. This is the next architectural step after P0/P1 server evidence09f905e. No further approval is needed for this scoped implementation.

## Contract

Original local mixed-precision DeepSeek-V4-Flash, H4096/I2048/E256/top6/43layers, TP4/EP4, existing lmsysorg/sglang:v0.5.17 and installed packages. Respect four scheduler-assigned H100s. Server uses GitHub only; no online installs/models/images. Keep the immutable raw MXFP4 expert weights and cached persistent path. Real service and throughput are validated on the server, not by local CPU tests.

## Execution

Add an explicit decode implementation switch `persistent` (default) or `staged_direct`. Staged service supports eager decode batches1..32, rejects unsupported decode shapes/graph/profile combinations, and uses the existing cached persistent implementation for prefill. Prefill use is documented and counted; no native expert fallback exists.

Online GPU grouping scans actual top-k IDs in token/slot order, preserves the P1 planner's first-encounter grouping with at most16 rows, and writes bounded preallocated metadata. No CPU routing download, `.item()`, device synchronization, or device-dependent host grid calculation occurs in forward. Direct mapping launches a safe capacity bound; blocks beyond actual group_count exit. Report this bound rather than claiming an exact active-group launch.

Each layer has a lifetime-owning immutable bound weight handle; workspaces/scratch are shared across sequential layers. Preallocate x/IDs/routes, group metadata, intermediates, slot contributions and output views for B<=32. Dynamic forwards copy active inputs, build groups, and run the existing gate/up and down math. Keep P1 frozen replay API unchanged.

Cross-rank assembly preserves current routed-product rounding and slot order: form local BF16-rounded routed products in a zero-filled FP32 [B,topk,H] slot buffer; NCCL SUM over the existing TP device group has exactly one owning expert rank per slot. Sum slots0..5 and cast once to BF16, returning only the rank's origin rows. The audited outer SGLang shared-expert/scaling/TP-reduce path then remains intact. This first service version therefore has one added slot all-reduce before the existing outer reduction; both are included in end-to-end timing. The design deliberately avoids prematurely rounded local BF16 sums or duplicated replicated outputs. P2 optimization can later remove redundant communication after separate numerical validation.

Evidence adds decode implementation, per-layer staged/persistent decode counts, prefill counts and cheap CPU batch histogram. GPU snapshots only while idle; no execution probe or per-step CPU GPU sync in throughput runs.

## Gates and comparison

Fresh source core/all43 checkpoint remains mandatory. New four-rank online-staged GPU test covers changing batches1,7,17,32, empty local ranks, repeated calls and changing routes, metadata versus CPU planner, exact slot assembly, original real weights/all43 against persistent/reference. Numerical requirements stay relative-L2<=.025 and existing rtol/atol. No arbitrary threshold relaxation; logprob remains record-only.

C32 script runs native eager, persistent-cached eager (optimization control), staged-direct eager sequentially with identical original9x200 custom dataset and1024-token requests, concurrency32, temperature0/ignore_eos. The native/staged pair is the serving comparison; persistent is a clearly labeled optimization control. Default measured window600s/group, independently configurable setup/drain budgets. No fullgraph comparison in this delivery.

An offline client in the existing image counts actual streaming cumulative completion tokens inside a shared monotonic window, including unfinished requests and excluding each request's first token for decode TPS. Do not divide only fully completed request tokens by a short window, count model rows as output tokens, or mix scheduler logs with client measurements. Preserve completed/failed/unfinished counts, TTFT/ITL boundaries, workload/source/image identity, correctness proof, actual batch evidence and cleanup. Incomplete evidence is not a throughput pass. All prompts, token IDs, output text and full raw logs stay private; allowlisted counts/hashes/timing and manifests may go to public GitHub.

## Delivery

Implement with independent task reviews, focused CPU and full-suite baseline accounting, then GitHub main and an explicit server prompt. Preserve the unrelated untracked handoff document in the primary checkout. The new source invalidates older core gates, but does not require an unrelated cost-model scan.
