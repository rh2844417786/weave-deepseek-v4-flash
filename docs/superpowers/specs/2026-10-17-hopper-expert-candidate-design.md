# Hopper expert compute candidate

**Policy update after the first server run:** The user's subsequent2026-10-17 instruction supersedes all numerical-threshold requirements below. Hopper error metrics are record-only under AGENTS.md; finite differences cannot gate timing/completion/promotion. The original design text is retained as historical context. Current policy andschema2 are documented in docs/HOPPER_EXPERT_CANDIDATE.md.

User approved the staged optimization sequence on2026-10-17. This delivery implements the first executable stage: a separate FP8-activation/MXFP4 Hopper candidate and real B32 trace differential benchmark, preserving the BF16 reference and production defaults. Service replacement, fewer collectives and B32 service graphs depend on this candidate's returned numerical/performance evidence; they are not promoted from CPU checks.

## Evidence and constraints

Base714af26 includes two failed C32 comparisons: core/online GPU gates pass, cached persistent generation acceptance differs from native, staged serving throughput never starts. Preserve those gates and failures. Do not lower them or claim this candidate resolves the unrelated full-model mismatch.

Original local DeepSeek-V4-Flash mixed checkpoint, H4096/I2048/E256/top6/43 layers, four allocated H100, TP4/EP4. Existing SGLang0.5.17/Torch2.11/CUDA13 image. Server retrieves only GitHub: no downloads/build dependencies at runtime. A minimal pinned CUTLASS header closure is vendored on the development machine with per-file hashes and licenses. Do not import the attachment's entire SGLang0.5.19 overlay or EP1 serving path.

## Candidate contract

Separate extension under `experimental/hopper_fp8`, loaded only by explicit candidate tools. Existing `csrc`/BF16 extension identity and default service behavior remain unchanged. CUDA SM90a uses real TMA and FP8 WGMMA M64xN64xK32, stagingK128. Use the attached4fb1e6f implementation as attributed reference; preserve MIT/BSD notices. No scalar GEMM or installed native MoE fallback.

Workspace maxB32 has local-expert row/slot metadata, active-expert list/count, FP8 grouped input and down-input storage, per128 activation scales, BF16 gate/up/down storage and FP32 routed slot outputs. Online GPU planner preserves token→slot order, validates unique/range IDs, uses at most32 rows/expert and compacts active experts. Bounds/empty-block exits never require GPU values on CPU. Buffers/maps are allocated before capture, weights bound once with lifetime ownership; CUDA driver tensor maps bind original raw packed weights without making full-model weight copies.

Quantization: for each BF16 activation row and128-column block, use scale=maxabs/448 (allzero→1), FP8 E4M3 RNE saturated finite of x/scale. MXFP4 signed E2M1 nibble maps exactly to E4M3 code; E8M0 block32 scales apply to each K32 FP32 partial. GEMM outputs round BF16. Gate/up output and SwiGLU preserve current BF16 clamp/product boundaries before quantizing down input. Routed products use current BF16 route/product rounding and write zero for nonlocal slots.

Only active experts enter TMA/WGMMA; no M padding tiles beyond the oneM64 needed for maxB32. Padded rows are zeroed consistently. M64 efficiency is an experiment, not assumed to exceed M16 at sparse decode.

## Interfaces

Extension `FP8Workspace(rank,ep,max_batch,h,i,experts,topk)` accepts h/i multiples128, topk<=16,ep<=4,B1..32.
`bind_weights(raw_six) -> FP8BoundWeights` owns original tensors/workspace.
`run(x,ids,routing,bound,clamp=10) -> None`, noalloc/nosync/noCPUrouting; stable `slot_outputs` FP32[maxB*topk,H], `group_counts` int32[localE], `active_experts` int32[localE], `active_count` int32[1], `expert_slots` int32[localE,maxB]. Fixed input dtype BF16, IDs int32/int64 validated before narrowing, routing floating converted into fixed FP32 buffer.
Optional phase API `prepare(x,ids,routing)`, `gate_up(bound,clamp)`, `down(bound)`, `route()` runs same steps asrun; no different math. `resources()` reports compile attrs, shapes/maps/grid capacity and numeric contract outside timing.

Python loader has deterministic source manifest including candidate code, vendored header hashes, flags and numeric contract version. A missing/mismatched dependency fails before compilation, with no online installation. BF16 source hash recorded separately.

## Numerical and performance gates

Separate two comparisons: (a) candidate vs independently quantized Torch oracle for implementation correctness; (b) candidate vs current BF16 output for FP8 drift. Use existing rtol.03/atol.02 andrelativeL2<=.025 for (a). RecordallFP8-vsBF16 errors without hidingfailedcases; promotion requires the existing .025relativeL2 bound too. Ifcandidate correctness fails, no performance promotion; ifprecisiondrift fails, preserve validated timings as diagnostic only anddenydefaultservicepromotion.

Four-rank driver reuses private capture samples64/128 atall43layers; verifies allreplicas and checkpoint/image/tokenizer/capture identities. Oldcaptureinputs can be reused, oldtimings cannot. Synthetic cases cover changingB1/7/17/32, empty localrank, zeroactivation, scales/nibblepatterns, stablebuffers, eager andB32 coregraph replay. Compare baseline online BF16 staged slot production and candidate total slot production with the same active inputs and weights; include grouping, quantization, math and slot routing. Measure both eager andgraph microbench pairs, never call themservingTPS. CUDAevents prewarmed, variantorderdeterministic, correctnessbeforetiming, publicscalarerrors/counts/resources/hashonly. Privateactivation/token/route data never enterGitHub.

## Delivery

CPU testfirst helpers and contracts; independentCUDA/loader/oracle/replayreviews. Serverentry compilescandidate in existingimage thenruns gates/replayall43 with configurablebudget andatomic0664evidence. GitHubmainhandoff andexplicitserverprompt. NoGPUresultclaimedfrommacOS. ExistingdefaultBF16service andfailedacceptance remainvisible.
