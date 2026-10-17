# Hopper FP8 EP4 serving integration

Approved after the complete 1376-point Hopper replay at `dbe03bd`. Connect the measured compute candidate to actual EP4 eager decode and compare matched C32 serving. Numerical errors and generation differences are record-only; runtime, nonfinite, provenance and coverage failures remain failures.

## Runtime

Add explicit `--decode-implementation hopper_fp8`, with cached persistent prefill and original local MXFP4 checkpoint weights. Use TP4/EP4, batch 1..32, eager, the existing SGLang 0.5.17 image and scheduler GPU allocation. No native fallback. Preserve existing persistent/staged modes.

Keep measured candidate CUDA `bcabd0f4...` and BF16 CUDA `25b51a34...` unchanged. Add a separate Triton finalizer using the image's existing Triton: ordered FP32 slot 0..5 addition with explicit `add.rn.f32`, then BF16 RNE and contiguous origin masking. Preallocate output and cached prefix views. Give assembly its own source identity and GPU checks.

Forward sequence: `FP8Workspace.run(full replicated inputs, bound weights)` → NCCL SUM of active FP32 slots on the existing TP device group → finalizer → origin-masked BF16 output. Preserve outer shared expert/scaling/TP reduction. BF16 staged and FP8 use the same communication layout; all overhead enters serving time. Bind once per layer, reuse one workspace sequentially, and avoid CPU route inspection, D2H synchronization and tensor allocation in the forward path. Count actual Hopper calls separately.

## Verification and comparison

Require an immutable complete schema2 record-only replay bundle: 1376 points, verified manifest, matching current candidate/BF16 native hashes, checkpoint, tokenizer and image. Preserve historical replay/controller Python identities without requiring equality to current serving Python. Recompute the stored summary. Never reinterpret failed or schema1 evidence as complete.

Add a four-rank GPU smoke using the actual runtime and synthetic original raw weights. Exercise B1/7/17/32, shrinking, changing inputs, empty expert/origin ranks, stable buffers, bound weight lifetime and one collective. Compare assembly exactly to the same computed precollective slots. Add adversarial known-slot finalizer fixtures; no FP8-versus-BF16 numerical tolerance.

Hopper C32 suite: native eager, staged_direct BF16 eager, hopper_fp8 eager. Retain the original 9×200 custom dataset, concurrency 32, output cap 1024 and default 600-second window. Reuse the stream client and metric. Keep the older staged suite available. The new suite uses its own assembly smoke and record-only 32×8 generation comparisons; incomplete, malformed or foreign-ID responses fail.

Coverage verifies all 43 actual requested decode paths, candidate/assembly identity, B32 histograms and exactly one extra slot collective per custom decode call. Collect coverage while idle. Full-model throughput includes communication, attention, shared expert and sampling.

## Delivery

Use existing image/model/packages and GitHub only. No server pip, apt, HF download or image pull. Parent owns fresh directories, child deadlines and cleanup. Public export is allowlisted; prompts, responses, IDs, trace and full logs remain private. Source manifests bind the finalizer and adapter even though native CUDA hashes stay unchanged. CPU review is not GPU evidence. Push main and provide a complete server prompt. Defer B32 service graph and collective reduction until serving evidence returns.
