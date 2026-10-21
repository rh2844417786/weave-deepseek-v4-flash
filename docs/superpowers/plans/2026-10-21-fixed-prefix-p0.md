# Fixed-prefix P0 diagnosis implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent runtime and P0 controls tasks, with root integrating the controller and evidence.

**Goal:** Deliver an offline short fixed-prefix native repeat/P0 decomposition probe and source-bound public evidence.

**Architecture:** Existing launch/lifecycle/GPQA preparation are reused. P0 event and buffer controls, a native sampling-boundary observer/teacher forcer, and a host orchestration plus CPU comparison pipeline form separately testable components.

**Tech Stack:** Existing Python3.12, Torch2.11/CUDA13, SGLang0.5.17, aiohttp/dill in existing image.

**Spec:** docs/superpowers/specs/2026-10-21-fixed-prefix-p0-design.md

## Global constraints

- Server only obtains code/data via GitHub; no installations or image/model downloads.
- Numeric and generated-content differences are record_only; no thresholds.
- Default8 positions, same32 natural prepared inputs, same teacher tokens, TP4/EP4 full graph.
- Raw tokens/logits/traces private; only allowlisted aggregate artifacts export.

## Tasks

- [x] Add event/buffer/no-readback controls preserving old P0 default; tests for actual event allocation and reference retention; launch flags.
- [x] Add source-pinned ModelRunner.sample context and processor, private logits/proofs, actual teacher-token/position checks and native trace; CPU fake-runtime tests.
- [x] Add batched fixed-prefix client, five-arm controller, existing-image CPU tensor comparison, public evidence and failure cleanup; tests for identity/coverage and record-only differences.
- [x] Validate installed native interfaces against pinned sources, run focused regressions, independent review and real launcher previews.
- [x] Document server prompt, commit/push and report local/server execution boundaries.

Local CPU and preview checks are complete. Real Torch file integration, four-H100 execution and causal conclusions remain pending server evidence.
