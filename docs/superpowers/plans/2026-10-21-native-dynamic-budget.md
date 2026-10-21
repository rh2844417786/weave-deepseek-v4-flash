# Native dynamic budget implementation plan

> Execution: superpowers:subagent-driven-development. User approval is the latest request to start coding; no repeated permission request.

**Goal:** native Marlin with routing-driven CTA budgets and graph-compatible replay.
**Architecture:** source-verified private JIT overlay, original Marlin arithmetic, device virtual-grid scheduler, per-layer counters, native/adaptive serving comparison.
**Tech Stack:** existing Python 3.12, Torch 2.11 CUDA 13, SGLang 0.5.17; local stdlib tests.
**Spec:** docs/superpowers/specs/2026-10-21-native-dynamic-budget-design.md

## Constraints
GitHub-only delivery; original read-only mixed-precision checkpoint; GPUQ assigned GPUs; numerical_policy record_only; immutable historical evidence; no paper-reproduction or speedup claim without server evidence.

## Tasks
- [x] 1. Core: add src/weave/native_budget.py (policy/reference/evidence), native_budget_patch.py (pinned transformations/overlay), csrc/native_budget.h. Test policy workload changes, zero-local and every virtual stripe coverage, all gridDim replacements, drift rejection. Add configs/native_budget_sources.json.
- [x] 2. Runtime: add native_budget_runtime.py for original/patched dispatch and layer-scoped counters. Integrate graph_hooks install and /budget-coverage endpoint. Warmup compilation before capture, persistent buffers, source identity; test forward mode and role handling.
- [x] 3. Launcher/controller: extend run_native_ep.py with --native-budget off/adaptive/full and target percentage, reject P0/profiling combinations. Add run_native_budget.py using existing 1200-second C32 streaming client and graph evidence. Counter reads only outside timing. Public allowlist/cleanup and offline dry-run tests. Write docs/NATIVE_DYNAMIC_BUDGET.md and SERVER_PROMPT_NATIVE_DYNAMIC_BUDGET.md.
- [x] 4. GPU validation: tests/gpu/check_native_budget.py uses installed source-pinned Marlin on MXFP4 with real grouping and changing routing under graph replay. Check finite outputs and complete task coverage, record all numerical differences without cutoff.
- [x] 5. Verify changed CPU tests, native launcher/graph regressions, syntax/source transforms, independent code review. Commit/push GitHub. State CUDA execution remains server-side.

## Rulings
- User authorized implementation and GitHub handoff after reviewed feasibility; skill review/approval ceremony must not ask them again.
- Start with GPU virtual-grid budgeting to preserve graph topology and native register-aware tile-selection rules. It is soft CTA resource control, and neither physical SM ownership nor communication stealing.
- Default policy is a transparent heuristic, not a fitted cost model. Report observed policy choices and throughput; no new diagnostic campaign.

## Verification outcome
115 CPU tests passed on Python 3.12; host C++ and Python policy outputs match over 504 route/geometry/configuration cases. All 20 pinned installed-source files were used to generate a private overlay and its Python wrapper was syntax-checked. Independent review found no remaining implementation defect; corrected the geometry guarantee. CUDA compilation, real H100 graph replay and throughput are not run locally and must be reported by the server.
