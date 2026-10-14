# C32 fixed-plan eager diagnostics

User-approved scope: diagnose the existing tiled/eager/(1,1,0) C32 workload and provide a GitHub-only server handoff. No expert arithmetic or schedule optimization in this change.

Implementation uses existing device trace buffers and extends their provenance to discriminate measured and fixed schedules. Fixed reports bind the explicit plan and checkpoint, while existing measured reports keep the bank and graph gates. Add bounded, sampled full-model-forward CUDA events and host enqueue durations for eager diagnostics; read events only after the owned service is idle. Keep prefill and decode separate and record actual forward row counts.

Execution stays on branch `codex/c32-fixed-diagnostic` in the existing checkout. The untracked GPT6 Pro handoff document is separate from this change. Server dependencies, model, image and four-GPU allocation follow AGENTS.md.

- [x] Add failing behavior tests for fixed/eager identity acceptance, plan mismatch rejection, full-forward sample aggregation, explicit launch options and export privacy.
- [x] Extend `execution_profile.py`, adapter metadata, launcher validation and the collector for explicitly identified fixed/eager diagnostics. Preserve measured-profile validation.
- [x] Add bounded eager-forward event sampling and integrate reset/snapshot with the existing idle evidence API. Test timing aggregation and sample lifecycle; add a server GPU acceptance exercise.
- [x] Add an owned C32 diagnostic driver that validates core/checkpoint/image, starts the exact tiled fixed/eager service, runs the existing installed SGLang benchmark with the existing local dataset, stops only its own load, drains/aborts owned requests, collects four ranks, and exports aggregate evidence with SHA256. Failure paths must preserve stage and cleanup evidence without exporting prompts or raw logs.
- [x] Add an offline server runbook/DeepSeek V4 Flash prompt with exact flags, four allocated GPUs, existing dataset discovery, and no online installation. Add source-aware interpretation of the phase and full-forward clocks.
- [x] Run focused CPU tests, inspect the diff and perform an independent code review. Delivery uses the established GitHub workflow; GPU timing and actual serving validation remain server-side.

Local verification: 41 focused CPU tests passed. Full suite: 180 tests, 7 failures and 3 errors; the same pre-existing GPQA provenance/order conflicts were reproduced in an archive of base4613c6d. Independent review found three issues, now fixed and re-reviewed with no remaining P1/P2. GPU smoke and real C32 execution await the server.
