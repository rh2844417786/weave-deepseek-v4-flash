# Server connectivity constraint

The user confirmed that the experiment server can retrieve material from GitHub but has no general internet access. Apply this constraint to all future code, dependencies, and runbooks in this repository.

- Use the server's existing model files, Docker images, and installed packages. Server jobs must not depend on PyPI, Hugging Face downloads, apt, or pulling container images.
- Fetch required public data from pinned GitHub sources and verify its SHA256 before use. The existing GPQA fetcher uses the author's GitHub archive.
- Prepare any new external dependencies on a connected development machine. Provide a versioned, hash-checked GitHub delivery path for artifacts compatible with the server's Linux x86_64, Python, CUDA, and Torch contract; do not substitute local macOS binaries.
- Keep CPU data preparation offline after the GitHub download. Prefer the existing SGLang image when the server host lacks Python packages. Do not prescribe server-side online pip installation.
- Model loading must use the read-only local checkpoint and offline Hugging Face settings. Missing files or packages must produce an actionable error rather than trigger an automatic download.
- Preserve localhost HTTP and inter-rank communication required by the serving experiment. Offline CPU preparation can use Docker's `--network=none`; serving still needs its documented local network endpoints.

# Numerical error metrics are record-only

The user explicitly removed the Hopper numerical-error gates on 2026-10-17 and instructed that they must not be reintroduced.

- For Hopper and future experimental expert-compute comparisons, record finite numerical differences against the FP8 oracle, BF16 oracle, and BF16 reference. Do not stop runs, reject timings, label results failed, or impose promotion restrictions based on relative L2, absolute error, `allclose`, or a replacement numerical-error cutoff.
- Do not add new default or configurable error thresholds unless the user explicitly changes this decision. Current reports must declare `numerical_policy: record_only`.
- Retain checks for nonfinite output, invalid memory access, compilation/runtime failure, malformed tensors/routing, source identity, complete case coverage and cleanup. Deterministic algorithm unit tests remain applicable.
- Preserve historical reports and their original status/source identity. Do not rewrite an old threshold-failed run as a completed run; rerun under the new policy and evidence schema.
- In the Hopper C32 serving suite, generated-token agreement is also record-only: publish matched/mismatched counts and digests without stopping the throughput comparison for differing token contents. Continue requiring complete valid responses, request identity, expected token counts, finite output, and source/runtime/coverage evidence. Historical strict suites retain their recorded semantics.

# Paper-method identity

The user explicitly restored the paper-reproduction objective on 2026-10-18.
- `hopper_fp8` is an operator candidate; its standalone compute + NCCL slot SUM is not the Weave paper scheduler. Do not present its results as complete Weave.
- `hopper_persistent_paper_v1` contains the paper-mechanism integration, but source/CPU checks do not prove GPU execution or full reproduction. Report source implementation, kernel mechanism evidence, actual 43-layer service evidence and paper experiment coverage separately.
- Require actual routing-driven c/K decisions, resident SM evidence, chunk overlap and stealing evidence at their stated scopes. Preserve failures and observed-zero mechanisms; do not replace missing evidence with enabled flags or fabricated profiles.
- Keep historical reports unchanged. Numerical differences remain record-only under the policy above.
