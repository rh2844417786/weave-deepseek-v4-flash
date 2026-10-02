# Server connectivity constraint

The user confirmed that the experiment server can retrieve material from GitHub but has no general internet access. Apply this constraint to all future code, dependencies, and runbooks in this repository.

- Use the server's existing model files, Docker images, and installed packages. Server jobs must not depend on PyPI, Hugging Face downloads, apt, or pulling container images.
- Fetch required public data from pinned GitHub sources and verify its SHA256 before use. The existing GPQA fetcher uses the author's GitHub archive.
- Prepare any new external dependencies on a connected development machine. Provide a versioned, hash-checked GitHub delivery path for artifacts compatible with the server's Linux x86_64, Python, CUDA, and Torch contract; do not substitute local macOS binaries.
- Keep CPU data preparation offline after the GitHub download. Prefer the existing SGLang image when the server host lacks Python packages. Do not prescribe server-side online pip installation.
- Model loading must use the read-only local checkpoint and offline Hugging Face settings. Missing files or packages must produce an actionable error rather than trigger an automatic download.
- Preserve localhost HTTP and inter-rank communication required by the serving experiment. Offline CPU preparation can use Docker's `--network=none`; serving still needs its documented local network endpoints.
