# Source fixtures for native Marlin budget and tail extraction

Unmodified source files from SGLang commit 29481685462732237d80d86076d6563e1f658102, used for offline patch validation. SHA256 values are in configs/native_budget_sources.json. Upstream licenses and copyright notices are preserved in each file; SGLang is Apache-2.0 licensed. Runtime requires matching installed files and uses installed dependencies from the existing Linux image. No binaries are included.

The fused Marlin MoE and runner Python files also exercise the tail front extraction
against the complete pinned native control flow in CPU tests. They remain byte
identical to upstream; `configs/native_tail_sources.json` binds the tail source
identity. CPU stand-ins replace only tensor storage and CUDA operators in these
tests, so passing them does not establish CUDA compilation or GPU execution.
