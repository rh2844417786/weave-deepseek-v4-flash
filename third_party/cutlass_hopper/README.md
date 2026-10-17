# Pinned Hopper header closure

These62 unmodified headers come from NVIDIA/CUTLASS commit
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`, the exact CUTLASS gitlink in the
user-supplied `weave-deepgemm-repro` delivery at4fb1e6f. `LICENSE.txt` is the
upstream license. Original per-file license notices are retained.

`manifest.json` records the upstream archiveSHA256 and every included fileSHA256.
The roots are the five CUTE/CUTLASS SM90 headers required by the experimental
candidate. Quoted/project includes were traversed recursively, including
conditional includes, so the checked-in closure is self-contained with respect
to CUTE/CUTLASS. CUDA/Torch/system headers come from the existing server image.

The archive was obtained on the connected development machine from:
https://codeload.github.com/NVIDIA/cutlass/tar.gz/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8

Reconstruction on a development machine uses an already downloaded local archive:

```bash
python3 scripts/vendor_hopper_headers.py \
  --archive /absolute/path/cutlass-f3fde583.tar.gz \
  --destination /absolute/path/fresh-cutlass-hopper
```

Server jobs never run this preparation step or fetch headers. They consume the
committed files and `src/weave/hopper_candidate.py` verifies the exact manifest,
all contents and the include closure before importing Torch or invoking NVCC.
