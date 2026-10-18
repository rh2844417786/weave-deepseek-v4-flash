# Third party notices for the experimental Hopper candidate

## NVIDIA CUTLASS and CUTE

`third_party/cutlass_hopper/include` is a minimal unmodified header closure from
https://github.com/NVIDIA/cutlass at commit
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`. Its BSD-style license is reproduced in
`third_party/cutlass_hopper/LICENSE.txt`; original copyright/license headers are
retained. The source/archive hashes are in the adjacent manifest.

## DeepGEMM and the supplied Weave research fork

The grouped SM90 TMA/WGMMA mapping, nibble lookup, B128 shared-memory layout and
K32 scale application in `experimental/hopper_fp8/kernel.cu` and
`math_contract.h` were adapted with reference to the user-supplied
`rh2844417786/weave-deepgemm-repro` commit
`4fb1e6ff12e184c50b90c0c35156783ee52f0196`, itself a DeepGEMM research fork.
Relevant source paths and hashes are recorded in
`experimental/hopper_fp8/reference_provenance.json`.

This project adds its own Torch workspace/binding, B32 online packing, explicit
FP8-activation contract, numerical oracle, archived-trace comparison and
evidence gates. It does not import the fork's SGLang overlay or use its reported
EP1 short-request performance as a performance claim for this candidate.

The supplied fork's MIT license is reproduced in
`experimental/hopper_fp8/LICENSE.reference.txt`.

## Compact Hopper prefill adaptation

`experimental/hopper_prefill/kernel.cu` and `math_contract.h` extend the same audited TMA/WGMMA and packed-MXFP4 mapping to compact multi-tile expert groups. The original license and provenance are also retained under `experimental/hopper_prefill`. Grouping, workspace sizing, multi-tile addressing and ordered distributed assembly are repository additions; the existing decode candidate remains independently versioned.
