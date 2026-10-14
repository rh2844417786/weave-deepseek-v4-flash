# Staged compute control contract

This P1 control uses the same M16 x N64 expert arithmetic as the persistent
kernel. It measures three separate launches on one current CUDA stream. It
does not measure dispatch, IPC, cross-rank waits, or service throughput.

## Arithmetic extraction audit

The extraction baseline is `46a2e1d826874b2dde3341eb7d8b47c2de9ba686`,
`csrc/megakernel.cu`. Shared helpers now live in `csrc/tile_math.cuh`.

| Original operation | Shared implementation and preserved behavior |
| --- | --- |
| `decode4` and `weight` | Same nibble order, sign handling, eight literal magnitudes, E8M0 exponent and scale-255 NaN. No new decode table. |
| `group_mma` A load | Same 16-row padding, BF16 inputs and column order; storage access is supplied by a functor. Persistent reads its payload cache/group mid; staged reads original global token rows/group mid. |
| `group_mma` B load | Same packed weight/scales address, BF16 conversion, output-column padding, four warps with 16 columns each. |
| WMMA accumulation | Same 16 x 16 x 16 BF16 fragments, FP32 accumulator, zero initialization, ascending K tiles, warp synchronizations, and row-major FP32 store. |
| Gate/up boundary | Compute gate first, retain eight FP32 results per lane, compute up, round both to BF16, then call unchanged `swiglu_product`. |
| SwiGLU | `expert_numerics.h` still clamps gate above and up on both sides, rounds SiLU to BF16 before multiplication, and rounds the product before BF16 mid storage. |
| Down boundary | Same group MMA and BF16 rounding of each result, then store the rounded value as FP32. |
| Routing | Unchanged `routed_product`: routing weight rounds to BF16 before multiplying the BF16-rounded down output; product rounds to BF16. |
| Calibration | Persistent calibration 1 still skips both MMAs and stores zero at the same gate/up and down positions. |

Persistent `compute_tile` retains input readiness waits, CTA barriers, memory
fence, tile/task counters, gate/down completion publication, and trace calls.
Persistent planning, dispatch, combine, peer protocol and worker grid are not
part of this extraction. Compiled resource usage can change after extraction
and must be reported from both persistent specializations and all staged kernels.

## Python interface

```python
ws = extension().StagedWorkspace(rank, ep, batch, h, i, experts, topk)
ws.bind_weights(raw_six)  # retains the original uint8 CUDA tensors
ws.bind_inputs(x, ids, routing, group_meta, group_slots, group_count)

# Bind before timing; use one current stream and this stage order.
ws.run_gate_up(grid, checkpoint_swiglu_limit)
ws.run_down(grid)
ws.run_reduce()
local_output = ws.output       # FP32 [batch, h]
local_slots = ws.slot_outputs  # FP32 [batch * topk, h]
```

The convenience `run(x, ids, routing, group_meta, group_slots, group_count,
grid, clamp=10.0)` binds inputs then calls those three stages. Replay must pass
the checkpoint's actual `swiglu_limit`. Stable output and mid buffers allocate
in construction; bound tensors remain owned by the workspace. Stage methods
perform no allocation, device-to-host read, `.item()`, or synchronization.

`build_staged_groups(ids_cpu, rank=..., ep=..., experts=..., topk=6,
capacity=None)` in `src/weave/staged_plan.py` validates IDs and returns frozen
`StagedGroups`. Upload `group_meta`, `group_slots`, and `[group_count]` as int32
tensors once before timing. Groups follow the persistent K1 first-encounter
order: scan token then slot, append a new group when first seeing an expert or
when its current group reaches 16 rows. Staged execution has no communication
chunks. Padding is `(-1, 0)` for metadata and `-1` for unused slots. An empty
rank has zero active groups and a minimum metadata capacity of one.

Grid 132/264/528 controls gate/up and down with fixed strided tile loops. Grid
0 uses `metadata_capacity * ceil(output_width / 64)` CTAs; each invocation
reads the live GPU group count and ignores inactive capacity. Reduction uses
`ceil(batch * h / 128)` CTAs for all variants. Each thread sums one output
column in original slot order and selects only this rank's experts. Nonlocal
slot outputs become zero. The local FP32 partials are not wired into SGLang;
final validation must assemble the original global slot order outside timing.

`resources()` reports `stages[gate_up/down/reduce]` with `numRegs`,
`localSizeBytes`, `sharedSizeBytes`, `maxDynamicSharedSizeBytes`,
`dynamic_shared_bytes`, `threads_per_block`, `active_blocks_per_sm`,
`binaryVersion`, and `ptxVersion`, plus grid and capacity metadata.

## Validation boundary

CPU tests exercise group ownership and ordering, full/partial/empty groups,
duplicate/invalid route rejection, padding, and the actual host/device grid
helper's exact tile coverage. Existing host expert-numerics witnesses still
check BF16 SiLU and routing-product boundaries.

On an allocated H100 in the existing image, run:

```bash
python tests/gpu/check_staged.py --output /private/results/staged-synthetic.json
```

This gate uses small original-format synthetic weights and the independent
checkpoint tensor oracle, tests all four grids, binding rejection, stable
addresses, graph replay, changed device group counts, and an empty rank. It
does not replace the real 43-layer checkpoint replay gate. CUDA compilation,
GPU numerical results, graph behavior and performance require server evidence;
local CPU tests do not establish any of them.
