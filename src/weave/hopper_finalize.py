"""Ordered origin assembly for Hopper slots, using the image's existing Triton.

Importing this module is CPU-safe. The kernel and its imports are prepared once
at runtime construction; all launches use the caller's current CUDA stream.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def prepare_finalizer():
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError('Hopper assembly requires the existing Triton server image; '
                           'no package download is attempted') from exc

    @triton.jit
    def ordered_finalize(Slots, Output, BATCH: tl.constexpr, HIDDEN: tl.constexpr,
                         TOPK: tl.constexpr, RANK: tl.constexpr, EP: tl.constexpr,
                         BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, col = offsets // HIDDEN, offsets % HIDDEN
        valid = offsets < BATCH * HIDDEN
        total = tl.full((BLOCK,), 0.0, tl.float32)
        for slot in tl.static_range(TOPK):
            value = tl.load(Slots + (row * TOPK + slot) * HIDDEN + col,
                            mask=valid, other=0.0)
            # Explicit PTX preserves sequential FP32 round-to-nearest addition;
            # the compiler must not reassociate a six-slot reduction.
            total = tl.inline_asm_elementwise(
                'add.rn.f32 $0, $1, $2;', constraints='=f,f,f',
                args=[total, value], dtype=tl.float32, is_pure=True, pack=1)
        chunk = (BATCH + EP - 1) // EP
        own = (row >= RANK * chunk) & (row < (RANK + 1) * chunk)
        result = tl.where(own, total, 0.0).to(tl.bfloat16, fp_downcast_rounding='rtne')
        tl.store(Output + offsets, result, mask=valid)

    return ordered_finalize


def finalize_slots(slots, output, batch, hidden, topk, rank, ep):
    """Enqueue ordered FP32 slot sum and BF16 RTNE into existing output storage."""
    if (type(batch) is not int or not 1 <= batch <= 32 or hidden <= 0
            or topk <= 0 or ep <= 0 or not 0 <= rank < ep):
        raise ValueError('invalid Hopper finalizer shape or rank metadata')
    prepare_finalizer()[((batch * hidden + 255) // 256,)](
        slots, output, batch, hidden, topk, rank, ep, BLOCK=256)
