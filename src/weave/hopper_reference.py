"""Independent FP8-activation/MXFP4 Torch oracle, outside timed execution.

No candidate extension or native decoder is used. Scalar encoding is a small
IEEE-style bit-format reference; Torch encoding selects the nearest point from
its finite codebook. GEMMs explicitly accumulate increasing K32 partials,
applying the original E8M0 and activation scales after each partial.
"""
import bisect
import math
import struct


def decode_fp8_e4m3(code):
    """Decode one E4M3FN byte, preserving signed zero and both NaN codes."""
    if type(code) is not int or not 0 <= code <= 255:
        raise ValueError('an FP8 code must be an integer byte')
    magnitude, sign = code & 127, -1. if code & 128 else 1.
    exponent, fraction = magnitude >> 3, magnitude & 7
    if magnitude == 127:
        return math.nan
    if exponent == 0:
        return math.copysign(math.ldexp(float(fraction), -9), sign)
    return sign * math.ldexp(1. + fraction / 8., exponent - 7)


_FINITE_FP8 = tuple(decode_fp8_e4m3(code) for code in range(127))


def encode_fp8_e4m3(value):
    """RNE with saturation to finite +/-448; reject NaN, preserve -0."""
    value = float(value)
    if math.isnan(value):
        raise ValueError('NaN has no finite FP8 quantization')
    sign = 128 if math.copysign(1., value) < 0 else 0
    magnitude = abs(value)
    if magnitude >= 448.:
        return sign | 126
    upper = bisect.bisect_left(_FINITE_FP8, magnitude)
    if upper == 0:
        return sign
    lower = upper - 1
    left, right = magnitude - _FINITE_FP8[lower], _FINITE_FP8[upper] - magnitude
    code = lower if left < right or (left == right and lower % 2 == 0) else upper
    return sign | code


def _f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def quantize_fp8_rows(rows):
    """CPU scalar oracle for finite rows whose values already represent BF16.

    Returns nested lists (dequantized FP32, byte codes, per128 FP32 scales).
    This helper does not introduce a second BF16 input rounding operation.
    """
    rows = [list(row) for row in rows]
    if not rows or not rows[0] or len(rows[0]) % 128 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('nonempty rectangular rows with width a multiple of 128 are required')
    dequantized, encoded, all_scales = [], [], []
    for row in rows:
        values, codes, scales = [], [], []
        for start in range(0, len(row), 128):
            block = [_f32(float(value)) for value in row[start:start + 128]]
            if not all(math.isfinite(value) for value in block):
                raise ValueError('activation values must be finite')
            peak = max(abs(value) for value in block)
            scale = _f32(peak / 448.) if peak else 1.
            if scale == 0.:
                raise ValueError('nonzero activation scale underflows FP32')
            block_codes = [encode_fp8_e4m3(_f32(value / scale)) for value in block]
            values.extend(_f32(decode_fp8_e4m3(code) * scale) for code in block_codes)
            codes.extend(block_codes)
            scales.append(scale)
        dequantized.append(values)
        encoded.append(codes)
        all_scales.append(scales)
    return dequantized, encoded, all_scales


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('run the independent oracle in the existing Torch server image; '
                           'no dependency download is attempted') from exc
    return torch


def _quantize_tensor(tensor, torch):
    if tensor.ndim != 2 or tensor.shape[1] == 0 or tensor.shape[1] % 128 or tensor.dtype != torch.bfloat16:
        raise ValueError('quantization requires BF16 [rows, width] with width a positive multiple of 128')
    blocks = tensor.float().reshape(tensor.shape[0], tensor.shape[1] // 128, 128)
    if not bool(torch.isfinite(blocks).all()):
        raise ValueError('activation values must be finite')
    peaks = blocks.abs().amax(dim=-1)
    scales = torch.where(peaks == 0., torch.ones_like(peaks), peaks / 448.)
    if bool((scales == 0.).any()):
        raise ValueError('nonzero activation scale underflows FP32')
    normalized = blocks / scales.unsqueeze(-1)
    codebook = torch.tensor(_FINITE_FP8, dtype=torch.float32, device=tensor.device)
    magnitude = normalized.abs().clamp(max=448.).contiguous()
    upper = torch.searchsorted(codebook, magnitude).clamp(max=126)
    lower = (upper - 1).clamp(min=0)
    left, right = magnitude - codebook[lower], codebook[upper] - magnitude
    select_lower = (left < right) | ((left == right) & ((lower & 1) == 0))
    absolute_codes = torch.where(select_lower, lower, upper)
    negative = torch.signbit(normalized)
    codes = (absolute_codes | (negative.to(torch.int64) << 7)).to(torch.uint8)
    values = torch.where(negative, -codebook[absolute_codes], codebook[absolute_codes])
    return values.reshape(tensor.shape), codes.reshape(tensor.shape), scales


def quantize_fp8_reference(tensor):
    """Return (dequantized FP32, uint8 codes, [rows,width/128] FP32 scales)."""
    torch = _torch()
    values, codes, scales = _quantize_tensor(tensor, torch)
    dequant = (values.reshape(tensor.shape[0], -1, 128) * scales.unsqueeze(-1)).reshape(tensor.shape)
    return dequant, codes, scales


def _mxfp4_values(packed, scales, torch):
    # Decode nibble pairs directly from original checkpoint bytes. No candidate
    # decoder or prepacked/reformatted weights are involved.
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=packed.device, dtype=torch.float32)
    values = lut[(nibbles & 7).long()]
    values = torch.where((nibbles & 8) != 0, -values, values)
    if bool((scales == 255).any()):
        raise ValueError('nonfinite original E8M0 expert scale')
    # E8M0 exponent byte zero means 2^-127, not a zero scale.
    factors = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)
    return values, factors


def _partial_gemm(activation, packed, weight_scales, torch):
    values, _, scales = _quantize_tensor(activation, torch)
    weights, factors = _mxfp4_values(packed, weight_scales, torch)
    result = torch.zeros((activation.shape[0], weights.shape[0]), dtype=torch.float32, device=activation.device)
    for k in range(0, activation.shape[1], 32):
        partial = values[:, k:k + 32] @ weights[:, k:k + 32].T
        partial = partial * scales[:, k // 128].unsqueeze(1)
        partial = partial * factors[:, k // 32].unsqueeze(0)
        result = result + partial
    return result.bfloat16()


def expert_reference_fp8(x, raw, expert, clamp=10.):
    """One local expert's BF16 output, before BF16 route/product rounding.

    ``raw`` is (w1,w3,w2,s1,s3,s2), still original uint8 MXFP4/E8M0.
    This oracle is intentionally synchronization-heavy and must stay outside
    benchmark timing and CUDA graph capture. It never calls the candidate.
    """
    torch = _torch()
    if len(raw) != 6 or type(expert) is not int or not math.isfinite(clamp) or clamp <= 0:
        raise ValueError('six original tensors, a local expert index, and a finite positive clamp are required')
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] == 0 or x.shape[1] % 128:
        raise ValueError('expert reference requires BF16 [rows,H] and H a positive multiple of 128')
    if any(value.ndim != 3 or value.dtype != torch.uint8 or value.device != x.device for value in raw):
        raise ValueError('all six raw expert tensors must be uint8 [experts,rows,columns] on the activation device')
    experts, intermediate = raw[0].shape[:2]
    hidden = x.shape[1]
    expected = [(experts, intermediate, hidden // 2), (experts, intermediate, hidden // 2),
                (experts, hidden, intermediate // 2), (experts, intermediate, hidden // 32),
                (experts, intermediate, hidden // 32), (experts, hidden, intermediate // 32)]
    if not 0 <= expert < experts or intermediate == 0 or intermediate % 128 or any(tuple(value.shape) != shape for value, shape in zip(raw, expected)):
        raise ValueError('raw MXFP4/E8M0 shapes or local expert index do not match H/I block128 contract')
    # FP8 and E2M1 operands are exactly representable in FP32. Force the Torch
    # reference away from TF32/reduced-precision matmul independently of the
    # caller's global setting, then restore that setting even on failure.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        gate = _partial_gemm(x, raw[0][expert], raw[3][expert], torch).float().clamp(max=clamp)
        up = _partial_gemm(x, raw[1][expert], raw[4][expert], torch).float().clamp(-clamp, clamp)
        # Same existing BF16 storage boundaries: GEMM, F.silu, product, down.
        activation = (gate / (1. + torch.exp(-gate))).bfloat16()
        middle = (activation.float() * up).bfloat16()
        return _partial_gemm(middle, raw[2][expert], raw[5][expert], torch)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
