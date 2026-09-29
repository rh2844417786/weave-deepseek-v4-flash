"""Token-ready decode metrics shared by both serving modes."""

import math


def decode_metrics(output_tokens: int, first_token_s: float, last_token_s: float) -> dict:
    """Measure work after the first actual generated token ID.

    Timestamps must be captured at the same server-side token-ready boundary.
    This function does not infer token counts from text or streaming chunks.
    """
    if output_tokens < 0:
        raise ValueError("output_tokens must be nonnegative")
    if not math.isfinite(first_token_s) or not math.isfinite(last_token_s):
        raise ValueError("token timestamps must be finite")
    elapsed = last_token_s - first_token_s
    if elapsed < 0 or (output_tokens >= 2 and elapsed == 0):
        raise ValueError("last_token_s must follow first_token_s for decode work")
    decode_tokens = max(output_tokens - 1, 0)
    return {
        "decode_tokens": decode_tokens,
        "decode_seconds": elapsed,
        "decode_tps": decode_tokens / elapsed if decode_tokens else None,
        "tpot_ms": elapsed * 1000 / decode_tokens if decode_tokens else None,
    }
