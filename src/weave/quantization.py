"""Small CPU oracle for checkpoint MXFP4 E2M1 and E8M0 block scales."""
import math

E2M1 = (0., .5, 1., 1.5, 2., 3., 4., 6.)


def decode_row(packed: bytes, scales: bytes) -> tuple:
    if len(packed) == 0 or len(packed) % 16 or len(scales) != len(packed) // 16:
        raise ValueError('one E8M0 scale required for each 32 logical values')
    output = []
    for column in range(2 * len(packed)):
        exponent = scales[column // 32]
        if exponent == 255:
            raise ValueError('E8M0 NaN scale in checkpoint')
        bits = (packed[column // 2] >> (4 * (column % 2))) & 15
        value = math.ldexp(E2M1[bits & 7], exponent - 127)
        output.append(-value if bits & 8 else value)
    return tuple(output)
