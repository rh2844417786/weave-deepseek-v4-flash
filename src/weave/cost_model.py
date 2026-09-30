"""Weave equations 1-16, with explicit units and measured-profile inputs.

The functional MXFP4 kernel uses a whole token/expert forward as its steal work
unit. Dispatch is BF16 and return slots are FP32; the different byte widths are
explicit here instead of silently reusing the paper's single B value.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Workload:
    local_pairs: int
    remote_pairs: int
    remote_unique_tokens: int
    unique_tokens: int
    hidden_size: int
    intermediate_size: int
    dispatch_bytes: int = 2
    return_bytes: int = 4


@dataclass(frozen=True)
class Candidate:
    comm_workers: int
    chunks: int
    compute_flops_s: float
    bandwidth_bytes_s: float
    chunk_efficiency: float
    overlap: float


@dataclass(frozen=True)
class Plan:
    comm_workers: int
    chunks: int
    steal_tasks: int
    seconds: float


def select_plan(work: Workload, workers: int, candidates) -> Plan:
    for name in ('local_pairs', 'remote_pairs', 'remote_unique_tokens', 'unique_tokens'):
        value = getattr(work, name)
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if workers < 2 or work.hidden_size <= 0 or work.intermediate_size <= 0 or work.dispatch_bytes <= 0 or work.return_bytes <= 0:
        raise ValueError('positive dimensions and at least two workers required')
    if work.remote_unique_tokens > work.remote_pairs or work.unique_tokens > work.local_pairs + work.remote_pairs:
        raise ValueError('invalid unique token counts')
    pairs = work.local_pairs + work.remote_pairs
    if pairs == 0:
        return Plan(0, 1, 0, 0.)
    tile_flops = 6 * work.hidden_size * work.intermediate_size
    compute = pairs * tile_flops
    dispatch = work.remote_unique_tokens * work.hidden_size * work.dispatch_bytes
    combine = work.remote_pairs * work.hidden_size * work.return_bytes
    plans = []
    for row in candidates:
        c, k = row.comm_workers, row.chunks
        if type(c) is not int or type(k) is not int or not 0 <= c < workers or not 1 <= k <= min(8, work.unique_tokens):
            continue
        if c == 0 and work.remote_pairs:
            continue
        if not all(math.isfinite(x) for x in (row.compute_flops_s, row.bandwidth_bytes_s, row.chunk_efficiency, row.overlap)):
            continue
        if row.compute_flops_s <= 0 or not 0 < row.chunk_efficiency <= 1 or not 0 <= row.overlap < 1:
            continue
        if row.bandwidth_bytes_s < 0 or (c > 0 and row.bandwidth_bytes_s == 0):
            continue
        communication = (dispatch + combine) / row.bandwidth_bytes_s if c else 0.
        tail = (1 - row.overlap) * combine / (row.bandwidth_bytes_s * k) if c else 0.
        seconds = max(compute / (row.compute_flops_s * row.chunk_efficiency) + tail, communication)
        residual = max(0., compute - communication * row.compute_flops_s)
        steal = math.floor(residual / (workers * tile_flops)) if c else 0
        plans.append(Plan(c, k, steal, seconds))
    if not plans:
        raise ValueError('no valid measured scheduling candidates')
    return min(plans, key=lambda p: (p.seconds, p.chunks, p.comm_workers))
