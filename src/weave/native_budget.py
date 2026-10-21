"""Native Marlin CTA budget policy and public aggregate evidence (no Torch import).

This is a routing heuristic, not a fitted cost model or physical SM partition.
The device implementation lives in csrc/native_budget.h.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_SIZE = 19
METHOD = 'native_marlin_dynamic_cta_v1'
ROLES = ('gate_up', 'down')


def select_workers(native_grid, local_routes, total_routes, local_blocks, n_tiles, k_tiles,
                   *, target_percent=75, ep_size=4, mode='adaptive'):
    values = (native_grid, local_routes, total_routes, local_blocks, n_tiles, k_tiles,
              target_percent, ep_size)
    if (any(type(x) is not int for x in values) or native_grid < 1 or total_routes < 1
            or not 0 <= local_routes <= total_routes or local_blocks < 0
            or min(n_tiles, k_tiles, ep_size) < 1 or not 1 <= target_percent <= 100
            or mode not in ('adaptive', 'full')
            or (local_routes == 0) != (local_blocks == 0)):
        raise ValueError('invalid native budget input/configuration')
    if not local_routes:
        return 0, 0
    if mode == 'full':
        return native_grid, 8
    numerator = native_grid * local_routes * ep_size * target_percent
    denominator = total_routes * 100
    desired = max((native_grid + 3) // 4,
                  min(native_grid, (numerator + denominator - 1) // denominator))
    bucket = next(i for i in range(1, 9) if (native_grid * i + 7) // 8 >= desired)
    workers = min((native_grid * bucket + 7) // 8, local_blocks * n_tiles * k_tiles)
    return workers, min(8, (workers * 8 + native_grid - 1) // native_grid)


def state_row(layer, role, values):
    if role not in ROLES or len(values) != STATE_SIZE or any(type(x) is not int or x < 0 for x in values):
        raise ValueError('malformed native budget state')
    keys = ('calls', 'empty_calls', 'native_grid', 'last_active_workers',
            'last_global_blocks', 'last_local_blocks', 'last_n_tiles', 'last_k_tiles')
    return dict(layer=layer, role=role, **dict(zip(keys, values[:8])),
                bucket_counts=values[8:17], last_local_routes=values[17], last_total_routes=values[18])


def source_pins():
    return json.loads((ROOT / 'configs/native_budget_sources.json').read_text())['sha256']


def native_budget_source_sha256():
    digest = hashlib.sha256()
    for name in ('src/weave/native_budget.py', 'src/weave/native_budget_patch.py',
                 'src/weave/native_budget_runtime.py', 'csrc/native_budget.h',
                 'configs/native_budget_sources.json'):
        digest.update(name.encode() + b'\0' + (ROOT / name).read_bytes())
    return digest.hexdigest()
