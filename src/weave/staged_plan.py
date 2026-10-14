"""Offline, deterministic groups for the P1 staged compute control.

Inputs are CPU integer rows from a validated private trace. This module never
imports torch or downloads data. Upload its frozen metadata once before timing
or CUDA graph capture. Group creation follows the persistent kernel's K1
first-encounter order; each expert's rows retain original token/slot order.
"""
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class StagedGroups:
    batch: int
    topk: int
    rank: int
    ep: int
    experts: int
    group_meta: tuple
    group_slots: tuple
    group_count: int

    @property
    def capacity(self):
        return len(self.group_meta)

    @property
    def local_pairs(self):
        return sum(rows for _, rows in self.group_meta[:self.group_count])


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def build_staged_groups(ids, *, rank, ep, experts, topk=6, capacity=None):
    """Return padded metadata and original ``token * topk + slot`` indices.

    ``group_meta`` has rows ``(local_expert, valid_rows)``; ``group_slots`` has
    16 entries per group with -1 padding. ``group_count`` excludes padding.
    The default capacity is the exact live group count, or one for an empty
    rank, so grid=0 always has a legal host-known launch dimension.
    """
    rank, ep, experts, topk = (_integer(v, n) for v, n in
                               ((rank, 'rank'), (ep, 'ep'), (experts, 'experts'), (topk, 'topk')))
    if not (1 <= ep <= 4 and 0 <= rank < ep and 1 <= topk <= 16
            and experts >= topk and experts % ep == 0):
        raise ValueError('invalid rank, EP, expert count, or topk')
    rows = [list(row) for row in ids]
    if not 1 <= len(rows) <= 32:
        raise ValueError('staged replay requires a batch in 1..32')
    local_experts = experts // ep
    groups, heads = [], {}
    for token, row in enumerate(rows):
        if len(row) != topk:
            raise ValueError('every routing row must contain exactly topk IDs')
        row = [_integer(expert, 'expert ID') for expert in row]
        if any(expert < 0 or expert >= experts for expert in row):
            raise ValueError('expert ID is outside the global expert range')
        if len(set(row)) != topk:
            raise ValueError('duplicate expert ID within one token')
        for slot, expert in enumerate(row):
            owner, local = divmod(expert, local_experts)
            if owner != rank:
                continue
            group = heads.get(local)
            if group is None or len(groups[group][1]) == 16:
                group = len(groups)
                heads[local] = group
                groups.append((local, []))
            groups[group][1].append(token * topk + slot)
    count = len(groups)
    capacity = max(1, count) if capacity is None else _integer(capacity, 'capacity')
    if capacity < max(1, count) or capacity > len(rows) * topk:
        raise ValueError('group capacity must cover all live groups and not exceed route count')
    meta = tuple((expert, len(slots)) for expert, slots in groups)
    slots = tuple(tuple(values) + (-1,) * (16 - len(values)) for _, values in groups)
    return StagedGroups(len(rows), topk, rank, ep, experts,
                        meta + ((-1, 0),) * (capacity - count),
                        slots + ((-1,) * 16,) * (capacity - count), count)
