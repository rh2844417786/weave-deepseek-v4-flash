"""CPU oracle for EP token ownership, payload deduplication and weighted combine.

This module consumes final native-router choices and weights. It does not run
GPU collectives, quantized GEMMs, shared experts or a Weave serving backend.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Tuple


@dataclass(frozen=True, order=True)
class TokenKey:
    origin_rank: int
    token_index: int


@dataclass(frozen=True)
class TokenRoute:
    token: TokenKey
    expert_ids: Tuple[int, ...]
    weights: Tuple[float, ...]


@dataclass(frozen=True)
class ExpertTask:
    token: TokenKey
    slot: int
    expert_id: int
    local_expert: int
    weight: float


@dataclass(frozen=True)
class RankWorkload:
    local_pairs: int
    remote_pairs: int
    remote_unique_tokens: int


@dataclass(frozen=True)
class DispatchPlan:
    routes: Tuple[TokenRoute, ...]
    num_experts: int
    ep_size: int
    top_k: int
    dispatch_tokens: Tuple[Tuple[TokenKey, ...], ...]
    tasks: Tuple[Tuple[ExpertTask, ...], ...]
    workloads: Tuple[RankWorkload, ...]


@dataclass(frozen=True)
class ExpertResult:
    token: TokenKey
    slot: int
    values: Tuple[float, ...]


def _integer(value, label: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def build_dispatch_plan(
    routes: Iterable[TokenRoute], num_experts: int = 256, ep_size: int = 4, top_k: int = 6
) -> DispatchPlan:
    """Partition experts contiguously; send one token payload per destination.

    All ranks, including ranks receiving zero tokens, have entries in the plan.
    Workload counts are inbound counts for the destination rank.
    """
    for value, label in ((num_experts, 'num_experts'), (ep_size, 'ep_size'), (top_k, 'top_k')):
        _integer(value, label, 1)
    if num_experts % ep_size or top_k > num_experts:
        raise ValueError("experts must divide evenly across ranks and cover top_k")
    per_rank = num_experts // ep_size
    normalized = []
    seen_tokens = set()
    payloads = [dict() for _ in range(ep_size)]
    tasks = [[] for _ in range(ep_size)]
    for route in routes:
        token = route.token
        _integer(token.origin_rank, 'origin_rank')
        _integer(token.token_index, 'token_index')
        if token.origin_rank >= ep_size:
            raise ValueError("origin_rank is outside the EP group")
        if token in seen_tokens:
            raise ValueError(f"duplicate token owner: {token}")
        seen_tokens.add(token)
        ids, weights = tuple(route.expert_ids), tuple(route.weights)
        if len(ids) != top_k or len(weights) != top_k:
            raise ValueError("each token must have exactly top_k experts and weights")
        for expert in ids:
            _integer(expert, 'expert_id')
            if expert >= num_experts:
                raise ValueError("expert_id outside the expert set")
        if len(set(ids)) != top_k:
            raise ValueError("duplicate expert for a token")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("routing weights must be finite and nonnegative")
        normalized.append(TokenRoute(token, ids, weights))
        for slot, (expert, weight) in enumerate(zip(ids, weights)):
            rank = expert // per_rank
            payloads[rank][token] = None
            tasks[rank].append(ExpertTask(token, slot, expert, expert % per_rank, weight))
    workloads = []
    for rank in range(ep_size):
        local = sum(task.token.origin_rank == rank for task in tasks[rank])
        remote_unique = sum(token.origin_rank != rank for token in payloads[rank])
        workloads.append(RankWorkload(local, len(tasks[rank]) - local, remote_unique))
    return DispatchPlan(
        tuple(normalized), num_experts, ep_size, top_k,
        tuple(tuple(tokens) for tokens in payloads),
        tuple(tuple(rank_tasks) for rank_tasks in tasks), tuple(workloads),
    )


def combine_results(plan: DispatchPlan, results: Iterable[ExpertResult], hidden_size: int) -> dict:
    """Merge unweighted expert outputs, enforcing one return per selected slot.

    Arrival order is irrelevant. Weighting follows original top-k slot order;
    native routing scales are applied exactly once, with no renormalization.
    Shared-expert output must be added by the caller in the native runtime.
    """
    _integer(hidden_size, 'hidden_size', 1)
    expected = {(route.token, slot) for route in plan.routes for slot in range(plan.top_k)}
    received = {}
    for result in results:
        _integer(result.slot, 'slot')
        key = (result.token, result.slot)
        if key not in expected:
            raise ValueError(f"unexpected result slot: {key}")
        if key in received:
            raise ValueError(f"duplicate result slot: {key}")
        values = tuple(result.values)
        if len(values) != hidden_size or any(not math.isfinite(value) for value in values):
            raise ValueError("expert output must have hidden_size finite values")
        received[key] = values
    if len(received) != len(expected):
        raise ValueError(f"missing {len(expected) - len(received)} result slots")
    return {
        route.token: tuple(
            math.fsum(received[(route.token, slot)][column] * route.weights[slot]
                      for slot in range(plan.top_k))
            for column in range(hidden_size)
        )
        for route in plan.routes
    }
