"""Compact Hopper prefill orchestration with one active-slot SUM.

The workspace performs ordered assembly and origin masking. Sequential layers
share storage; their bound handles retain the immutable raw checkpoint tensors.
"""
import hashlib
from pathlib import Path
import re


def runtime_source_sha256():
    """Identity of this loaded runtime, independent of the decode implementation."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


_LOADED_RUNTIME_SOURCE_SHA256 = runtime_source_sha256()


class HopperPrefillRuntime:
    max_batch = 8192

    def __init__(self, peer_runtime, device_group, experts=256, max_batch=8192):
        from .hopper_prefill_candidate import extension
        if type(max_batch) is not int or not 1 <= max_batch <= 8192:
            raise ValueError('Hopper prefill workspace capacity must be 1..8192')
        self.max_batch = max_batch
        self.torch, self.device_group = peer_runtime.torch, device_group
        self.rank, self.ep = peer_runtime.rank, peer_runtime.ep
        self.hidden, self.topk = peer_runtime.h, peer_runtime.topk
        module = extension()
        identity = getattr(module, 'weave_source_sha256', None)
        if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{64}', identity):
            raise RuntimeError('loaded Hopper prefill candidate has no valid source identity')
        self.candidate_source_sha256 = identity
        self.runtime_source_sha256 = _LOADED_RUNTIME_SOURCE_SHA256
        self.ws = module.PrefillWorkspace(self.rank, self.ep, self.max_batch,
            self.hidden, peer_runtime.i, experts, self.topk)
        self.resources = dict(self.ws.resources())
        slots = self.ws.slot_outputs
        self._slots = (None,) + tuple(slots[:n * self.topk]
                                     for n in range(1, self.max_batch + 1))
        self.additional_slot_collectives = 0

    def bind_weights(self, raw):
        return self.ws.bind_weights(list(raw))

    def run(self, x, ids, routing, bound, clamp=10):
        batch = x.shape[0]
        if not 1 <= batch <= self.max_batch:
            raise ValueError(f'hopper_fp8 prefill supports batches 1..{self.max_batch}; '
                             'set chunked-prefill-size to at most 8192')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('hopper_fp8 prefill requires eager execution')
        self.ws.run(x, ids, routing, bound, clamp)
        dist = self.torch.distributed
        dist.all_reduce(self._slots[batch], op=dist.ReduceOp.SUM, group=self.device_group)
        self.additional_slot_collectives += 1
        return self.ws.finalize(batch)
