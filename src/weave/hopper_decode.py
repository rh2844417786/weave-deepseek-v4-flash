"""Eager FP8 expert compute, one slot SUM, and ordered origin assembly.

The validated candidate remains unchanged. One workspace is shared by sequential
layers; each bound handle keeps its immutable raw checkpoint tensors alive.
"""
import hashlib
import json
from pathlib import Path
import re

from .hopper_finalize import finalize_slots, prepare_finalizer


def assembly_source_sha256():
    """Identity of the runtime and separate Triton finalizer, without GPU imports."""
    root = Path(__file__).resolve().parent
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in ('hopper_decode.py', 'hopper_finalize.py')}
    return hashlib.sha256(json.dumps(files, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


_LOADED_ASSEMBLY_SOURCE_SHA256 = assembly_source_sha256()


class HopperDecodeRuntime:
    """Preallocated full-input decode; output prefixes remain valid until reuse."""
    max_batch = 32

    def __init__(self, peer_runtime, device_group, experts=256):
        from .hopper_candidate import extension
        self.torch, self.device_group = peer_runtime.torch, device_group
        self.rank, self.ep = peer_runtime.rank, peer_runtime.ep
        self.hidden, self.topk = peer_runtime.h, peer_runtime.topk
        module = extension()
        identity = getattr(module, 'weave_source_sha256', None)
        if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{64}', identity):
            raise RuntimeError('loaded Hopper candidate has no valid source identity')
        self.candidate_source_sha256 = identity
        self.assembly_source_sha256 = _LOADED_ASSEMBLY_SOURCE_SHA256
        self.ws = module.FP8Workspace(self.rank, self.ep, self.max_batch,
            self.hidden, peer_runtime.i, experts, self.topk)
        self.resources = dict(self.ws.resources())
        self._output = self.torch.empty((self.max_batch, self.hidden),
            dtype=self.torch.bfloat16, device=f'cuda:{peer_runtime.device}')
        slots = self.ws.slot_outputs
        self._slots = (None,) + tuple(slots[:n * self.topk]
                                     for n in range(1, self.max_batch + 1))
        self._outputs = (None,) + tuple(self._output[:n]
                                       for n in range(1, self.max_batch + 1))
        self.additional_slot_collectives = 0
        prepare_finalizer()

    def bind_weights(self, raw):
        return self.ws.bind_weights(list(raw))

    def run(self, x, ids, routing, bound, clamp=10):
        batch = x.shape[0]
        if not 1 <= batch <= self.max_batch:
            raise ValueError('hopper_fp8 decode supports batches 1..32 only')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('hopper_fp8 requires eager decode')
        self.ws.run(x, ids, routing, bound, clamp)
        dist = self.torch.distributed
        dist.all_reduce(self._slots[batch], op=dist.ReduceOp.SUM, group=self.device_group)
        self.additional_slot_collectives += 1
        finalize_slots(self._slots[batch], self._output, batch,
                       self.hidden, self.topk, self.rank, self.ep)
        return self._outputs[batch]
