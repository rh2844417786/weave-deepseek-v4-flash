"""Exact-batch CUDA graph Hopper decode with preallocated device accounting."""
import hashlib
import json
from pathlib import Path

from .hopper_decode import HopperDecodeRuntime, assembly_source_sha256
from .hopper_finalize import finalize_slots


def graph_runtime_source_sha256():
    """Identity includes this graph runtime and the unchanged eager assembly."""
    root = Path(__file__).resolve().parent
    files = {'hopper_graph.py': hashlib.sha256((root / 'hopper_graph.py').read_bytes()).hexdigest(),
             'assembly': assembly_source_sha256()}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


_LOADED_GRAPH_RUNTIME_SOURCE_SHA256 = graph_runtime_source_sha256()


class HopperGraphDecodeRuntime(HopperDecodeRuntime):
    """Exact B1..32 decode; all allocations precede graph capture.

    Each exact batch must finish an eager call before capture. Bound weights and
    workspace/output views use the unchanged eager implementation's ownership.
    The Python count is eager only; the device count includes actual replays.
    """

    def __init__(self, peer_runtime, device_group, experts=256):
        super().__init__(peer_runtime, device_group, experts)
        self.graph_runtime_source_sha256 = _LOADED_GRAPH_RUNTIME_SOURCE_SHA256
        self.device_slot_collectives = self.torch.zeros(
            (), dtype=self.torch.int64, device=f'cuda:{peer_runtime.device}')
        self._eager_warmed_batches = set()

    def run(self, x, ids, routing, bound, clamp=10):
        batch = x.shape[0]
        if not 1 <= batch <= self.max_batch:
            raise ValueError('hopper graph decode supports exact batches 1..32 only')
        capturing = self.torch.cuda.is_current_stream_capturing()
        if capturing and batch not in self._eager_warmed_batches:
            raise RuntimeError('hopper graph decode batch requires successful eager warmup')
        self.ws.run(x, ids, routing, bound, clamp)
        dist = self.torch.distributed
        dist.all_reduce(self._slots[batch], op=dist.ReduceOp.SUM, group=self.device_group)
        self.device_slot_collectives.add_(1)
        finalize_slots(self._slots[batch], self._output, batch,
                       self.hidden, self.topk, self.rank, self.ep)
        if not capturing:
            self.additional_slot_collectives += 1
            self._eager_warmed_batches.add(batch)
        return self._outputs[batch]

    def device_slot_collectives_snapshot(self):
        """Read at an externally enforced idle coverage boundary only."""
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('device collective snapshot is forbidden during capture')
        self.torch.cuda.synchronize(self.device_slot_collectives.device)
        return int(self.device_slot_collectives.item())
