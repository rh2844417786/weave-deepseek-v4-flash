"""Independent BF16 peer-tail transport; no Marlin or historical compute dependency.

IPC setup is collective; standalone teardown uses a host barrier and service
teardown uses the controller's HTTP prepare/commit barrier. The steady path is one cooperative
CUDA kernel. Its same device helpers are included by the unified tail executor.
Run/snapshot must never overlap: one ordered stream owns this workspace, including
all graphs that capture it. An aborted workspace is terminal and must be closed.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys

METHOD = 'native_marlin_peer_tail_v1'
SOURCES = ('experimental/native_tail/task_protocol.cuh',
           'experimental/native_tail/peer_reduce.cuh',
           'experimental/native_tail/transport.cu')


@dataclass(frozen=True)
class TransportShape:
    tokens: int
    hidden: int
    chunks: int = 1

    def __post_init__(self):
        if (type(self.tokens) is not int or not 0 <= self.tokens <= 8192 or
            type(self.hidden) is not int or self.hidden <= 0 or
            type(self.chunks) is not int or self.chunks not in (1, 2, 4) or self.hidden % (4*self.chunks)):
            raise ValueError('BF16 transport needs 0<=tokens<=8192 and hidden divisible by 4*chunks, chunks in (1,2,4)')

    @property
    def channels_per_owner(self):
        return self.hidden // (4*self.chunks)

    @property
    def payload_elements(self):
        return self.tokens*self.hidden

    @property
    def task_count(self):
        return self.chunks*5

    def _check(self, chunk, owner, token, channel):
        if not (0 <= chunk < self.chunks and 0 <= owner < 4 and
                0 <= token < self.tokens and 0 <= channel < self.channels_per_owner):
            raise IndexError('chunk, owner, token or channel outside transport shape')

    def payload_index(self, chunk, owner, token, channel):
        self._check(chunk, owner, token, channel)
        return ((chunk*4+owner)*self.tokens+token)*self.channels_per_owner+channel

    def output_index(self, chunk, owner, token, channel):
        self._check(chunk, owner, token, channel)
        return token*self.hidden+(chunk*4+owner)*self.channels_per_owner+channel


def assess_transport_evidence(evidence):
    """Hard gates apply to protocol/coverage/memory; finite differences are recorded."""
    failures = []
    if evidence.get('abort_code') != 0:
        failures.append('protocol_abort')
    if not isinstance(evidence.get('sequence'), int) or evidence['sequence'] < 1:
        failures.append('missing_invocation')
    tasks = evidence.get('tasks', [])
    if not tasks or len(tasks) != evidence.get('expected_tasks') or any(n != 1 for n in tasks):
        failures.append('task_set_not_exactly_once')
    if evidence.get('nonfinite') != 0:
        failures.append('nonfinite_output')
    if evidence.get('acknowledged') is not True:
        failures.append('missing_consumer_ack')
    if evidence.get('canary_ok') is not True:
        failures.append('workspace_canary')
    return {'valid': not failures, 'failures': failures, 'numerical_policy': 'record_only'}


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    return {'schema_version': 1, 'method': METHOD, 'component': 'peer_transport',
            'files': [{'path': name, 'sha256': hashlib.sha256((root/name).read_bytes()).hexdigest()}
                      for name in SOURCES],
            'cuda_cflags': ['-O3', '-lineinfo', '-std=c++17', '-Xptxas=-v'],
            'cxx_cflags': ['-O3', '-std=c++17'],
            'cuda_arch_list': os.environ.get('TORCH_CUDA_ARCH_LIST', '9.0a')}


@lru_cache(maxsize=1)
def extension():
    import torch
    from torch.utils.cpp_extension import load
    if not torch.cuda.is_available():
        raise RuntimeError('native tail transport requires server CUDA; use the existing SGLang image, no download fallback')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0a')
    root = Path(__file__).resolve().parents[2]
    manifest = source_manifest()
    source_bytes = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    identity = f'{torch.__version__};{torch.version.cuda};{sys.version_info[:2]}'.encode()
    name = 'weave_native_tail_transport_' + hashlib.sha256(source_bytes+identity).hexdigest()[:20]
    module = load(name=name, sources=[str(root/'experimental/native_tail/transport.cu')],
                  extra_cuda_cflags=manifest['cuda_cflags'], extra_cflags=manifest['cxx_cflags'], verbose=True)
    if source_manifest() != manifest:
        raise RuntimeError('native tail source changed during compilation')
    module.source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    module.source_manifest = manifest
    return module


class NativeTailTransport:
    """Four processes, one GPU each, a Gloo setup group, stable IPC addresses.

    The caller selects devices from GPUQ's visible allocation before construction;
    physical device ids are never hard coded. close() is collective and two phase:
    all peer mappings close before any rank frees its exported allocation.
    """
    def __init__(self, group, capacity, hidden, *, session=None):
        import torch
        import torch.distributed as dist
        TransportShape(capacity, hidden)
        self.torch, self.dist, self.group = torch, dist, group
        self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)
        if self.world_size != 4 or dist.get_backend(group) != 'gloo':
            raise ValueError('native tail transport requires exactly four ranks in a Gloo setup group')
        self.capacity, self.hidden, self.ws = capacity, hidden, None
        self.closed = False
        self.peers_disconnected = False
        signatures = [None]*4
        dist.all_gather_object(signatures, (capacity, hidden), group=group)
        if any(value != (capacity, hidden) for value in signatures):
            raise ValueError('ranks disagree on workspace capacity or hidden size')
        sessions = [session if session is not None else (secrets.randbits(62)+1 if self.rank == 0 else None)]
        # Gloo rank may be a subgroup rank, so broadcast_object_list's src must be global.
        leader = dist.get_global_rank(group, 0) if group is not None else 0
        dist.broadcast_object_list(sessions, src=leader, group=group)
        self.session = int(sessions[0])
        try:
            self.module = self._stage('compile', extension)
            self.source_sha256 = self.module.source_sha256
            self.ws = self._stage('allocate', lambda: self.module.Workspace(self.rank, capacity, hidden, self.session))
            handles = [None]*4
            local = self._stage('IPC metadata', lambda: (self.source_sha256, bytes(self.ws.handle()), self.ws.device, self.ws.uuid))
            dist.all_gather_object(handles, local, group=group)
            if len({h[0] for h in handles}) != 1 or len({h[3] for h in handles}) != 4:
                raise RuntimeError('four distinct GPUs and identical source identity are required')
            self._stage('connect', lambda: self.ws.connect([h[1] for h in handles],
                        [h[2] for h in handles], [h[3] for h in handles]))
            self.device = self.ws.device
            self.output = self._stage('output allocation', lambda: torch.empty((capacity, hidden),
                                   dtype=torch.bfloat16, device=f'cuda:{self.device}'))
        except BaseException:
            # Setup stages agree on failure before this collective cleanup.
            self.close()
            raise

    def _stage(self, label, operation):
        result, error = None, None
        try:
            result = operation()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        errors = [None]*4
        self.dist.all_gather_object(errors, error, group=self.group)
        if any(errors):
            # A successful allocation must stay reachable when a peer fails allocation.
            if label == 'allocate' and result is not None:
                self.ws = result
            raise RuntimeError(f'native tail {label} failed on rank(s): {errors}')
        return result

    def native_config(self):
        """Host metadata for another extension including task_protocol.cuh.

        Executor must independently query occupancy for its FINAL kernel. It may
        use the same addresses/Layout and set cfg.chunks from a GPU agreed plan.
        """
        return {'addresses': list(self.ws.addresses()), 'rank': self.rank,
                'capacity': self.capacity, 'hidden': self.hidden, 'session': self.session,
                'device': self.device}

    def run(self, partial, *, chunks=1, layer=0, output=None, timeout_seconds=20.,
            delay_seconds=0., cancel_seconds=0.):
        TransportShape(partial.shape[0], self.hidden, chunks)
        if self.closed:
            raise RuntimeError('native tail transport is closed')
        if output is None:
            output = self.output[:partial.shape[0]]
        self.ws.run(partial, output, chunks, layer, timeout_seconds, delay_seconds, cancel_seconds)
        return output

    def snapshot(self, chunks=1):
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('evidence snapshot is allowed only outside CUDA Graph capture/replay')
        evidence = dict(self.ws.snapshot(chunks))
        evidence.update(assess_transport_evidence(evidence))
        return evidence

    def disconnect_peers(self):
        """Local prepare: exported allocation stays alive until every peer agrees."""
        if self.closed or self.peers_disconnected:
            return
        if self.ws is not None:
            self.ws.disconnect()
        self.peers_disconnected = True

    def release_local(self):
        """Local commit; caller must first confirm ALL ranks disconnected."""
        if self.closed:
            return
        if not self.peers_disconnected:
            raise RuntimeError('disconnect peers before releasing local IPC allocation')
        if self.ws is not None:
            self.ws.close()
        self.closed = True

    def close(self):
        """Standalone harness collective; never call from a serving HTTP thread."""
        if self.closed:
            return
        # Even a terminal abort must finish every kernel before mappings are unmapped.
        error = None
        try:
            self.disconnect_peers()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        errors = [None]*4
        self.dist.all_gather_object(errors, error, group=self.group)
        if any(errors):
            raise RuntimeError(f'cannot safely free IPC allocations after failed disconnect: {errors}')
        self.release_local()
