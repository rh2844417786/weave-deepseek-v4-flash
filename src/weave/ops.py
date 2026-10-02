"""Lazy CUDA build and peer workspace ownership for functional Weave execution."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys


def kernel_source_sha256():
    root = Path(__file__).resolve().parents[2]
    return hashlib.sha256((root/'csrc/megakernel.cu').read_bytes() +
                          (root/'csrc/scheduler.h').read_bytes()).hexdigest()


def extension_name(torch_version, cuda_version):
    root = Path(__file__).resolve().parents[2]
    identity = f'{torch_version};{cuda_version};py{sys.version_info.major}.{sys.version_info.minor};{os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0a")}'
    content = (root/'csrc/megakernel.cu').read_bytes() + (root/'csrc/scheduler.h').read_bytes()
    return 'weave_functional_' + hashlib.sha256(content + identity.encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def extension():
    import torch
    from torch.utils.cpp_extension import load
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/Hopper required; there is no native-MoE fallback')
    root = Path(__file__).resolve().parents[2]
    source = root / 'csrc/megakernel.cu'
    header = root / 'csrc/scheduler.h'
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0a')
    source_digest = kernel_source_sha256()
    module = load(name=extension_name(str(torch.__version__), torch.version.cuda), sources=[str(source)],
                extra_cuda_cflags=['-O3', '-lineinfo', '-std=c++17'],
                extra_cflags=['-O3', '-std=c++17'], verbose=True)
    if kernel_source_sha256() != source_digest:
        raise RuntimeError('CUDA source changed during extension build')
    module.weave_source_sha256 = source_digest
    return module


class PeerRuntime:
    """One process/device per EP rank; calls across layers are strictly ordered.

    Initialization exchanges IPC handles on the CPU group. Forward execution
    uses peer memory and the persistent kernel; no CPU routing statistics or
    NCCL dispatch are hidden outside the kernel.
    """
    def __init__(self, group, capacity, hidden_size, intermediate_size, top_k, experts, *, kernel='scalar'):
        import torch
        import torch.distributed as dist
        self.torch = torch
        if kernel not in ('scalar', 'wmma'):
            raise ValueError('kernel must be scalar or wmma')
        self.kernel = kernel
        self.group = group
        self.rank, self.ep = dist.get_rank(group), dist.get_world_size(group)
        if self.ep > 4 or dist.get_backend(group) != 'gloo':
            raise ValueError('IPC setup requires a Gloo CPU group of at most four ranks')
        self.device = torch.cuda.current_device()
        if self.device != self.rank:
            raise ValueError('CUDA ordinal must equal EP rank in the selected four-device group')
        self.capacity, self.h, self.i, self.topk = capacity, hidden_size, intermediate_size, top_k
        signature = (capacity, hidden_size, intermediate_size, top_k, experts)
        signatures = [None] * self.ep
        dist.all_gather_object(signatures, (signature, kernel), group=group)
        if any(s != (signature, kernel) for s in signatures):
            raise ValueError('ranks disagree on workspace dimensions')
        module = extension()
        self.source_sha256 = module.weave_source_sha256
        self.ws = module.Workspace(self.rank, self.ep, *signature)
        handles = [None] * self.ep
        dist.all_gather_object(handles, bytes(self.ws.handle()), group=group)
        self.ws.connect(handles)
        self.empty_profile = torch.empty((0, 6), dtype=torch.float64, device=f'cuda:{self.device}')
        self.phase = 'other'

    def profile_from_file(self, path):
        """Only consume measured profiles tied to this kernel and device shape."""
        torch = self.torch
        data = json.loads(Path(path).read_text())
        prop = torch.cuda.get_device_properties(self.device)
        key = f'{prop.name};sm{prop.major}{prop.minor};{prop.multi_processor_count}'
        if (data.get('status') != 'measured' or data.get('kernel') != f'{self.kernel}_mxfp4_token_v1'
                or data.get('source_sha256') != self.source_sha256
                or data.get('device_key') != key or data.get('hidden_size') != self.h
                or data.get('intermediate_size') != self.i):
            raise ValueError('profile is not a measured match for this kernel/device/shape')
        from .cost_model import Candidate, Workload, select_plan
        rows = data['candidates']
        if not rows or any(len(row) != 6 for row in rows):
            raise ValueError('profile candidates must have six fields')
        for row in rows:
            candidate = Candidate(*row)
            # Exercise validation with enough unique tokens for all allowed K.
            remote = 0 if candidate.comm_workers == 0 else 1
            select_plan(Workload(8-remote, remote, remote, 8, self.h, self.i),
                        prop.multi_processor_count, [candidate])
        return torch.tensor(rows, dtype=torch.float64, device=f'cuda:{self.device}')

    def run(self, x, ids, weights, raw_weights, *, valid_count=None, profile=None,
            fixed_plan=(1, 1, 0), timeout_seconds=60., clamp=10.):
        torch = self.torch
        n = x.shape[0]
        if x.ndim != 2 or x.shape[1] != self.h or n > self.capacity or x.dtype != torch.bfloat16:
            raise ValueError('input must fit the BF16 workspace shape')
        if ids.shape != (n, self.topk) or weights.shape != (n, self.topk):
            raise ValueError('native routing shape differs from the workspace')
        if any(t.device != x.device for t in (ids, weights)) or x.device.index != self.device:
            raise ValueError('all routing inputs must be on the workspace device')
        self.ws.x[:n].copy_(x)
        self.ws.ids[:n].copy_(ids.to(torch.int32))
        self.ws.routing[:n].copy_(weights.to(torch.float32))
        if valid_count is None:
            self.ws.n.fill_(n)
        else:
            if valid_count.dtype != torch.int32 or valid_count.numel() != 1 or valid_count.device != x.device:
                raise ValueError('valid_count must be one device int32 value')
            self.ws.n.copy_(valid_count.reshape(1))
        if profile is None:
            if fixed_plan is None:
                raise ValueError('provide a measured profile or an explicit functional fixed plan')
            c, k, steal = fixed_plan
            profile = self.empty_profile
        else:
            c, k, steal = -1, 1, 0
        self.ws.run(list(raw_weights), profile, c, k, steal, timeout_seconds, clamp, int(self.kernel == 'wmma'))
        return self.ws.output[:n]
