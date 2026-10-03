"""Lazy CUDA build and peer workspace ownership for functional Weave execution."""
from functools import lru_cache
import hashlib
import json
import os
import platform
from pathlib import Path
import sys


def _kernel_source_bytes():
    root = Path(__file__).resolve().parents[2]
    return b''.join((root/'csrc'/name).read_bytes() for name in
                    ('megakernel.cu', 'scheduler.h', 'expert_numerics.h', 'tile_scheduler.h'))


def kernel_source_sha256():
    return hashlib.sha256(_kernel_source_bytes()).hexdigest()


def extension_name(torch_version, cuda_version):
    identity = f'{torch_version};{cuda_version};py{sys.version_info.major}.{sys.version_info.minor};{os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0a")}'
    content = _kernel_source_bytes()
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


class ProfileBank:
    """All measured tensors are allocated before capture; selection uses shape metadata."""
    def __init__(self, buckets):
        self.buckets = tuple(buckets)
        if not self.buckets or any(type(n) is not int or n<1 for n,_ in self.buckets):
            raise ValueError('positive profile buckets required')
        if [n for n,_ in self.buckets] != sorted({n for n,_ in self.buckets}):
            raise ValueError('profile buckets must be sorted and unique')

    def for_tokens(self, n):
        if n<0:
            raise ValueError('negative profile workload')
        for limit,tensor in self.buckets:
            if n<=limit:
                return tensor
        raise ValueError('no measured profile for this token count')


class PeerRuntime:
    """One process/device per EP rank; calls across layers are strictly ordered.

    Initialization exchanges IPC handles on the CPU group. Forward execution
    uses peer memory and the persistent kernel; no CPU routing statistics or
    NCCL dispatch are hidden outside the kernel.
    """
    def __init__(self, group, capacity, hidden_size, intermediate_size, top_k, experts, *, kernel='scalar'):
        import torch
        import torch.distributed as dist
        self.torch, self.group = torch, group
        self.rank, self.ep = dist.get_rank(group), dist.get_world_size(group)
        if self.ep > 4 or dist.get_backend(group) != 'gloo':
            raise ValueError('IPC setup requires a Gloo CPU group of at most four ranks')

        def local_stage(label, operation):
            # Each stage contains only local work. All peers agree on failure
            # before any rank enters the next handle/signature collective.
            result, error = None, None
            try:
                result = operation()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            errors = [None]*self.ep
            dist.all_gather_object(errors, error, group=group)
            if any(errors):
                raise RuntimeError(f'{label} failed on EP rank(s): {errors}')
            return result

        def metadata():
            if kernel not in ('scalar', 'wmma', 'tiled'):
                raise ValueError('kernel must be scalar, wmma or tiled')
            device = torch.cuda.current_device()
            if device != self.rank:
                raise ValueError('CUDA ordinal must equal EP rank in the selected four-device group')
            return device
        self.device = local_stage('IPC metadata', metadata)
        self.kernel = kernel
        self.capacity, self.h, self.i, self.topk = capacity, hidden_size, intermediate_size, top_k
        signature = (capacity, hidden_size, intermediate_size, top_k, experts)
        signatures = [None]*self.ep
        dist.all_gather_object(signatures, (signature,kernel), group=group)
        if any(value != (signature,kernel) for value in signatures):
            raise ValueError('ranks disagree on workspace dimensions')

        def allocate():
            module = extension()
            workspace = module.Workspace(self.rank,self.ep,*signature)
            return module,workspace,bytes(workspace.handle())
        module,self.ws,handle = local_stage('IPC allocation', allocate)
        self.source_sha256 = module.weave_source_sha256
        handles = [None]*self.ep
        dist.all_gather_object(handles,(self.source_sha256,handle),group=group)
        if any(value[0] != self.source_sha256 for value in handles):
            raise ValueError('EP ranks loaded different compiled CUDA source')
        local_stage('IPC peer connection', lambda:self.ws.connect([value[1] for value in handles]))
        self.empty_profile = local_stage('IPC profile allocation', lambda:torch.empty(
            (0,6),dtype=torch.float64,device=f'cuda:{self.device}'))
        self.phase = 'other'

    def profile_from_file(self, path):
        """Only consume measured profiles tied to this kernel and device shape."""
        torch = self.torch
        path = Path(path)
        if path.is_dir(): path = path/f'profile-bank-rank{self.rank}.json'
        data = json.loads(path.read_text())
        prop = torch.cuda.get_device_properties(self.device)
        key = f'{prop.name};sm{prop.major}{prop.minor};{prop.multi_processor_count}'
        if self.kernel == 'tiled':
            from .calibration import load_verified_profile_bundle
            expected = os.environ.get('WEAVE_CHECKPOINT_CONTENT_SHA256')
            if not expected or data.get('checkpoint_content_sha256') != expected:
                raise ValueError('tiled profile must match verified checkpoint content identity')
            if (data.get('image_id') != os.environ.get('WEAVE_IMAGE_ID') or
                    any(data.get('runtime',{}).get(k)!=v for k,v in {'python': platform.python_version(),
                    'torch': str(torch.__version__), 'cuda': torch.version.cuda}.items())):
                raise ValueError('tiled profile runtime/image identity differs from current process')
            rows = load_verified_profile_bundle(path,kernel='tiled_mxfp4_tile_v1',
                source_sha256=self.source_sha256,device_key=key,hidden_size=self.h,
                intermediate_size=self.i,rank=self.rank,workers=prop.multi_processor_count)
            bank=ProfileBank([(row['max_local_tokens'],torch.tensor(row['candidates'],
                dtype=torch.float64,device=f'cuda:{self.device}')) for row in rows])
            bank.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            return bank
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
            fixed_plan=(1, 1, 0), timeout_seconds=60., clamp=10., calibration_mode=None):
        torch = self.torch
        n = x.shape[0]
        modes={None: 0, 'communication': 1, 'compute': 2}
        if calibration_mode not in modes or (calibration_mode is not None and (self.kernel != 'tiled' or profile is not None or fixed_plan is None)):
            raise ValueError('calibration requires tiled execution with a fixed plan and no profile')
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
            if isinstance(profile, ProfileBank): profile = profile.for_tokens(n)
            c, k, steal = -1, 1, 0
        self.ws.run(list(raw_weights), profile, c, k, steal, timeout_seconds, clamp,
                    {'scalar': 0, 'wmma': 1, 'tiled': 2}[self.kernel], modes[calibration_mode])
        return self.ws.output[:n]
