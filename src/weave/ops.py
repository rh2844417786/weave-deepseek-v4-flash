"""Lazy CUDA build and peer workspace ownership for functional Weave execution."""
from functools import lru_cache
import hashlib
import json
import os
import platform
from pathlib import Path
import sys


CUDA_CFLAGS = ('-O3', '-lineinfo', '-std=c++17', '-Xptxas=-v')
CXX_CFLAGS = ('-O3', '-std=c++17')


def kernel_source_manifest():
    """Deterministic identity of every runtime source/header and build setting."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted(path for path in (root / 'csrc').rglob('*')
                   if path.is_file() and path.suffix in ('.cu', '.cuh', '.h'))
    if not paths:
        raise RuntimeError('no runtime CUDA sources found under csrc')
    return {'schema_version': 1,
            'files': [{'path': path.relative_to(root).as_posix(),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in paths],
            'cuda_cflags': list(CUDA_CFLAGS), 'cxx_cflags': list(CXX_CFLAGS),
            'cuda_arch_list': os.environ.get('TORCH_CUDA_ARCH_LIST', '9.0a')}


def _manifest_bytes(manifest):
    return json.dumps(manifest, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def _kernel_source_bytes():
    return _manifest_bytes(kernel_source_manifest())


def kernel_source_sha256():
    return hashlib.sha256(_kernel_source_bytes()).hexdigest()


def _extension_name_from_manifest(torch_version, cuda_version, manifest):
    identity = f'{torch_version};{cuda_version};py{sys.version_info.major}.{sys.version_info.minor};{manifest["cuda_arch_list"]}'
    content = _manifest_bytes(manifest)
    return 'weave_functional_' + hashlib.sha256(content + identity.encode()).hexdigest()[:16]


def extension_name(torch_version, cuda_version):
    return _extension_name_from_manifest(torch_version, cuda_version, kernel_source_manifest())


@lru_cache(maxsize=1)
def extension():
    import torch
    from torch.utils.cpp_extension import load
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/Hopper required; there is no native-MoE fallback')
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0a')
    manifest = kernel_source_manifest()
    source_digest = hashlib.sha256(_manifest_bytes(manifest)).hexdigest()
    sources = [str(root / row['path']) for row in manifest['files']
               if row['path'].endswith('.cu')]
    module = load(name=_extension_name_from_manifest(str(torch.__version__), torch.version.cuda, manifest), sources=sources,
                extra_cuda_cflags=list(manifest['cuda_cflags']),
                extra_cflags=list(manifest['cxx_cflags']), verbose=True)
    if kernel_source_manifest() != manifest:
        raise RuntimeError('CUDA source changed during extension build')
    module.weave_source_sha256 = source_digest
    module.weave_source_manifest = manifest
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
    def __init__(self, group, capacity, hidden_size, intermediate_size, top_k, experts, *, kernel='scalar', cache_views=True):
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
        self.cache_views = cache_views
        self._views = local_stage('IPC tensor views', lambda:tuple(
            getattr(self.ws, name) for name in ('x', 'ids', 'routing', 'n', 'output')) if cache_views else None)
        self.empty_profile = local_stage('IPC profile allocation', lambda:torch.empty(
            (0,6),dtype=torch.float64,device=f'cuda:{self.device}'))
        self.phase = 'other'
        self.trace_config=None
        self.trace_buffers=None

    def bind_weights(self, raw_weights):
        """Bind immutable raw checkpoint storage before graph capture."""
        return self.ws.bind_weights(list(raw_weights))

    def enable_execution_profile(self,config):
        """Allocate immutable buffer addresses before any graph capture."""
        if self.kernel!='tiled' or self.trace_config is not None or self.torch.cuda.is_current_stream_capturing():
            raise ValueError('configure tiled execution profiling before graph capture')
        self.trace_workers=self.torch.cuda.get_device_properties(self.device).multi_processor_count
        self.trace_buffers=tuple(self.torch.zeros(shape,dtype=self.torch.int64,
            device=f'cuda:{self.device}') for shape in config.shapes(self.trace_workers))
        self.ws.configure_trace(*self.trace_buffers,config.layers,config.stride,config.capacity)
        self.trace_config=config

    def reset_execution_profile(self):
        if self.trace_buffers is None: raise ValueError('execution profiling is disabled')
        self.torch.cuda.synchronize(self.device)
        for buffer in self.trace_buffers: buffer.zero_()
        self.torch.cuda.synchronize(self.device)
        if getattr(self,'forward_sampler',None) is not None:self.forward_sampler.reset()

    def execution_profile_snapshot(self,identity):
        """Idle only; detect forwards interleaved with the multi-buffer copy."""
        from .execution_profile import summarize_profile
        if self.trace_buffers is None: raise ValueError('execution profiling is disabled')
        heads,totals,records=self.trace_buffers
        error=None
        for _ in range(3):
            self.torch.cuda.synchronize(self.device)
            before=heads.cpu().tolist()
            values=(totals.cpu().tolist(),records.cpu().tolist())
            after=heads.cpu().tolist()
            if before!=after:
                error=ValueError('service is active during execution profile snapshot')
                continue
            try:
                result=summarize_profile(self.trace_config,identity,before,*values,workers=self.trace_workers)
                if getattr(self,'forward_sampler',None) is not None:
                    result['forward_profile']=self.forward_sampler.snapshot()
                return result
            except ValueError as exc: error=exc
        raise error

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
            fixed_plan=(1, 1, 0), timeout_seconds=60., clamp=10., calibration_mode=None,trace_layer=None):
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
        workspace_x, workspace_ids, workspace_routing, workspace_n, workspace_output = (
            self._views if self._views is not None else
            tuple(getattr(self.ws, name) for name in ('x', 'ids', 'routing', 'n', 'output')))
        workspace_x[:n].copy_(x)
        workspace_ids[:n].copy_(ids.to(torch.int32))
        workspace_routing[:n].copy_(weights.to(torch.float32))
        if valid_count is None:
            workspace_n.fill_(n)
        else:
            if valid_count.dtype != torch.int32 or valid_count.numel() != 1 or valid_count.device != x.device:
                raise ValueError('valid_count must be one device int32 value')
            workspace_n.copy_(valid_count.reshape(1))
        if profile is None:
            if fixed_plan is None:
                raise ValueError('provide a measured profile or an explicit functional fixed plan')
            c, k, steal = fixed_plan
            profile = self.empty_profile
        else:
            if isinstance(profile, ProfileBank): profile = profile.for_tokens(n)
            c, k, steal = -1, 1, 0
        traced=self.trace_config is not None and trace_layer is not None and self.phase in ('prefill','decode')
        raw = isinstance(raw_weights, (list, tuple))
        launch = self.ws.run if raw else self.ws.run_bound
        launch(list(raw_weights) if raw else raw_weights, profile, c, k, steal, timeout_seconds, clamp,
                    {'scalar': 0, 'wmma': 1, 'tiled': 2}[self.kernel], modes[calibration_mode],
                    trace_layer if traced else -1,('prefill','decode').index(self.phase) if traced else -1)
        return workspace_output[:n]
