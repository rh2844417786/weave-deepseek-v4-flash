"""Offline build and graph-safe IPC runtime for the Hopper paper candidate.

The routed path uses the resident IPC kernel only. Host evidence reads are
restricted to the explicit idle snapshot. Numerical errors are record-only.
"""
from functools import lru_cache
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys

from . import hopper_candidate as vendor
from .ops import PeerRuntime, ProfileBank

IMPLEMENTATION = 'hopper_persistent_paper_v1'
SOURCE_DIR = Path('experimental/hopper_persistent')
INCLUDE_PATHS = (SOURCE_DIR, Path('csrc'), vendor._VENDOR / 'include')
MECHANISM_FIELDS = ('calls', 'dynamic_calls', 'computed_tiles', 'stolen_tiles',
    'dispatch_payloads', 'remote_pairs', 'remote_unique', 'return_compute_overlap',
    'dispatch_compute_overlap', 'c_sum', 'k_sum', 'steal_budget_sum', 'rows',
    'tasks', 'groups', 'profile_candidates')


def _root():
    return Path(__file__).resolve().parents[2]


def source_manifest():
    """Hash new CUDA sources, their actual old-header closure and pinned vendor."""
    root = _root()
    snapshot = vendor._vendor_snapshot(root)
    pending = sorted(p for p in (root / SOURCE_DIR).rglob('*')
                     if p.is_file() and p.suffix in ('.cu', '.h', '.cuh'))
    if not any(p.suffix == '.cu' for p in pending):
        raise RuntimeError('missing Hopper persistent CUDA sources; restore GitHub checkout')
    while pending:
        path = pending.pop()
        if path in snapshot:
            continue
        if path.resolve() != path.absolute():
            raise RuntimeError(f'source must not use symlink paths: {path}')
        content = vendor._read(path, 'persistent source/header')
        snapshot[path] = content
        for delimiter, name in vendor._INCLUDE.findall(content.decode('utf-8')):
            search = ([path.parent / name] if delimiter == '"' else [])
            search += [root / prefix / name for prefix in INCLUDE_PATHS]
            found = next((p.resolve() for p in search if p.is_file()), None)
            if found is not None:
                rel = found.relative_to(root)
                if not (rel.is_relative_to(SOURCE_DIR) or
                        rel.is_relative_to(Path('csrc')) or rel.is_relative_to(vendor._VENDOR)
                        or rel == Path('experimental/hopper_fp8/math_contract.h')):
                    raise RuntimeError(f'include escapes approved source closure: {name}')
                if found not in snapshot:
                    if rel.is_relative_to(vendor._VENDOR):
                        raise RuntimeError(f'undeclared vendored include: {name}')
                    pending.append(found)
            elif delimiter == '"' or name.startswith(('cute/', 'cutlass/')):
                raise RuntimeError(f'missing source include: {name}; restore GitHub checkout')
    return {'schema_version': 1, 'implementation': IMPLEMENTATION,
        'numeric_contract': vendor.NUMERIC_CONTRACT,
        'files': [{'path': p.relative_to(root).as_posix(), 'sha256': vendor._digest(v)}
                  for p, v in sorted(snapshot.items())],
        'cuda_cflags': list(vendor.CUDA_CFLAGS), 'cxx_cflags': list(vendor.CXX_CFLAGS),
        'ldflags': list(vendor.LDFLAGS), 'cuda_arch_list': '9.0a',
        'include_paths': [p.as_posix() for p in INCLUDE_PATHS]}


def source_sha256():
    return vendor._digest(vendor._manifest_bytes(source_manifest()))


@lru_cache(maxsize=1)
def extension():
    manifest = source_manifest()
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise RuntimeError('use the existing Torch/CUDA 13 image; no dependencies are downloaded') from exc
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError('Hopper persistent requires an allocated SM90 GPU')
    if not torch.version.cuda or torch.version.cuda.split('.')[0] != '13':
        raise RuntimeError('Hopper persistent requires the existing CUDA 13 Torch image')
    runtime = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
               'python': f'{sys.version_info.major}.{sys.version_info.minor}'}
    digest = vendor._digest(vendor._manifest_bytes(manifest))
    name = 'weave_hopper_persistent_' + vendor._digest(vendor._manifest_bytes(
        {'source': manifest, 'runtime': runtime}))[:20]
    root = _root()
    sources = [str(root / r['path']) for r in manifest['files']
               if r['path'].startswith(SOURCE_DIR.as_posix() + '/') and r['path'].endswith('.cu')]
    module = load(name=name, sources=sources, extra_cuda_cflags=manifest['cuda_cflags'],
        extra_cflags=manifest['cxx_cflags'], extra_ldflags=manifest['ldflags'],
        extra_include_paths=[str(root / p) for p in manifest['include_paths']], verbose=True)
    if source_manifest() != manifest:
        raise RuntimeError('persistent source changed during build; discard and retry')
    module.weave_source_sha256 = digest
    module.weave_source_manifest = manifest
    return module


def validate_profile_bank(data, *, source_sha256, device_key, hidden_size,
                          intermediate_size, experts, top_k, rank, workers,
                          capacity, image_id, checkpoint_content_sha256, runtime):
    expected = dict(schema_version=1, status='measured', implementation=IMPLEMENTATION,
        numerical_policy='record_only', source_sha256=source_sha256, device_key=device_key,
        hidden_size=hidden_size, intermediate_size=intermediate_size, experts=experts,
        top_k=top_k, rank=rank, image_id=image_id,
        checkpoint_content_sha256=checkpoint_content_sha256, runtime=runtime,
        model_constants={'efficiency': 1.0, 'overlap': 0.0})
    if not image_id or not checkpoint_content_sha256:
        raise ValueError('paper profiles require resolved image and checkpoint content identities')
    if not isinstance(data, dict) or any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('paper measured profile identity/contract differs from this runtime')
    buckets = data.get('profiles')
    if not isinstance(buckets, list) or not buckets:
        raise ValueError('paper measured profile buckets required')
    previous = 0
    for bucket in buckets:
        if not isinstance(bucket, dict):
            raise ValueError('invalid paper profile bucket')
        limit, rows = bucket.get('max_local_tokens'), bucket.get('candidates')
        if type(limit) is not int or not previous < limit <= capacity:
            raise ValueError('profile buckets must increase and fit capacity')
        previous = limit
        if not isinstance(rows, list) or not rows:
            raise ValueError('measured candidate rows required')
        pairs = set()
        for row in rows:
            if (not isinstance(row, list) or len(row) != 6 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in row)):
                raise ValueError('candidate requires six finite numeric fields')
            c, k, compute, comm, efficiency, overlap = row
            if (int(c) != c or not 1 <= c < workers or int(k) != k or
                    not 1 <= k <= min(8, limit) or compute <= 0 or comm <= 0 or
                    efficiency != 1.0 or overlap != 0.0 or (c, k) in pairs):
                raise ValueError('invalid measured c/K/rates or conservative model constants')
            pairs.add((c, k))
        if len({c for c, k in pairs}) < 2 or len({k for c, k in pairs}) < 2:
            raise ValueError('each paper bucket needs measured alternatives for both c and K')
    if previous != capacity:
        raise ValueError('paper profile must cover full workspace capacity')
    return buckets


class HopperPersistentRuntime(PeerRuntime):
    implementation = IMPLEMENTATION
    method_identity = IMPLEMENTATION

    def __init__(self, group, capacity, hidden_size, intermediate_size, top_k, experts,
                 *, kernel='tiled', cache_views=True):
        if kernel != 'tiled' or not cache_views or hidden_size % 128 or intermediate_size % 128:
            raise ValueError('paper runtime requires tiled hot-cache and H/I multiples of 128')
        self.experts = experts
        super().__init__(group, capacity, hidden_size, intermediate_size, top_k, experts,
                         kernel=kernel, cache_views=cache_views, extension_factory=extension)
        if self.ep != 4:
            raise ValueError('paper runtime requires EP4')
        resources = self.ws.resources
        self.resources = dict(resources() if callable(resources) else resources)
        self.profile_bank_sha256 = None

    def profile_from_file(self, path):
        path = Path(path)
        if path.is_dir():
            path = path / f'paper-profile-bank-rank{self.rank}.json'
        content = path.read_bytes()
        prop, torch = self.torch.cuda.get_device_properties(self.device), self.torch
        data = json.loads(content)
        buckets = validate_profile_bank(data, source_sha256=self.source_sha256,
            device_key=f'{prop.name};sm{prop.major}{prop.minor};{prop.multi_processor_count}',
            hidden_size=self.h, intermediate_size=self.i, experts=self.experts,
            top_k=self.topk, rank=self.rank, workers=prop.multi_processor_count,
            capacity=self.capacity, image_id=os.environ.get('WEAVE_IMAGE_ID'),
            checkpoint_content_sha256=os.environ.get('WEAVE_CHECKPOINT_CONTENT_SHA256'),
            runtime={'python': platform.python_version(), 'torch': str(torch.__version__),
                     'cuda': torch.version.cuda})
        bank = ProfileBank([(b['max_local_tokens'], torch.tensor(b['candidates'],
            dtype=torch.float64, device=f'cuda:{self.device}')) for b in buckets])
        bank.sha256 = hashlib.sha256(content).hexdigest()
        self.profile_bank_sha256 = bank.sha256
        return bank

    def enable_execution_profile(self, config):
        raise ValueError('legacy BF16 tracing is unsupported; use paper mechanism evidence')

    def run(self, x, ids, weights, raw_weights, *, valid_count=None, profile=None,
            fixed_plan=(1, 1, 0), timeout_seconds=60., clamp=10.,
            calibration_mode=None, trace_layer=None):
        torch, n = self.torch, x.shape[0]
        modes = {None: 0, 'communication': 1, 'compute': 2}
        if calibration_mode not in modes or (calibration_mode is not None and
                (profile is not None or fixed_plan is None)):
            raise ValueError('calibration requires explicit fixed c/K and no dynamic profile')
        if x.ndim != 2 or x.shape[1] != self.h or n > self.capacity or x.dtype != torch.bfloat16:
            raise ValueError('input must fit the BF16 workspace shape')
        if ids.shape != (n, self.topk) or weights.shape != (n, self.topk):
            raise ValueError('routing shape differs from workspace')
        if any(t.device != x.device for t in (ids, weights)) or x.device.index != self.device:
            raise ValueError('inputs must use the workspace device')
        if valid_count is not None:
            raise ValueError('paper runtime uses shape-defined origin rows; valid_count unsupported')
        if trace_layer is not None and (type(trace_layer) is not int or not 0 <= trace_layer < 43):
            raise ValueError('paper evidence layer must be in 0..42')
        wx, wi, wr, wn, output = self._views
        wx[:n].copy_(x)
        wi[:n].copy_(ids.to(torch.int32))
        wr[:n].copy_(weights.to(torch.float32))
        wn.fill_(n)
        if profile is None:
            if fixed_plan is None:
                raise ValueError('provide a measured paper profile or explicit calibration/ablation plan')
            c, k, steal = fixed_plan
            selected = self.empty_profile
        else:
            selected = profile.for_tokens(n) if isinstance(profile, ProfileBank) else profile
            c, k, steal = -1, 1, 0
        phase = ('prefill', 'decode').index(self.phase) if self.phase in ('prefill', 'decode') else -1
        layer = trace_layer if trace_layer is not None and phase >= 0 else -1
        raw = isinstance(raw_weights, (list, tuple))
        launch = self.ws.run if raw else self.ws.run_bound
        launch(list(raw_weights) if raw else raw_weights, selected, c, k, steal,
               timeout_seconds, clamp, 2, modes[calibration_mode], layer, phase)
        return output[:n]

    def mechanism_snapshot(self):
        """Read only while service is idle; reject evidence changed during copy."""
        if self.torch.cuda.is_current_stream_capturing():
            raise ValueError('mechanism evidence is idle-only')
        self.torch.cuda.synchronize(self.device)
        before = self.ws.mechanism_evidence.cpu().tolist()
        last = self.ws.mechanism_last.cpu().tolist()
        smids = self.ws.worker_smids.cpu().tolist()
        after = self.ws.mechanism_evidence.cpu().tolist()
        if before != after:
            raise ValueError('service is active during mechanism evidence snapshot')
        phases = {}
        for phase_index, phase in enumerate(('prefill', 'decode')):
            phases[phase] = {}
            for layer in range(43):
                row = before[phase_index * 43 + layer]
                phases[phase][str(layer)] = {
                    'counters': dict(zip(MECHANISM_FIELDS, row[:16])),
                    'plan_histogram': [{'c': index // 8, 'k': index % 8 + 1, 'calls': count}
                                       for index, count in enumerate(row[16:]) if count]}
        return {'fields': list(MECHANISM_FIELDS), 'worker_smids': smids,
                'mechanism_last': last, 'mechanism_evidence': before, 'phases': phases}
