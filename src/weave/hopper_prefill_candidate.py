"""Explicit, offline-only build of the experimental Hopper FP8 expert path.

The production BF16 loader is independent of this module. Importing this module
needs only the standard library; verification precedes importing Torch or
starting the compiler, and never downloads a missing dependency.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


CUTLASS_COMMIT = 'f3fde58372d33e9a5650ba7b80fc48b3b49d40c8'
NUMERIC_CONTRACT = 'hopper_fp8_e4m3_rne_satfinite_block128_mxfp4_k32_bf16_v1'
CUDA_CFLAGS = ('-O3', '-lineinfo', '-std=c++17', '-Xptxas=-v',
               '-gencode=arch=compute_90a,code=sm_90a')
CXX_CFLAGS = ('-O3', '-std=c++17')
LDFLAGS = ('-lcuda',)
_INCLUDE = re.compile(r'^\s*#\s*include\s*(["<])([^">]+)[">]', re.MULTILINE)
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_VENDOR = Path('third_party/cutlass_hopper')
_CANDIDATE = Path('experimental/hopper_prefill')
_INCLUDE_PATHS = (_VENDOR / 'include', Path('csrc'), _CANDIDATE)


def _root():
    return Path(__file__).resolve().parents[2]


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _manifest_bytes(manifest):
    return json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')


def _safe_relative(value):
    return (isinstance(value, str) and bool(value) and '\\' not in value
            and not PurePosixPath(value).is_absolute()
            and all(part not in ('', '.', '..') for part in value.split('/')))


def _read(path, description):
    try:
        if path.is_symlink():
            raise RuntimeError(f'{description} must not be a symlink: {path}')
        return path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f'missing/unreadable {description}: {path}; '
                           'restore the pinned repository files from GitHub; no download is attempted') from exc


from .hopper_candidate import _vendor_snapshot


def _verify_include_closure(root, snapshot):
    """Reject local/quoted dependencies that would escape the hashed inputs.

    System and Torch angle includes are supplied by the existing image. CUTE
    and CUTLASS angle includes are always part of the pinned vendored closure.
    """
    included_paths = {path.resolve() for path in snapshot}
    for source, content in snapshot.items():
        if source.name == 'manifest.json':
            continue
        for delimiter, name in _INCLUDE.findall(content.decode('utf-8')):
            candidates = ([source.parent / name] if delimiter == '"' else [])
            candidates += [root / prefix / name for prefix in _INCLUDE_PATHS]
            existing = next((path.resolve() for path in candidates if path.is_file()), None)
            if existing is not None:
                if existing not in included_paths:
                    raise RuntimeError(f'include escapes hashed source closure: {source.relative_to(root)} -> {name}')
            elif delimiter == '"' or name.startswith(('cute/', 'cutlass/')):
                raise RuntimeError(f'missing include in source closure: {source.relative_to(root)} -> {name}')


def source_manifest():
    """Return the verified candidate closure and fixed build/numerical contract."""
    root = _root()
    snapshot = _vendor_snapshot(root)
    paths = sorted(path for path in (root / _CANDIDATE).rglob('*')
                   if path.is_file() and path.suffix in ('.cu', '.h', '.cuh'))
    if not any(path.suffix == '.cu' for path in paths):
        raise RuntimeError('no candidate CUDA translation units under experimental/hopper_prefill')
    paths.append(root / 'csrc/expert_numerics.h')
    for path in paths:
        if path.resolve() != path.absolute():
            raise RuntimeError(f'candidate source must not use symlink paths: {path}')
        snapshot[path] = _read(path, 'candidate CUDA source/header')
    _verify_include_closure(root, snapshot)
    return {
        'schema_version': 1, 'numerical_policy': 'record_only',
        'numeric_contract': NUMERIC_CONTRACT,
        'files': [{'path': path.relative_to(root).as_posix(), 'sha256': _digest(content)}
                  for path, content in sorted(snapshot.items())],
        'cuda_cflags': list(CUDA_CFLAGS), 'cxx_cflags': list(CXX_CFLAGS),
        'ldflags': list(LDFLAGS), 'cuda_arch_list': '9.0a',
        'include_paths': [path.as_posix() for path in _INCLUDE_PATHS],
    }


def source_sha256():
    return _digest(_manifest_bytes(source_manifest()))


def _extension_name(torch_version, cuda_version, manifest):
    runtime = {'torch': torch_version, 'cuda': cuda_version,
               'python': f'{sys.version_info.major}.{sys.version_info.minor}'}
    return 'weave_hopper_prefill_' + _digest(_manifest_bytes({'runtime': runtime, 'source': manifest}))[:20]


@lru_cache(maxsize=1)
def extension():
    """Build explicitly on the server's existing CUDA 13/Hopper Torch image."""
    manifest = source_manifest()
    source_digest = _digest(_manifest_bytes(manifest))
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise RuntimeError('candidate requires the existing Torch/CUDA 13 server image; '
                           'do not install packages or download dependencies in the server job') from exc
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError('candidate requires an allocated Hopper SM90 GPU in the existing server image')
    if not torch.version.cuda or torch.version.cuda.split('.')[0] != '13':
        raise RuntimeError('candidate requires the existing CUDA 13 Torch server image')
    root = _root()
    sources = [str(root / row['path']) for row in manifest['files']
               if row['path'].startswith(_CANDIDATE.as_posix() + '/') and row['path'].endswith('.cu')]
    module = load(name=_extension_name(str(torch.__version__), torch.version.cuda, manifest), sources=sources,
                  extra_cuda_cflags=list(manifest['cuda_cflags']),
                  extra_cflags=list(manifest['cxx_cflags']),
                  extra_ldflags=list(manifest['ldflags']),
                  extra_include_paths=[str(root / path) for path in manifest['include_paths']],
                  verbose=True)
    try:
        current_manifest = source_manifest()
    except RuntimeError as exc:
        raise RuntimeError('candidate source changed during extension build; discard this build and retry') from exc
    if current_manifest != manifest:
        raise RuntimeError('candidate source changed during extension build; discard this build and retry')
    module.weave_source_sha256 = source_digest
    module.weave_source_manifest = manifest
    return module


def tile_capacity(max_batch, topk, local_experts):
    if not (1 <= max_batch <= 8192 and 1 <= topk <= 16 and 1 <= local_experts <= 4096):
        raise ValueError('prefill requires B1..8192, topk1..16, local experts1..4096')
    return (max_batch * topk + 63) // 64 + local_experts


def workspace_bytes(max_batch, h, i, experts, ep, topk):
    if not (1 <= ep <= 4 and topk <= experts <= 4096 and experts % ep == 0
            and 128 <= h <= 8192 and h % 128 == 0 and 128 <= i <= 8192 and i % 128 == 0):
        raise ValueError('invalid prefill dimensions')
    local=experts//ep
    tiles=tile_capacity(max_batch, topk, local)
    return (4*(local+2*tiles+1+local*max_batch) + max_batch*topk*h*4
            + max_batch*h*4 + max_batch*topk*4
            + tiles*64*(3*h+7*i) + tiles*64*(h//128+i//128)*4)
