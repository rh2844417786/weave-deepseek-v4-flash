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
_CANDIDATE = Path('experimental/hopper_fp8')
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


def _vendor_snapshot(root):
    manifest_path = root / _VENDOR / 'manifest.json'
    content = _read(manifest_path, 'vendor manifest')
    try:
        manifest = json.loads(content)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError('invalid vendored CUTLASS manifest JSON') from exc
    upstream = manifest.get('upstream', {}) if isinstance(manifest, dict) else {}
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
            or not isinstance(upstream, dict)
            or upstream.get('repository') != 'https://github.com/NVIDIA/cutlass'
            or upstream.get('commit') != CUTLASS_COMMIT
            or not isinstance(upstream.get('archive_sha256'), str)
            or not _SHA256.fullmatch(upstream['archive_sha256'])):
        raise RuntimeError('vendored CUTLASS manifest must identify the pinned upstream commit and archive SHA256')
    files, roots = manifest.get('files'), manifest.get('roots')
    if (not isinstance(files, dict) or not files or not isinstance(roots, list) or not roots
            or any(not _safe_relative(name) for name in files)
            or any(not isinstance(sha, str) or not _SHA256.fullmatch(sha) for sha in files.values())
            or any(not isinstance(name, str) or name not in files for name in roots)):
        raise RuntimeError('invalid vendored CUTLASS file paths, hashes, or include roots')
    include = root / _VENDOR / 'include'
    actual = {path.relative_to(include).as_posix() for path in include.rglob('*') if path.is_file()}
    if actual != set(files):
        missing, extra = sorted(set(files) - actual), sorted(actual - set(files))
        raise RuntimeError(f'vendored CUTLASS closure mismatch: missing={missing[:5]}, undeclared={extra[:5]}; '
                           'restore the pinned GitHub checkout')
    snapshot = {manifest_path: content}
    for name in sorted(files):
        path = include / name
        if path.resolve() != path.absolute():
            raise RuntimeError(f'vendored header escapes its physical include path: {name}')
        value = _read(path, 'vendored CUTLASS header')
        if _digest(value) != files[name]:
            raise RuntimeError(f'vendored CUTLASS SHA256 mismatch: {name}; restore the pinned GitHub checkout')
        snapshot[path] = value
    return snapshot


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
        raise RuntimeError('no candidate CUDA translation units under experimental/hopper_fp8')
    paths.append(root / 'csrc/expert_numerics.h')
    for path in paths:
        if path.resolve() != path.absolute():
            raise RuntimeError(f'candidate source must not use symlink paths: {path}')
        snapshot[path] = _read(path, 'candidate CUDA source/header')
    _verify_include_closure(root, snapshot)
    return {
        'schema_version': 1,
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
    return 'weave_hopper_fp8_' + _digest(_manifest_bytes({'runtime': runtime, 'source': manifest}))[:20]


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
