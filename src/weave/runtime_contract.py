"""Runtime validation without importing SGLang or constructing its model."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_contract():
    return json.loads((ROOT/'configs/runtime.json').read_text())


def validate_inventory(observed, stage, contract=None):
    contract = contract or load_contract()
    if stage not in ('core', 'native', 'adapter'):
        raise ValueError('unknown validation stage')
    errors = []
    packages = observed.get('packages', {})
    checks = [('python', observed.get('python'), contract['python_version']),
              ('torch', packages.get('torch'), contract['torch_version']),
              ('torch CUDA', observed.get('torch_cuda'), contract['torch_cuda']),
              ('nvcc', observed.get('nvcc_release'), contract['nvcc_release'])]
    if stage != 'core':
        checks += [('sglang', packages.get('sglang'), contract['sglang_version']),
                   ('transformers', packages.get('transformers'), contract['transformers_version'])]
    for name, actual, expected in checks:
        if actual != expected:
            errors.append(f'{name}: expected {expected}, observed {actual}')
    if not observed.get('cuda_available'):
        errors.append('PyTorch CUDA is unavailable')
    devices = observed.get('devices', [])
    if len(devices) != 4:
        errors.append(f'expected four container-visible GPUs, observed {len(devices)}')
    if any(device.get('capability') != [9, 0] for device in devices):
        errors.append('all selected GPUs must be Hopper SM90')
    return errors


def verify_adapter_sources(package_root=None, expected=None):
    if expected is None:
        manifest = json.loads((ROOT/'configs/sglang_adapter_sources.json').read_text())
        expected = manifest['sha256']
    if package_root is None:
        spec = importlib.util.find_spec('sglang')
        if spec is None or not spec.submodule_search_locations:
            raise ValueError('cannot locate installed SGLang source files')
        package_root = Path(next(iter(spec.submodule_search_locations)))
    hashes, errors = {}, []
    for relative, pinned in expected.items():
        path = Path(package_root)/relative
        if not path.is_file():
            errors.append(f'{relative}: missing')
            continue
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[relative] != pinned:
            errors.append(f'{relative}: source SHA256 differs from the audited commit ({hashes[relative]})')
    if errors:
        raise ValueError('; '.join(errors))
    return hashes
