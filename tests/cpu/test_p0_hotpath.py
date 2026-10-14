"""Host-path contracts; CUDA allocation is represented by instrumented views."""
from collections import Counter
from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from src.weave import ops


class Tensor:
    def __init__(self, shape, dtype, device=None):
        self.shape, self.dtype = shape, dtype
        self.device = device or types.SimpleNamespace(index=0)
        self.ndim = len(shape)
        self.copies = []
        self.fills = []
        self.conversions = []

    def __getitem__(self, index):
        return self

    def copy_(self, value):
        self.copies.append(value)
        return self

    def fill_(self, value):
        self.fills.append(value)
        return self

    def to(self, dtype):
        self.conversions.append(dtype)
        return self

    def reshape(self, *shape):
        return self

    def numel(self):
        return 1


class Workspace:
    def __init__(self, *args):
        self.reads = Counter()
        self.views = {name: Tensor((2, 32), 'workspace')
                      for name in ('x', 'ids', 'routing', 'n', 'output')}
        self.calls = []

    def __getattr__(self, name):
        if name in self.views:
            self.reads[name] += 1
            return self.views[name]
        raise AttributeError(name)

    def handle(self):
        return b'handle'

    def connect(self, handles):
        self.handles = handles

    def bind_weights(self, raw):
        # Deliberately not iterable: dispatch must preserve the native object.
        return types.SimpleNamespace(owner=self, weights=tuple(raw))

    def run(self, *args):
        self.calls.append(('raw', args))

    def run_bound(self, *args):
        self.calls.append(('bound', args))


@contextmanager
def runtime_fixture(**kwargs):
    dist = types.ModuleType('torch.distributed')
    dist.get_rank = lambda group: 0
    dist.get_world_size = lambda group: 1
    dist.get_backend = lambda group: 'gloo'
    dist.all_gather_object = lambda values, item, group=None: values.__setitem__(0, item)
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(current_device=lambda: 0)
    torch.distributed = dist
    torch.bfloat16, torch.int32, torch.float32, torch.float64 = 'bf16', 'i32', 'f32', 'f64'
    torch.empty = lambda shape, dtype, device: Tensor(shape, dtype)
    module = types.SimpleNamespace(Workspace=Workspace, weave_source_sha256='source')
    with patch.dict('sys.modules', {'torch': torch, 'torch.distributed': dist}), \
            patch.object(ops, 'extension', return_value=module):
        runtime = ops.PeerRuntime(None, 2, 32, 32, 2, 8, **kwargs)
    x = Tensor((2, 32), 'bf16')
    ids, routing = Tensor((2, 2), 'i64', x.device), Tensor((2, 2), 'f64', x.device)
    yield runtime, x, ids, routing


class P0HotpathTest(unittest.TestCase):
    def test_default_caches_each_view_once_and_keeps_per_call_conversions(self):
        with runtime_fixture() as (runtime, x, ids, routing):
            for _ in range(3):
                result = runtime.run(x, ids, routing, [object()] * 6)
            self.assertIs(result, runtime.ws.views['output'])
            self.assertEqual(dict(runtime.ws.reads), dict.fromkeys(runtime.ws.views, 1))
            self.assertEqual(ids.conversions, ['i32'] * 3)
            self.assertEqual(routing.conversions, ['f32'] * 3)
            self.assertEqual(runtime.ws.views['n'].fills, [2] * 3)

    def test_uncached_control_recreates_each_view_per_call(self):
        with runtime_fixture(cache_views=False) as (runtime, x, ids, routing):
            self.assertEqual(dict(runtime.ws.reads), {})
            for _ in range(3):
                runtime.run(x, ids, routing, [object()] * 6)
            self.assertEqual(dict(runtime.ws.reads), dict.fromkeys(runtime.ws.views, 3))

    def test_bound_dispatch_preserves_object_and_raw_control_stays_available(self):
        with runtime_fixture() as (runtime, x, ids, routing):
            raw = tuple(object() for _ in range(6))
            bound = runtime.bind_weights(raw)
            runtime.run(x, ids, routing, bound)
            runtime.run(x, ids, routing, raw)
            self.assertEqual([kind for kind, _ in runtime.ws.calls], ['bound', 'raw'])
            self.assertIs(runtime.ws.calls[0][1][0], bound)
            self.assertEqual(runtime.ws.calls[1][1][0], list(raw))
            self.assertEqual(runtime.ws.calls[0][1][1:], runtime.ws.calls[1][1][1:])

    def test_invalid_dynamic_input_still_fails_before_bound_launch(self):
        with runtime_fixture() as (runtime, x, ids, routing):
            bound = runtime.bind_weights([object()] * 6)
            ids.device = types.SimpleNamespace(index=1)
            with self.assertRaisesRegex(ValueError, 'workspace device'):
                runtime.run(x, ids, routing, bound)
            self.assertEqual(runtime.ws.calls, [])


class SourceManifestTest(unittest.TestCase):
    def test_file_added_after_first_manifest_read_rejects_incomplete_build(self):
        # A late .cu must not be certified by a digest that includes it when the
        # compiler source list came from the earlier directory enumeration.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / 'csrc').mkdir()
            original = root / 'csrc/megakernel.cu'
            original.write_bytes(b'main')
            late = root / 'csrc/staged_compute.cu'
            read_bytes = Path.read_bytes

            def read_then_add_source(path):
                content = read_bytes(path)
                if path == original and not late.exists():
                    late.write_bytes(b'new runtime source')
                return content

            torch = types.ModuleType('torch')
            torch.__version__ = 'test-torch'
            torch.version = types.SimpleNamespace(cuda='test-cuda')
            torch.cuda = types.SimpleNamespace(is_available=lambda: True)
            cpp_extension = types.ModuleType('torch.utils.cpp_extension')
            cpp_extension.load = lambda **kwargs: types.SimpleNamespace()
            modules = {'torch': torch, 'torch.utils': types.ModuleType('torch.utils'),
                       'torch.utils.cpp_extension': cpp_extension}
            ops.extension.cache_clear()
            try:
                with patch.dict('sys.modules', modules), \
                        patch.dict('os.environ', {'TORCH_CUDA_ARCH_LIST': '9.0a'}), \
                        patch.object(ops, '__file__', str(root / 'src/weave/ops.py')), \
                        patch.object(Path, 'read_bytes', read_then_add_source):
                    with self.assertRaisesRegex(RuntimeError, 'source changed during extension build'):
                        ops.extension()
            finally:
                ops.extension.cache_clear()

    def test_manifest_tracks_new_nested_runtime_sources_and_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'csrc/tiles').mkdir(parents=True)
            files = {'csrc/megakernel.cu': b'main', 'csrc/tiles/math.cuh': b'math',
                     'csrc/staged_compute.cu': b'staged', 'csrc/tiles/layout.h': b'layout'}
            for name, content in files.items():
                (root / name).write_bytes(content)
            (root / 'csrc/ignored.txt').write_text('not compiled')
            with patch.object(ops, '__file__', str(root / 'src/weave/ops.py')):
                manifest = ops.kernel_source_manifest()
                self.assertEqual(manifest['files'], [
                    {'path': name, 'sha256': hashlib.sha256(files[name]).hexdigest()}
                    for name in sorted(files)])
                self.assertIn('-Xptxas=-v', manifest['cuda_cflags'])
                self.assertNotIn('--use_fast_math', manifest['cuda_cflags'])
                before = ops.kernel_source_sha256()
                (root / 'csrc/tiles/math.cuh').write_bytes(b'different math')
                self.assertNotEqual(before, ops.kernel_source_sha256())
                before = ops.kernel_source_sha256()
                with patch.object(ops, 'CUDA_CFLAGS', (*ops.CUDA_CFLAGS, '-DTEST_FLAG')):
                    self.assertNotEqual(before, ops.kernel_source_sha256())


if __name__ == '__main__':
    unittest.main()
