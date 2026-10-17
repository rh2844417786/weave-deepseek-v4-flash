"""Offline source closure and hand-derived E4M3FN quantization tests."""
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _modules():
    try:
        return (importlib.import_module('src.weave.hopper_candidate'),
                importlib.import_module('src.weave.hopper_reference'))
    except ModuleNotFoundError as exc:
        raise AssertionError('Hopper loader and independent reference are not implemented') from exc


def fixture(root):
    files = {
        'experimental/hopper_fp8/kernel.cu': b'#include "math_contract.h"\n#include <cute/arch/mma.h>\n',
        'experimental/hopper_fp8/math_contract.h': b'#include "../../csrc/expert_numerics.h"\n',
        'csrc/expert_numerics.h': b'// BF16 boundaries\n',
        'third_party/cutlass_hopper/include/cute/arch/mma.h': b'#include "cute/config.h"\n',
        'third_party/cutlass_hopper/include/cute/config.h': b'// pinned config\n',
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    manifest = {'schema_version': 1, 'upstream': {
        'repository': 'https://github.com/NVIDIA/cutlass',
        'commit': 'f3fde58372d33e9a5650ba7b80fc48b3b49d40c8',
        'archive_sha256': 'a' * 64}, 'roots': ['cute/arch/mma.h'],
        'files': {name.removeprefix('third_party/cutlass_hopper/include/'): hashlib.sha256(value).hexdigest()
                  for name, value in files.items() if '/include/' in name}}
    path = root / 'third_party/cutlass_hopper/manifest.json'
    path.write_text(json.dumps(manifest))
    return files, path


class OfflineSourceTest(unittest.TestCase):
    def test_modules_import_without_torch(self):
        _modules()
        result = subprocess.run([sys.executable, '-c',
            'import sys; from src.weave import hopper_candidate, hopper_reference; '
            'assert "torch" not in sys.modules'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_vendor_fails_before_any_torch_import(self):
        candidate, _ = _modules()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with patch.object(candidate, '__file__', str(root / 'src/weave/hopper_candidate.py')):
                candidate.extension.cache_clear()
                with self.assertRaisesRegex(RuntimeError, 'manifest.*GitHub|GitHub.*manifest'):
                    candidate.extension()

    def test_missing_corrupt_or_extra_header_rejects_vendor(self):
        candidate, _ = _modules()
        for mutation in ('missing', 'corrupt', 'extra'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                fixture(root)
                path = root / 'third_party/cutlass_hopper/include/cute/config.h'
                if mutation == 'missing':
                    path.unlink()
                elif mutation == 'corrupt':
                    path.write_text('changed')
                else:
                    path.with_name('unlisted.h').write_text('unexpected')
                with patch.object(candidate, '__file__', str(root / 'src/weave/hopper_candidate.py')):
                    with self.assertRaisesRegex(RuntimeError, 'vendor|vendored'):
                        candidate.source_manifest()

    def test_rejects_unpinned_and_unsafe_manifest(self):
        candidate, _ = _modules()
        for mutation in ('commit', 'path', 'sha', 'root'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                _, path = fixture(root)
                data = json.loads(path.read_text())
                if mutation == 'commit':
                    data['upstream']['commit'] = '0' * 40
                elif mutation == 'path':
                    data['files']['../outside.h'] = '0' * 64
                elif mutation == 'sha':
                    data['files']['cute/config.h'] = 'not-a-hash'
                else:
                    data['roots'] = ['cute/missing.h']
                path.write_text(json.dumps(data))
                with patch.object(candidate, '__file__', str(root / 'src/weave/hopper_candidate.py')):
                    with self.assertRaises(RuntimeError):
                        candidate.source_manifest()

    def test_all_candidate_sources_and_numeric_header_affect_identity(self):
        candidate, _ = _modules()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            files, vendor_manifest = fixture(root)
            with patch.object(candidate, '__file__', str(root / 'src/weave/hopper_candidate.py')):
                manifest = candidate.source_manifest()
                paths = {row['path'] for row in manifest['files']}
                self.assertEqual(paths, set(files) | {vendor_manifest.relative_to(root).as_posix()})
                initial = candidate.source_sha256()
                (root / 'csrc/unrelated.cu').write_text('BF16 core only')
                self.assertEqual(initial, candidate.source_sha256())
                for name in ('experimental/hopper_fp8/math_contract.h', 'csrc/expert_numerics.h'):
                    before = candidate.source_sha256()
                    with (root / name).open('a') as out:
                        out.write('// changed\n')
                    self.assertNotEqual(before, candidate.source_sha256())
                before = candidate.source_sha256()
                (root / 'experimental/hopper_fp8/additional.cu').write_text('// new translation unit\n')
                self.assertNotEqual(before, candidate.source_sha256())
                before = candidate.source_sha256()
                with patch.object(candidate, 'NUMERIC_CONTRACT', 'changed-contract'):
                    self.assertNotEqual(before, candidate.source_sha256())
                with patch.object(candidate, 'CUDA_CFLAGS', (*candidate.CUDA_CFLAGS, '-DTEST_FLAG')):
                    self.assertNotEqual(before, candidate.source_sha256())
                with patch.dict('os.environ', {'TORCH_CUDA_ARCH_LIST': '8.0'}):
                    self.assertEqual(before, candidate.source_sha256())

    def test_missing_quoted_transitive_dependency_is_rejected(self):
        candidate, _ = _modules()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            fixture(root)
            (root / 'experimental/hopper_fp8/math_contract.h').write_text('#include "missing.cuh"\n')
            with patch.object(candidate, '__file__', str(root / 'src/weave/hopper_candidate.py')):
                with self.assertRaisesRegex(RuntimeError, 'include|closure'):
                    candidate.source_manifest()

    def test_unmanifested_local_dependency_is_rejected(self):
        candidate, _ = _modules()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            fixture(root)
            (root / 'csrc/unlisted.h').write_text('// other header\n')
            (root / 'experimental/hopper_fp8/math_contract.h').write_text('#include "../../csrc/unlisted.h"\n')
            with patch.object(candidate, '__file__', str(root / 'src/weave/hopper_candidate.py')):
                with self.assertRaisesRegex(RuntimeError, 'closure'):
                    candidate.source_manifest()

    def test_single_build_snapshot_and_changed_source_rejection(self):
        candidate, _ = _modules()
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                fixture(root)
                torch = types.ModuleType('torch')
                torch.__version__ = '2.11.0'
                torch.version = types.SimpleNamespace(cuda='13.0')
                torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (9, 0))
                compiler = types.ModuleType('torch.utils.cpp_extension')
                def load(**kwargs):
                    # The build must consume the same source set as its identity.
                    self.assertEqual(kwargs['sources'], [str(root / 'experimental/hopper_fp8/kernel.cu')])
                    self.assertIn('-gencode=arch=compute_90a,code=sm_90a', kwargs['extra_cuda_cflags'])
                    self.assertNotIn('--use_fast_math', kwargs['extra_cuda_cflags'])
                    self.assertEqual(kwargs['extra_ldflags'], ['-lcuda'])
                    self.assertIn(str(root / 'third_party/cutlass_hopper/include'), kwargs['extra_include_paths'])
                    if changed:
                        (root / 'experimental/hopper_fp8/late.cu').write_text('// late\n')
                    return types.SimpleNamespace()
                compiler.load = load
                modules = {'torch': torch, 'torch.utils': types.ModuleType('torch.utils'),
                           'torch.utils.cpp_extension': compiler}
                candidate.extension.cache_clear()
                try:
                    with patch.dict(sys.modules, modules), patch.object(candidate, '__file__', str(root / 'src/weave/hopper_candidate.py')):
                        if changed:
                            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                                candidate.extension()
                        else:
                            result = candidate.extension()
                            self.assertEqual(result.weave_source_sha256, candidate.source_sha256())
                            self.assertEqual(result.weave_source_manifest, candidate.source_manifest())
                finally:
                    candidate.extension.cache_clear()


class IndependentEncodingTest(unittest.TestCase):
    def test_e4m3_finite_values_subnormal_and_saturation(self):
        _, reference = _modules()
        fixtures = [(0.0, 0x00), (-0.0, 0x80), (2 ** -9, 0x01), (2 ** -6, 0x08),
                    (0.5, 0x30), (1., 0x38), (1.5, 0x3c), (2., 0x40),
                    (3., 0x44), (4., 0x48), (6., 0x4c), (448., 0x7e),
                    (449., 0x7e), (1e30, 0x7e), (-1e30, 0xfe),
                    (math.inf, 0x7e), (-math.inf, 0xfe)]
        for value, expected in fixtures:
            with self.subTest(value=value):
                self.assertEqual(reference.encode_fp8_e4m3(value), expected)
        self.assertTrue(math.isnan(reference.decode_fp8_e4m3(0x7f)))
        self.assertTrue(math.isnan(reference.decode_fp8_e4m3(0xff)))
        self.assertEqual(math.copysign(1., reference.decode_fp8_e4m3(0x80)), -1.)

    def test_rne_ties_choose_even_mantissa_across_exponents(self):
        _, reference = _modules()
        fixtures = [(2 ** -10, 0x00), (3 * 2 ** -10, 0x02), (1.0625, 0x38),
                    (1.1875, 0x3a), (1.9375, 0x40), (432., 0x7e),
                    (-1.0625, 0xb8), (-2 ** -10, 0x80)]
        for value, expected in fixtures:
            with self.subTest(value=value):
                self.assertEqual(reference.encode_fp8_e4m3(value), expected)
        self.assertEqual(reference.encode_fp8_e4m3(math.nextafter(1.0625, 0.)), 0x38)
        self.assertEqual(reference.encode_fp8_e4m3(math.nextafter(1.0625, 2.)), 0x39)

    def test_finite_encoding_round_trip_exhaustive(self):
        _, reference = _modules()
        for code in range(256):
            if code & 0x7f == 0x7f:
                continue
            self.assertEqual(reference.encode_fp8_e4m3(reference.decode_fp8_e4m3(code)), code)
        with self.assertRaises(ValueError):
            reference.encode_fp8_e4m3(math.nan)

    def test_per_128_block_scaling_zero_policy_and_bf16_input_boundary(self):
        _, reference = _modules()
        dequant, codes, scales = reference.quantize_fp8_rows([[0.] * 128 + [448., -224.] + [0.] * 126])
        self.assertEqual(scales, [[1., 1.]])
        self.assertEqual(codes[0][:128], [0] * 128)
        self.assertEqual(codes[0][128:130], [0x7e, 0xf6])
        self.assertEqual(dequant[0][128:130], [448., -224.])
        dequant, codes, scales = reference.quantize_fp8_rows([[224., -112.] + [0.] * 126])
        self.assertEqual(scales, [[0.5]])
        self.assertEqual(codes[0][:2], [0x7e, 0xf6])
        self.assertEqual(dequant[0][:2], [224., -112.])
        for rows in ([[]], [[0.] * 127], [[math.inf] * 128], [[0.] * 128, [0.] * 256]):
            with self.assertRaises(ValueError):
                reference.quantize_fp8_rows(rows)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch oracle runs in the existing server image')
class TorchOracleTest(unittest.TestCase):
    def test_tensor_quantizer_matches_hand_checked_block_codes(self):
        import torch
        _, reference = _modules()
        row = [448., -224., 1., -0.] + [0.] * 124 + [224., -112.] + [0.] * 126
        tensor = torch.tensor([row], dtype=torch.bfloat16)
        dequant, codes, scales = reference.quantize_fp8_reference(tensor)
        self.assertEqual(scales.tolist(), [[1., .5]])
        self.assertEqual(codes[0, :4].tolist(), [0x7e, 0xf6, 0x38, 0x80])
        self.assertEqual(codes[0, 128:130].tolist(), [0x7e, 0xf6])
        self.assertEqual(dequant[0, :3].tolist(), [448., -224., 1.])
        self.assertEqual(dequant[0, 128:130].tolist(), [224., -112.])
        self.assertTrue(torch.signbit(dequant[0, 3]).item())
        with self.assertRaisesRegex(ValueError, 'BF16'):
            reference.quantize_fp8_reference(tensor.float())

    def test_original_nibbles_scales_and_bf16_swiglu_produce_known_output(self):
        import torch
        _, reference = _modules()
        # Code 2 is +1; byte 0x22 packs two +1 values, E8M0=120
        # scales them by 1/128. Each gate/up dot with 128 ones is 1.
        # BF16(silu(1)) = 0.73046875; the down matrix preserves it.
        weights = [torch.full((1, 128, 64), 0x22, dtype=torch.uint8) for _ in range(3)]
        scales = [torch.full((1, 128, 4), 120, dtype=torch.uint8) for _ in range(3)]
        raw = tuple(weights + scales)
        x = torch.stack((torch.ones(128), torch.zeros(128))).bfloat16()
        result = reference.expert_reference_fp8(x, raw, 0)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result[0].tolist(), [.73046875] * 128)
        self.assertEqual(result[1].tolist(), [0.] * 128)
        raw[3][0, 0, 0] = 255
        with self.assertRaisesRegex(ValueError, 'E8M0'):
            reference.expert_reference_fp8(x, raw, 0)


if __name__ == '__main__':
    unittest.main()
