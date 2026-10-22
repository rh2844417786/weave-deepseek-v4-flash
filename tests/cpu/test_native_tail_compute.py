"""Source extraction and exact task/layout contracts; CUDA runs separately."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'third_party/sglang_marlin_2948168'


class DownComputeTests(unittest.TestCase):
    def api(self):
        path = ROOT / 'src/weave/native_tail_patch.py'
        self.assertTrue(path.is_file(), 'S1 private overlay implementation missing')
        spec = importlib.util.spec_from_file_location('native_tail_patch_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_extraction_has_native_math_and_no_global_partner(self):
        api = self.api()
        source = (FIXTURE / api.TEMPLATE).read_text()
        helper = api.extract_down_header(source)
        self.assertEqual(helper, (ROOT / 'experimental/native_tail/marlin_down_tile.cuh').read_text())
        for unwanted in ('gridDim', 'blockIdx', 'locks[', 'global_reduce', 'barrier_acquire', '__global__'):
            self.assertNotIn(unwanted, helper)
        # These complete source regions contain native packed addressing,
        # dequantization, MMA, and CTA-local reduction, unchanged byte for byte.
        for begin, end in api.PRESERVED_REGIONS:
            original = source[source.index(begin):source.index(end)]
            self.assertIn(original, helper)
        self.assertIn('int slice_row = 0;', helper)
        self.assertIn('int slice_iters = k_tiles;', helper)
        self.assertIn('cp_async_wait<0>();\n  __syncthreads();', helper)

    def test_upstream_mismatch_is_rejected(self):
        api = self.api()
        source = (FIXTURE / api.TEMPLATE).read_text()
        with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
            api.extract_down_header(source + '\n')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
                api.verify_sources(directory)

    def test_exact_pair_channel_coverage_all_native_m_blocks(self):
        api = self.api()
        for block in (8, 16, 32, 48, 64):
            for batch in (1, 32, 65, 128):
                rows = batch * 6
                # Deliberate empty, partial and multiple groups of an expert.
                groups = [[i for i in range(rows) if i % 4 == e] for e in range(4)]
                sorted_ids, experts = [], []
                for e, slots in enumerate(groups):
                    for start in range(0, len(slots), block):
                        part = slots[start:start + block]
                        sorted_ids.extend(part + [rows] * (block - len(part)))
                        experts.append(-1 if e == 3 else e)
                result = api.audit_task_coverage(sorted_ids, experts, len(sorted_ids), rows, 4096,
                                                 block, 128, num_experts=64)
                self.assertEqual(result['valid_pairs'], sum(i % 4 != 3 for i in range(rows)))
                self.assertEqual(result['covered_elements'], result['valid_pairs'] * 4096)
                self.assertEqual(result['local_tasks'], sum(e >= 0 for e in experts) * 32)

    def test_bad_grouping_rejected_and_empty_work_valid(self):
        api = self.api()
        empty = api.audit_task_coverage([0] + [1] * 7, [-1], 8, 1, 128, 8, 128, num_experts=64)
        self.assertEqual(empty['local_tasks'], 0)
        for ids, experts in (([0, 0] + [2] * 6, [0]), ([-1] + [2] * 7, [0]),
                             ([0] + [2] * 7, [64]), ([2, 0] + [2] * 6, [0])):
            with self.assertRaises(ValueError):
                api.audit_task_coverage(ids, experts, 8, 2, 128, 8, 128, num_experts=64)

    def test_packed_channel_offsets_cover_original_weight_and_scales(self):
        api = self.api()
        # int4 strides, independent of row-major scalar slicing assumptions.
        for n in (128, 256, 4096):
            for tile_n in (128, 256):
                if n % tile_n:
                    continue
                offsets = [api.packed_offsets(3, col, n, 2048, tile_n)
                           for col in range(n // tile_n)]
                self.assertEqual(offsets[0], (3 * n * 2048 // 32, 3 * n * 2048 // 512))
                self.assertEqual(offsets[-1][0] + tile_n // 2,
                                 3 * n * 2048 // 32 + n // 2)
                self.assertEqual(offsets[-1][1] + tile_n // 16,
                                 3 * n * 2048 // 512 + n // 16)

    def test_host_keeps_native_register_aware_selection_algorithm(self):
        api = self.api()
        host = (FIXTURE / api.HOST).read_text()
        generated = api.extract_static_host(host)
        start = host.index('template <typename scalar_t, bool kIsEP, bool kHasBias>\nexec_config_t determine_exec_config')
        end = host.index('template <typename scalar_t, bool kIsEP, bool kHasBias>\nvoid marlin_mm', start)
        self.assertIn(host[start:end], generated)
        self.assertIn('cudaOccupancyMaxActiveBlocksPerMultiprocessor', generated)
        self.assertIn('run_marlin_down_tile<', generated)


if __name__ == '__main__':
    unittest.main()
