import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.weave import hopper_prefill_candidate as candidate

class PrefillCandidateTests(unittest.TestCase):
    def test_capacity(self):
        self.assertEqual(candidate.tile_capacity(8192,6,32),800)
        self.assertEqual(candidate.tile_capacity(1,6,32),33)
        for b in (0,8193):
            with self.assertRaises(ValueError): candidate.tile_capacity(b,6,32)
    def test_manifest(self):
        manifest=candidate.source_manifest()
        paths=[r['path'] for r in manifest['files']]
        self.assertIn('experimental/hopper_prefill/kernel.cu',paths)
        self.assertNotIn('experimental/hopper_fp8/kernel.cu',paths)
        self.assertEqual(manifest['numerical_policy'],'record_only')
        self.assertEqual(len(candidate.source_sha256()),64)
        self.assertTrue(candidate._extension_name('2.11','13',manifest).startswith('weave_hopper_prefill_'))
    def test_memory(self):
        b=candidate.workspace_bytes(8192,3072,1024,128,4,6)
        self.assertLess(b, 2*1024**3)
        self.assertGreater(b,8192*6*3072*4)

    def test_portable_header(self):
        import pathlib, shutil, subprocess, tempfile
        if not shutil.which('c++'): self.skipTest('C++ compiler required')
        root=pathlib.Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            exe=pathlib.Path(tmp)/'prefill'
            build=subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror',
                '-I',str(root/'experimental/hopper_prefill'),str(root/'tests/cpp/test_hopper_prefill_contract.cpp'),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(exe)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)

    def test_import_does_not_load_torch(self):
        import subprocess
        result=subprocess.run([sys.executable,'-c',
            'import sys; from src.weave import hopper_prefill_candidate; assert "torch" not in sys.modules'],
            cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_missing_vendor_fails_before_build(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(candidate,'__file__',str(Path(tmp)/'src/weave/hopper_prefill_candidate.py')):
                candidate.extension.cache_clear()
                with self.assertRaisesRegex(RuntimeError,'GitHub'):
                    candidate.extension()

    def test_independent_source_closure_rejects_unhashed_include(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            src=root/'experimental/hopper_prefill/kernel.cu'
            src.parent.mkdir(parents=True)
            src.write_text('#include "../../../outside.h"')
            with self.assertRaisesRegex(RuntimeError,'missing include'):
                candidate._verify_include_closure(root,{src:src.read_bytes()})

    def test_build_flags_affect_identity(self):
        from unittest.mock import patch
        before=candidate.source_sha256()
        with patch.object(candidate,'CUDA_CFLAGS',candidate.CUDA_CFLAGS+('-DTEST=1',)):
            self.assertNotEqual(before,candidate.source_sha256())
