import ast
import importlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
FIXTURE=ROOT/'third_party/sglang_marlin_2948168'
PREFIX='kernels/jit/csrc/gemm/marlin_moe/'


class CostOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:cls.patch=importlib.import_module('src.weave.native_cost_patch')
        except ImportError:cls.patch=None

    def setUp(self):
        if self.patch is None and self._testMethodName!='test_cost_overlay_exists':self.skipTest('overlay missing')

    def sources(self):
        return {p.relative_to(FIXTURE).as_posix():p.read_text() for p in FIXTURE.rglob('*') if p.suffix in ('.py','.h','.cuh')}

    def test_cost_overlay_exists(self):
        self.assertIsNotNone(self.patch,'measured-cost Marlin overlay is not implemented')

    def test_overlay_python_keeps_native_signature_with_required_cost_buffers(self):
        result=self.patch.patch_sources(self.sources())
        tree=ast.parse(result['kernels/ops/moe/moe_wna16_marlin.py'])
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='moe_wna16_marlin_gemm')
        self.assertEqual([a.arg for a in function.args.kwonlyargs],
                         ['budget_state','cost_plan','cost_table','cost_control','cost_role','cost_refresh'])
        self.assertEqual(len(function.args.args),26)
        names=[n.name for n in tree.body if isinstance(n,ast.FunctionDef)]
        self.assertIn('weave_plan',names)
        # Code generation must complete against pinned input; altered sources never patch.
        inputs=self.sources();inputs[PREFIX+'kernel.h']+='\n'
        with self.assertRaisesRegex(RuntimeError,'source mismatch'):self.patch.patch_sources(inputs)

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler unavailable')
    def test_extended_kernel_macro_and_real_native_template_declarations_typecheck(self):
        result=self.patch.patch_sources(self.sources())
        kernel=result[PREFIX+'kernel.h']
        template=result[PREFIX+'marlin_template.h']
        # Compile the actual generated Marlin declaration and active definition
        # signature in a CPU TU; CUDA arithmetic is outside this host check.
        kernel=re.sub(r'^#include.*$', '',kernel,flags=re.M)
        definitions=list(re.finditer(r'template\s*<[^;{}]+>\s*__global__ void Marlin\(',template))
        start=definitions[-1].start();end=template.index(') {',definitions[-1].end())+1
        signature=template[start:end]
        code='#include <cstdint>\n#define __global__\nstruct int4{};namespace host{using ScalarTypeId=int;}\n'
        code+=kernel+'\nnamespace device::marlin_moe {\n'+signature+' {}\n'
        code+='void macro_probe(MARLIN_KERNEL_PARAMS){};\n'
        code+='static_assert(__is_same(decltype(&macro_probe), decltype(&Marlin<int,0,0,128,1,8,8,false,4,2,false,true,false>)));}\n'
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'signature.cpp';path.write_text(code)
            subprocess.run(['c++','-std=c++17','-fsyntax-only',str(path)],check=True,capture_output=True)

    def test_cost_replaces_group_scan_preserving_native_launch_selection_and_all_lock_uses(self):
        sources=self.sources();result=self.patch.patch_sources(sources)
        # Parse the native selection function boundary and compare it verbatim:
        # it must still query the candidate kernel's actual register attributes.
        def launch_algorithm(text):
            start=text.index('exec_config_t determine_exec_config(')
            end=text.index('\ntemplate <typename scalar_t, bool kIsEP, bool kHasBias>\nvoid marlin_mm',start)
            return text[start:end]
        host_name=PREFIX+'moe_wna16_marlin.cuh'
        self.assertEqual(launch_algorithm(sources[host_name]),launch_algorithm(result[host_name]))
        original=sources[PREFIX+'marlin_template.h']
        patched=result[PREFIX+'marlin_template.h']
        self.assertEqual(original.count('gridDim.x'),3)
        self.assertEqual(patched.count('parallel * n_tiles >= weave_workers'),2)
        self.assertEqual(patched.count('div_ceil(k_tiles * n_tiles * parallel, weave_workers)'),1)
        self.assertNotIn('if (expert_ids_ptr[i] == -1) num_valid_blocks--;',patched)
        self.assertNotIn('weave_budget_local_routes',patched)
        self.assertLess(patched.index('if (int(blockIdx.x) >= weave_workers) return;'),patched.index('int slice_row ='))
        self.assertIn('if (cost_role == 0 || cost_refresh)',result[host_name])


if __name__=='__main__':unittest.main()
