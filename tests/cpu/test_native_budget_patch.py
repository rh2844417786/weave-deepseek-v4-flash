import ast
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from src.weave.native_budget_patch import patch_sources, verify_sources
from src.weave.native_budget import ROOT, select_workers

FIXTURE = ROOT / 'third_party/sglang_marlin_2948168'

class PatchTests(unittest.TestCase):
    def sources(self):
        return {p.relative_to(FIXTURE).as_posix(): p.read_text()
                for p in FIXTURE.rglob('*') if p.suffix in ('.py', '.h', '.cuh')}

    def test_patch_changes_all_grid_uses_and_ffi_consistently(self):
        source = self.sources(); result = patch_sources(source)
        template = result['kernels/jit/csrc/gemm/marlin_moe/marlin_template.h']
        self.assertEqual(template.count('gridDim.x'), 2) # choice input and native-grid evidence
        self.assertEqual(template.count('parallel * n_tiles >= weave_workers'), 2)
        self.assertIn('if (int(blockIdx.x) >= weave_workers) return;', template)
        self.assertLess(template.index('if (int(blockIdx.x) >= weave_workers) return;'),
                        template.index('int slice_row ='))
        host = result['kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh']
        self.assertIn('int blocks = sms * exec_cfg.blocks_per_sm;', host)
        self.assertIn('with_dtype<int64_t>()', host)
        self.assertIn('static_cast<int64_t*>(budget_state.data_ptr())', host)
        op = result['kernels/ops/moe/moe_wna16_marlin.py']
        ast.parse(op)
        self.assertIn('external_cuda_files=[WEAVE_BUDGET_HEADER]', op)
        self.assertIn('budget_state, budget_target_percent, budget_ep_size, budget_mode,', op)

    def test_refuses_modified_upstream(self):
        source = self.sources(); source[next(iter(source))] += '\n// changed\n'
        with self.assertRaisesRegex(RuntimeError, 'source mismatch'):patch_sources(source)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, 'source mismatch'):verify_sources(td)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler unavailable')
    def test_cpp_policy_matches_python_on_route_and_geometry_matrix(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'policy.cpp';exe=root/'policy'
            src.write_text('#include <iostream>\n#include "native_budget.h"\n'
                'int main(){int g,l,t,b,n,k,p,e,m;while(std::cin>>g>>l>>t>>b>>n>>k>>p>>e>>m)'
                '{auto c=weave_budget_choose(g,l,t,b,n,k,p,e,m);std::cout<<c.workers<<" "<<c.bucket<<"\\n";}}')
            subprocess.run(['c++','-std=c++17','-I',str(ROOT/'csrc'),str(src),'-o',str(exe)],check=True,capture_output=True)
            cases=[(g,l,192,min(l,64),n,k,p,4,m) for g in (4,33,132,528)
                   for l in (0,1,3,12,48,84,192) for n,k in ((1,1),(16,64),(32,112))
                   for p in (1,75,100) for m in (1,2)]
            got=subprocess.run([str(exe)],input=''.join(' '.join(map(str,c))+'\n' for c in cases),text=True,capture_output=True,check=True)
            expected=[select_workers(*c[:6],target_percent=c[6],ep_size=c[7],mode='adaptive' if c[8]==1 else 'full') for c in cases]
            self.assertEqual([tuple(map(int,row.split())) for row in got.stdout.splitlines()],expected)
