"""Compile actual extracted type guards/lookup on CPU, not a CUDA build claim.

CUDA arithmetic is deliberately not emulated. Minimal scalar definitions stand in
for CUDA-only types; the template declaration, type assertions, packed-width
branch and div_ceil call are read unchanged from the delivered helper.
"""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
HEADER=ROOT/'experimental/native_tail/marlin_down_tile.cuh'


def probe_source(header):
    signature=re.search(r'(template<int threads,.*?>)\n__device__ __forceinline__ void run_marlin_down_tile',header,re.S).group(1)
    start=header.index('    int4* sh, int max_shared_mem) {')+len('    int4* sh, int max_shared_mem) {')
    prologue=header[start:header.index('  constexpr int top_k',start)]
    guards=header[header.index('  static constexpr auto w_type'):header.index('  constexpr bool has_zp')]
    packed=header[header.index('      if constexpr (w_type_id == host::kFE2M1f.id())'):header.index('      dequant_data(b_quant_0')]
    ceil=next(line for line in header.splitlines() if 'constexpr int a_sh_wr_iters =' in line)
    return '''#include <type_traits>
struct nv_bfloat16 {}; struct half {};
namespace host {
using ScalarTypeId=int;
struct ScalarType {
 int value;
 constexpr int id() const {return value;}
 constexpr int size_bits() const {return value==1?4:value>=4?16:8;}
 static constexpr ScalarType from_id(int id){return {id};}
 constexpr bool operator==(const ScalarType&) const=default;
};
constexpr ScalarType kFE2M1f{1},kFE8M0fnu{2},kFE4M3fn{3},kBFloat16{4},kFloat16{5};
template<class T,class U> constexpr auto div_ceil(T a,U b){return (a+b-1)/b;}
}
namespace device {template<class T,class U> constexpr auto div_ceil(T a,U b){return (a+b-1)/b;}}
namespace weave::native_tail {
''' + signature + '\nint probe() {\n'+prologue+guards+'''
 int frag_b_quant[2][2][4]={};int k2=0,j=0,b_quant_0=0,b_quant_1=0;
'''+packed+'''
 constexpr int a_sh_stage=96,a_sh_wr_delta=32;
'''+ceil+'''
 return a_sh_wr_iters;
}
// A device-only using-declaration must not leak into the host configurator.
using namespace host;
int configure(){return div_ceil(33,16);}
}
int main(){using namespace weave::native_tail;
 return configure()!=3 || probe<128,1,8,4,true>()!=3 || probe<128,1,8,4,false>()!=3
 || probe<256,2,8,4,false>()!=3 || probe<256,3,8,4,false>()!=3 || probe<256,4,8,4,false>()!=3;
}
'''


class DownCompileContractTests(unittest.TestCase):
    def test_scalar_and_quantization_ids_remain_template_dependent(self):
        header=HEADER.read_text()
        signature=re.search(r'(template<int threads,.*?>)\n__device__ __forceinline__ void run_marlin_down_tile',header,re.S).group(1)
        self.assertRegex(signature,r'typename scalar_t\s*=\s*nv_bfloat16')
        self.assertRegex(signature,r'host::ScalarTypeId w_type_id\s*=\s*host::kFE2M1f.id\(\)')
        self.assertRegex(signature,r'host::ScalarTypeId s_type_id\s*=\s*host::kFE8M0fnu.id\(\)')

    @unittest.skipUnless(shutil.which('c++'),'host C++ compiler unavailable')
    def test_actual_type_guards_packed_branch_and_div_ceil_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'probe.cpp';binary=Path(directory)/'probe'
            source.write_text(probe_source(HEADER.read_text()))
            build=subprocess.run(['c++','-std=c++20',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            subprocess.run([str(binary)],check=True)

    @unittest.skipUnless(shutil.which('c++'),'host C++ compiler unavailable')
    def test_non_native_arithmetic_override_is_rejected(self):
        source=probe_source(HEADER.read_text()).replace('probe<128,1,8,4,true>()','probe<128,1,8,4,true,4,half>()',1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.cpp';path.write_text(source)
            build=subprocess.run(['c++','-std=c++20','-fsyntax-only',str(path)],capture_output=True,text=True)
            self.assertNotEqual(build.returncode,0)
            self.assertIn('native tail requires BF16/MXFP4/E8M0',build.stderr)
