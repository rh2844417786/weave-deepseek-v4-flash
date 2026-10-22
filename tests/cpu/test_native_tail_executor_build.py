"""S3 build-boundary regressions without pretending to compile CUDA locally."""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from src.weave.native_tail_executor import NativeTailExecutor

ROOT=Path(__file__).resolve().parents[2]


class ExecutorBuildTests(unittest.TestCase):
    def test_current_arch_macro_is_passed_to_nvcc(self):
        cuda=NS(is_current_stream_capturing=lambda:False,get_device_capability=Mock(return_value=(9,0)))
        module=NS(resources=lambda scratch:dict(workers=132,scratch_bytes=scratch))
        load=Mock(return_value=module)
        libinfo=NS(find_include_path=lambda:'/installed/tvm/include',
                   find_dlpack_include_path=lambda:'/installed/dlpack/include',
                   find_libtvm_ffi=lambda:'/installed/lib/libtvm_ffi.so')
        geometry=dict(threads=128,m_blocks=1,n_blocks=8,k_blocks=4,m8=True,scratch_bytes=32768)
        with tempfile.TemporaryDirectory() as directory:
            engine=object.__new__(NativeTailExecutor)
            engine.torch=NS(cuda=cuda,__version__='test-runtime');engine.modules={}
            engine.cache_root=Path(directory)
            engine.down_op=NS(helper_header=Path('/overlay/marlin_down_tile.cuh'),include_dirs=['/overlay/include'])
            with patch.dict(sys.modules,{'torch.utils.cpp_extension':NS(load=load),'tvm_ffi':NS(libinfo=libinfo)}),patch.dict('os.environ',{},clear=False):
                compiled,resources=engine._compile(geometry)
            self.assertIs(compiled,module)
            self.assertEqual(resources['workers'],132)
            kwargs=load.call_args.kwargs
            self.assertIn('-DSGL_CUDA_ARCH=900',kwargs['extra_cuda_cflags'])
            cuda.get_device_capability.assert_called_once_with()
            self.assertIn('-std=c++20',kwargs['extra_cuda_cflags'])

    @unittest.skipUnless(shutil.which('c++'),'host C++ compiler unavailable')
    def test_clock_query_compiles_without_cuda_device_prop_clock_rate(self):
        source=(ROOT/'experimental/native_tail/executor.cu').read_text()
        launch=source[source.index('void launch('):source.index('pybind11::dict snapshot')]
        query=launch[launch.index('  cudaDeviceProp prop;'):launch.index('  int active;')]
        # If a separately named device is used, preserve that actual declaration.
        declaration=re.search(r'  const int device=activation.get_device\(\);',launch)
        if declaration:query=declaration.group(0)+'\n'+query
        assignment=re.search(r'a.transport.timeout_cycles\s*=.*?;',launch).group(0)
        probe='''#include <cstdint>
using U64=unsigned long long;
struct cudaDeviceProp {}; // CUDA13 no longer supplies clockRate here.
constexpr int cudaDevAttrClockRate=13;
int calls=0;
void check(int value){if(value)throw value;}
int cudaGetDeviceProperties(cudaDeviceProp*,int device){return device==3?0:1;}
int cudaDeviceGetAttribute(int*value,int attribute,int device){
 if(attribute!=cudaDevAttrClockRate||device!=3)return 1;
 *value=1800000;++calls;return 0;
}
struct {int get_device()const{return 3;}} activation;
int main(){
 struct {struct {U64 timeout_cycles;}transport;} a;
 double timeout=2.5;
'''+query+assignment+'''
 return calls!=1||a.transport.timeout_cycles!=4500000000ULL;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'clock.cpp';binary=Path(directory)/'clock';path.write_text(probe)
            build=subprocess.run(['c++','-std=c++20',str(path),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            subprocess.run([str(binary)],check=True)
