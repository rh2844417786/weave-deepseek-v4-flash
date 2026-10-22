"""Exercise actual executor branch selection with skewed per-warp clocks.

This host test checks CTA control-flow invariants, not CUDA execution or memory
safety. A barrier-containing task must never be entered by only part of a CTA.
"""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


def control_source():
    source=(ROOT/'experimental/native_tail/executor.cu').read_text()
    body=source[source.index('__global__ void resident_executor'):]
    declaration=re.search(r'__shared__ bool [^;]+;',body).group(0).replace('__shared__ ','')
    poll=body.split('  while(true){',1)[1].split('    __syncthreads();if(stop)break;',1)[0]
    guard=re.search(r'^    if\((.*)\)\{worked=try_finalize',body,re.M).group(1)
    return declaration,poll,guard


class CtaControlTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'),'host C++ compiler unavailable')
    def test_clock_skew_cannot_split_cta_at_delay_boundary(self):
        declaration,poll,guard=control_source()
        source='''#include <iostream>
using U64=unsigned long long;
struct {unsigned x;} threadIdx;
U64 now;U64 clock64(){return now;}
struct State {unsigned stop=0,finalized=0;};
struct Config {int chunks=2;};struct Invocation {};
unsigned dev_load(unsigned*p){return *p;}
bool protocol_aborted(const Config&,U64){return false;}
bool protocol_headers_ready(const Config&,Invocation){return true;}
bool communication_ready(const Config&,Invocation){return false;}
int sample(int threads,int diagnostic,int delay,U64 elapsed,bool comm_role,bool worked){
 State state;State*s=&state;Config cfg;Invocation inv;
 struct {bool serial=false;}plan;
 int mode[4]={1,0,diagnostic,delay};struct {int*mode;}a{mode};
 const U64 start=10000;
'''+declaration+'''
 // Simulate the leader publishing the real poll, then the existing CTA barrier.
 threadIdx.x=0;now=start+elapsed;
'''+poll+'''
 bool expected=!worked&&!comm_role&&(!diagnostic||elapsed>=U64(delay));
 int entered=0;
 for(int lane=0;lane<threads;++lane){
  threadIdx.x=lane;now=start+elapsed+8*(lane/32);
  bool decision=('''+guard+''');entered+=decision;
 }
 if(entered!=(expected?threads:0)){
  std::cerr<<"partial or premature CTA entry: threads="<<threads<<" entered="<<entered<<" delay="<<delay<<" elapsed="<<elapsed<<"\\n";
  return 1;
 }
 return 0;
}
int main(){int errors=0;
 for(int threads:{128,256}){
  for(U64 elapsed:{1999900ULL,1999999ULL,2000000ULL,2000001ULL})
   errors+=sample(threads,1,2000000,elapsed,false,false);
  errors+=sample(threads,0,2000000,0,false,false); // Formal mode remains immediate.
  errors+=sample(threads,1,0,0,false,false); // Uninjected diagnostic.
  errors+=sample(threads,1,2000000,2000001,true,false); // COMM keeps its own work path.
  errors+=sample(threads,1,2000000,2000001,false,true);
 }
 return errors?1:0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'control.cpp';binary=Path(directory)/'control';path.write_text(source)
            build=subprocess.run(['c++','-std=c++20',str(path),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)

    def test_delay_decision_is_published_before_cta_barrier(self):
        declaration,poll,guard=control_source()
        self.assertIn('compute_ready',declaration)
        self.assertRegex(poll,r'if\(threadIdx.x==0\)\{')
        self.assertIn('compute_ready=',poll)
        self.assertIn('clock64()',poll)
        self.assertIn('compute_ready',guard)
        self.assertNotIn('clock64',guard)
        self.assertNotIn('mode[3]',guard)
