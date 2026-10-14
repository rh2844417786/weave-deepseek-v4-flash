#!/usr/bin/env python3
"""Run the installed SGLang benchmark, resetting probes at its first formal request."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.collect_execution_profile import reset_profiles
from src.weave.calibration import atomic_json


def install_window_hook(benchmark,marker,base):
    original=benchmark.ASYNC_REQUEST_FUNCS['sglang']
    started=False
    progress={'started':0,'succeeded':0,'failed':0,'cancelled':0,'completed_output_tokens':0,
              'scope':'formal requests only; incomplete streaming output is not counted'}
    def publish():atomic_json(marker.with_name('load-progress.json'),progress)

    async def request(request_func_input,pbar=None):
        nonlocal started
        # Pinned benchmark warms up with <=32 output tokens; formal requests use1024.
        if request_func_input.output_len==1024 and not started:
            reset_profiles(base)  # Benchmark warmups have finished; no formal request has started.
            atomic_json(marker,{'status':'load_started','started_monotonic':time.monotonic(),
                                'window_boundary':'before_first_formal_request_after_probe_reset'})
            started=True
        formal=request_func_input.output_len==1024
        if formal:progress['started']+=1;publish()
        try:
            output=await original(request_func_input=request_func_input,pbar=pbar)
        except BaseException:
            if formal:progress['cancelled']+=1;publish()
            raise
        if formal:
            progress['succeeded' if output.success else 'failed']+=1
            if output.success:progress['completed_output_tokens']+=int(output.output_len)
            publish()
        return output

    benchmark.ASYNC_REQUEST_FUNCS['sglang']=request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--marker',type=Path,required=True)
    parser.add_argument('--evidence-port-base',type=int,required=True)
    options,remaining=parser.parse_known_args()
    if remaining[:1]==['--']:remaining=remaining[1:]
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[key]='1'
    from sglang.benchmark import serving
    from sglang.benchmark.datasets import custom
    expected={serving:'73a7ce619e8d4c6539d89755f465beedfc41fff9',custom:'452c7db74c0f1aefd2792c6308f3b9f9a3f28e93'}
    for module,git_blob in expected.items():
        payload=Path(module.__file__).read_bytes()
        actual=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
        if actual!=git_blob:
            raise ValueError(f'installed {module.__name__} differs from audited SGLang commit2948168; export source for audit, do not install online')
    install_window_hook(serving,options.marker,options.evidence_port_base)
    sys.argv=['sglang.benchmark.serving']+remaining
    serving.cli_main()


if __name__=='__main__':main()
