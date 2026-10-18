#!/usr/bin/env python3
"""Collector (optional), unchanged Hopper assembly, then full prefill matrix."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--model-path',required=True)
    p.add_argument('--stall-diagnostic',action='store_true')
    a=p.parse_args(argv)
    scripts=(['check_stall_timeline.py'] if a.stall_diagnostic else [])+['check_hopper_serving.py','check_hopper_prefill.py']
    for script in scripts:
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',str(ROOT/'tests/gpu'/script),'--output-dir',a.output_dir]
        if script=='check_hopper_prefill.py':command+=['--model-path',a.model_path]
        subprocess.run(command,check=True)
    return 0
if __name__=='__main__':sys.exit(main())
