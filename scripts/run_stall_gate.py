#!/usr/bin/env python3
"""Run collector smoke, then actual Hopper assembly in the same pinned container."""
import argparse
import subprocess
import sys

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    for script in ('check_stall_timeline.py','check_hopper_serving.py'):
        subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
            '/workspace/tests/gpu/'+script,'--output-dir',args.output_dir],check=True)
    return 0
if __name__=='__main__':sys.exit(main())
