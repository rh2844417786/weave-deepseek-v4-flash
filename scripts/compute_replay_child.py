#!/usr/bin/env python3
"""Run staged numerical smoke before the four-rank frozen-trace replay."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    subprocess.run([sys.executable,str(ROOT/'tests/gpu/check_staged.py'),'--output',
                    str(args.output_dir/'staged-synthetic.json')],check=True)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
             str(ROOT/'tests/gpu/replay_compute.py'),'--output-dir',str(args.output_dir)]+remaining
    os.execv(sys.executable,command)


if __name__=='__main__':main()
