#!/usr/bin/env python3
"""Normalize GPUQ allocation for shell launchers, without Torch imports."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.weave.gpu_allocation import allocated_gpus


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container-visible',action='store_true')
    args=parser.parse_args()
    try: allocation=allocated_gpus()
    except ValueError as error: parser.error(str(error))
    print(allocation.container_visible if args.container_visible else allocation.csv)


if __name__=='__main__': main()
