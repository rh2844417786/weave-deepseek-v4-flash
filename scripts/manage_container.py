#!/usr/bin/env python3
"""Signal-aware wrapper for a uniquely named Docker container in a GPUQ job."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.weave.container_lifecycle import run_container


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--log',type=Path)
    parser.add_argument('--exit-code-file',type=Path)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if args.log:
        args.log.parent.mkdir(parents=True,exist_ok=True)
        with args.log.open('w') as stream:
            result=run_container(command,args.name,args.evidence,stdout=stream,exit_code_path=args.exit_code_file)
    else:
        result=run_container(command,args.name,args.evidence,exit_code_path=args.exit_code_file)
    print(json.dumps(result),flush=True)
    return result['exit_code']


if __name__=='__main__': sys.exit(main())
