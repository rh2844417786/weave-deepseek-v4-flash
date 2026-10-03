#!/usr/bin/env python3
"""Export one actual completed/partial run's public-safe metrics and GPU proof."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.weave.experiment_export import export_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    destination=args.output_dir or ROOT/'reports/server'/args.run_dir.name
    manifest=export_run(args.run_dir,destination)
    print(json.dumps({'status':'exported','experiment_status':manifest['experiment_status'],
                      'output_directory':str(destination),'raw_content_exported':False}))


if __name__=='__main__': main()
