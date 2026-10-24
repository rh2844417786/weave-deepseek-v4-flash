#!/usr/bin/env python3
"""Summarize existing public native-tail calibration without GPU work or downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.weave.calibration import atomic_json
from src.weave.native_tail_timing import summarize_profiles

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    manifest=json.loads((args.report_dir/'manifest.json').read_text())
    profiles={}
    for rank in range(4):
        name=f'calibration/native-tail-rank{rank}.json'
        raw=(args.report_dir/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['files'].get(name):raise ValueError('calibration manifest SHA256 mismatch')
        profiles[str(rank)]=json.loads(raw)
    result=summarize_profiles(profiles)
    result['input_manifest_sha256']=hashlib.sha256((args.report_dir/'manifest.json').read_bytes()).hexdigest()
    atomic_json(args.output,result)
    print(json.dumps(dict(output=str(args.output),rank_group_count=len(result['rows']),time_unit='us')))
    return 0
if __name__=='__main__':raise SystemExit(main())
