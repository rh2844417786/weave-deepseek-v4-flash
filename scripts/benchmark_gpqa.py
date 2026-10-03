#!/usr/bin/env python3
"""Formal GPQA group or common-completed comparison; uses offline prepared IDs."""
import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weave.performance import benchmark_group, compare_groups, _atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    group = commands.add_parser('group')
    group.add_argument('--url', required=True)
    group.add_argument('--records', type=Path, default=ROOT/'data/gpqa/prepared/deepseek-v4-flash/main.jsonl')
    group.add_argument('--output-dir', type=Path, required=True)
    group.add_argument('--raw-root',type=Path,help='absolute working disk for private token artifacts')
    group.add_argument('--ready-monotonic', type=float, required=True,
                       help='parent-observed /health success monotonic timestamp')
    compare = commands.add_parser('compare')
    compare.add_argument('--native', type=Path, required=True)
    compare.add_argument('--weave', type=Path, required=True)
    compare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'compare':
        result = compare_groups(json.loads(args.native.read_text()), json.loads(args.weave.read_text()))
        _atomic_json(args.output, result)
    else:
        def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        result = benchmark_group(args.url, args.records, args.output_dir, ready_monotonic=args.ready_monotonic,raw_root=args.raw_root)
    print(json.dumps({key: result.get(key) for key in ('status', 'completed_count', 'relative_decode_speed')}))
    return 0 if result['status'] in ('complete', 'deadline', 'incomplete', 'compared', 'no_common_completed') else 1


if __name__ == '__main__': raise SystemExit(main())
