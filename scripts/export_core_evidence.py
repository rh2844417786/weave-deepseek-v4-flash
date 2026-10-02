#!/usr/bin/env python3
"""Export an existing core run into a tracked report directory, without rerunning."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.weave.service_acceptance import validate_checkpoint_gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--include-build-log',action='store_true')
    args=parser.parse_args()
    source=args.run_dir.resolve()
    output=(args.output_dir or ROOT/'reports/server'/source.name).resolve()
    if source==output:
        parser.error('output directory must differ from the original run directory')
    runtime_path=source/'runtime.json'
    runtime=json.loads(runtime_path.read_text())
    if runtime.get('validation_stage')!='core':
        parser.error('this exporter accepts core functional runs only')
    names=['runtime.json']+[f'rank{r}.json' for r in range(4)]+[
        'exit_code.txt','image_id.txt','allocation.txt','container_lifecycle.json']
    if runtime.get('checkpoint_validation_status') is not None:
        names += ['checkpoint_identity.json']+[f'checkpoint-rank{r}.json' for r in range(4)]
    if args.include_build_log:
        names.append('build-and-test.log')
    data,parsed={},{}
    for name in names:
        path=source/name
        if not path.exists(): continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size>10*1024*1024:
            parser.error(f'unsupported evidence file: {path}')
        data[name]=path.read_bytes()
        if name.endswith('.json'): parsed[name]=json.loads(data[name])
    missing=[r for r in range(4) if f'rank{r}.json' not in parsed]
    passed=(not missing and runtime.get('runtime_status')=='passed'
            and runtime.get('core_validation_status')=='passed' and runtime.get('child_exit_code')==0
            and all(parsed[f'rank{r}.json'].get('status')=='functional_checks_passed'
                    and parsed[f'rank{r}.json'].get('core_graph_replay')=='passed' for r in range(4)))
    status='passed' if passed else 'incomplete' if missing else 'not_passed'
    if 'exit_code.txt' in data and data['exit_code.txt'].decode().strip()!='0': status='not_passed'
    if (runtime.get('resolved_image_id') and 'image_id.txt' in data
            and runtime['resolved_image_id']!=data['image_id.txt'].decode().strip()): status='inconsistent'
    if output.exists():
        stale=[name for name in names if (output/name).exists() and name not in data]
        if stale: parser.error('output contains older evidence; choose a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    for name,content in data.items(): (output/name).write_bytes(content)
    manifest={'exported_utc':datetime.now(timezone.utc).isoformat(),'source_run_dir':str(source),
              'scope':'small_synthetic_core_checks','reported_core_status':status,
              'missing_rank_reports':missing,'full_model':'not_verified_by_this_export',
              'performance':'not_verified_by_this_export',
              'sha256':{name:hashlib.sha256(content).hexdigest() for name,content in data.items()}}
    if runtime.get('checkpoint_validation_status') is not None:
        manifest['scope']='small_core_and_original_checkpoint_checks'
        try:
            ranks=[parsed[f'checkpoint-rank{r}.json'] for r in range(4)]
            validate_checkpoint_gate(runtime,ranks,source_sha256=runtime['kernel_source_sha256'],
                                     index_sha256=ranks[0]['checkpoint_index_sha256'],
                                     checkpoint_sha256=parsed['checkpoint_identity.json']['sha256'])
            manifest['reported_checkpoint_status']='passed'
        except (ValueError,KeyError) as exc:
            manifest['reported_checkpoint_status']='not_passed'
            manifest['checkpoint_validation_error']=str(exc)
    (output/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'exported_to':str(output),'reported_core_status':status,'files':list(data)}))
    return 0


if __name__=='__main__': raise SystemExit(main())
