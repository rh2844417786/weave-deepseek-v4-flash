"""Archive explicit aggregate/provenance artifacts, never raw GPQA payloads."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

UNSAFE={'input_ids','input_token_ids','output_ids','text','prompt','response','request',
        'first_token_logprobs','embedding','hidden_states','token_id','token_ids',
        'weave_token_id','native_only_token_ids','weave_only_token_ids','common_logprob_deltas'}


def aggregate_native_comparison(diagnostic):
    if not isinstance(diagnostic,dict): return None
    fields=('status','used_for_acceptance','scope','first_token_matches',
            'first_token_logprob_delta','top_logprob_support_matches',
            'max_abs_common_logprob_delta','max_abs_logprob_delta','matched_generated_tokens')
    result={key:diagnostic[key] for key in fields if key in diagnostic}
    for source,target in (('native_only_token_ids','native_only_count'),
                          ('weave_only_token_ids','weave_only_count'),
                          ('common_logprob_deltas','common_count')):
        if source in diagnostic: result[target]=len(diagnostic[source])
    return result


def ensure_public(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in UNSAFE: raise ValueError(f'decodable benchmark field cannot be exported: {key}')
            ensure_public(item)
    elif isinstance(value,list):
        for item in value: ensure_public(item)


def export_run(run_dir,destination):
    run_dir,destination=Path(run_dir),Path(destination)
    if destination.exists(): raise FileExistsError('fresh export destination required')
    report=json.loads((run_dir/'experiment.json').read_text())
    files={'experiment.json':(run_dir/'experiment.json').read_bytes()}
    def add(source,name):
        source=Path(source)
        if source.is_file(): files[name]=source.read_bytes()
    add(run_dir/'comparison.json','comparison.json')
    for group in ('native','weave'): add(run_dir/group/'summary.json',f'{group}/summary.json')
    if report.get('core_run_dir'):
        core=Path(report['core_run_dir'])
        for name in ('runtime.json','checkpoint_identity.json','container_lifecycle.json'):
            add(core/name,'core/'+name)
        for rank in range(4):
            for prefix in ('rank','checkpoint-rank'):
                add(core/f'{prefix}{rank}.json',f'core/{prefix}{rank}.json')
    if report.get('calibration_dir'):
        cal=Path(report['calibration_dir'])
        for rank in range(4):
            for prefix in ('profile-bank-rank','timings-rank','calibration-rank','progress-rank','diagnostic-rank'):
                add(cal/f'{prefix}{rank}.json',f'calibration/{prefix}{rank}.json')
        for file in cal.glob('failure-rank*.json'): add(file,'calibration/'+file.name)
    if report.get('model_acceptance_dir'):
        model=Path(report['model_acceptance_dir'])
        for label in ('weave_eager','weave_graph'):
            file=model/f'{label}-request.json'
            if file.exists():
                request=json.loads(file.read_text())
                selected={k:request.get(k) for k in ('status','input_sha256','row_index',
                    'coverage_before','coverage_after','coverage_check')}
                selected['native_comparison']=aggregate_native_comparison(request.get('native_comparison'))
                files[f'model/{label}-proof.json']=(json.dumps(selected,indent=2,allow_nan=False)+'\n').encode()
    # Validate all documents before creating the publicly tracked destination.
    for payload in files.values(): ensure_public(json.loads(payload))
    destination.mkdir(parents=True)
    manifest={'schema_version':1,'exported_utc':datetime.now(timezone.utc).isoformat(),
              'source_directory':str(run_dir),'experiment_status':report['status'],
              'performance_status':report.get('performance'),'raw_content_exported':False,'files':{}}
    for name,payload in files.items():
        output=destination/name;output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(payload)
        manifest['files'][name]=hashlib.sha256(payload).hexdigest()
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
