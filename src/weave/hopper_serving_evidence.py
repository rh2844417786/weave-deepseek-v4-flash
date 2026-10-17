"""Public provenance and record-only comparisons for Hopper C32 serving.

Historical replay Python identities are retained, while the unchanged native
compute sources and current model/image/tokenizer must match the replay.
"""
import hashlib
import json
from pathlib import Path
import re

from .c32_comparison import digest
from .experiment_export import ensure_public
from .hopper_evidence import aggregate_rank_reports, validate_rank_report, validate_identity, public_resources
from .replay_trace import aggregate_manifests

REPLAY_FILES=frozenset(['experiment.json','identity.json','runtime.json','container_lifecycle.json',
    'summary.json','oracle-unit-check.json','candidate-source-manifest.json','bf16-source-manifest.json',
    'controller-source-manifest.json','capture-manifest.json','compilation-resources.txt',
    'core/runtime.json','core/checkpoint_identity.json']+
    [f'{prefix}{rank}.json' for prefix in ('hopper-rank','hopper-progress-rank','core/rank','core/checkpoint-rank') for rank in range(4)])
ASSEMBLY_CASES=(('b1_spread',1),('b7_empty_rank',7),('b17_spread',17),('b32_spread',32),('b7_shrink',7),('b1_shrink',1))


def replay_bundle_bytes(path):
    """Read exactly the original allowlisted bundle; reject symlinks and extras."""
    root=Path(path)
    if root.is_symlink():raise ValueError('replay bundle must not be a symlink')
    expected=REPLAY_FILES|{'manifest.json'}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual!=expected or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('complete immutable 29-file replay bundle required')
    files={name:(root/name).read_bytes() for name in expected}
    manifest=json.loads(files['manifest.json'])
    if (manifest.get('schema_version')!=2 or manifest.get('status')!='compute_candidate_complete'
            or manifest.get('numerical_policy')!='record_only' or manifest.get('raw_content_exported') is not False
            or set(manifest.get('files',{}))!=REPLAY_FILES):
        raise ValueError('complete schema2 record-only replay manifest required')
    for name in REPLAY_FILES:
        if hashlib.sha256(files[name]).hexdigest()!=manifest['files'][name]:
            raise ValueError('replay manifest file SHA256 differs: '+name)
        if name.endswith('.json'):ensure_public(json.loads(files[name]))
    return files


def validate_replay_bundle(path, *, candidate_source_sha256,bf16_source_sha256,
                           checkpoint_content_sha256,image_id,tokenizer_manifest_sha256):
    files=replay_bundle_bytes(path)
    values={name:json.loads(raw) for name,raw in files.items() if name.endswith('.json')}
    identity=validate_identity(values['identity.json'])
    for key,current in (('candidate_source_sha256',candidate_source_sha256),('bf16_source_sha256',bf16_source_sha256),
                        ('checkpoint_content_sha256',checkpoint_content_sha256),('image_id',image_id)):
        if identity[key]!=current:raise ValueError('replay differs from current '+key)
    if identity['capture_identity']['tokenizer_manifest_sha256']!=tokenizer_manifest_sha256:
        raise ValueError('replay tokenizer identity differs')
    report=values['experiment.json']
    if (report.get('status')!='compute_candidate_complete' or report.get('schema_version')!=2
            or report.get('numerical_policy')!='record_only' or report.get('identity')!=identity
            or report.get('child_exit_code')!=0):raise ValueError('replay experiment did not complete')
    for name,key in (('candidate-source-manifest.json','candidate_source_sha256'),('bf16-source-manifest.json','bf16_source_sha256')):
        if digest(values[name])!=identity[key]:raise ValueError('replay native source manifest differs')
    if digest(values['controller-source-manifest.json'])!=report.get('controller_source_sha256'):
        raise ValueError('historical controller source manifest differs')
    capture=values['capture-manifest.json']
    if (hashlib.sha256(files['capture-manifest.json']).hexdigest()!=identity['capture_manifest_sha256']
            or capture.get('identity')!=identity['capture_identity']
            or aggregate_manifests(capture['ranks'],capture['identity'])!=capture):
        raise ValueError('replay capture provenance differs')
    rows=[values[f'hopper-rank{rank}.json'] for rank in range(4)]
    for rank,row in enumerate(rows):
        validate_rank_report(row,identity,require_complete=True)
        if row['rank']!=rank:raise ValueError('replay rank filename identity differs')
    summary=aggregate_rank_reports(rows,identity)
    if summary!=values['summary.json'] or summary['status']!='complete' or summary['point_count']!=1376:
        raise ValueError('stored replay summary differs from complete 1376-point evidence')
    runtime=values['runtime.json'];lifecycle=values['container_lifecycle.json'];oracle=values['oracle-unit-check.json']
    if (runtime.get('runtime_status')!='passed' or runtime.get('resolved_image_id')!=image_id
            or runtime.get('kernel_source_sha256')!=bf16_source_sha256
            or lifecycle.get('exit_code')!=0 or lifecycle.get('cleanup_status') not in ('not_required','removed')
            or oracle.get('status')!='passed' or oracle.get('errors')!=0 or oracle.get('failures')!=0):
        raise ValueError('replay runtime, cleanup or oracle unit evidence failed')
    return {'schema_version':1,'status':'passed','numerical_policy':'record_only','point_count':1376,
        'ranks':4,'points_per_rank':344,'manifest_file_count':29,
        'manifest_sha256':hashlib.sha256(files['manifest.json']).hexdigest(),
        'identity':identity,'replay_source_sha256':identity['replay_source_sha256'],
        'controller_source_sha256':report['controller_source_sha256'],
        'source_policy':'historical Python preserved; native compute and model/image/tokenizer match current'}


def validate_generation_proof(summary,proof, *, private_sha256,dataset_sha256):
    """Validate client response provenance without requiring matching content."""
    settings={'max_new_tokens':8,'temperature':0,'ignore_eos':True}
    if (summary.get('status')!='passed' or summary.get('dataset_sha256')!=dataset_sha256
            or summary.get('private_artifact_sha256')!=private_sha256
            or summary.get('requests_completed')!=32 or summary.get('generated_tokens')!=256
            or summary.get('concurrency')!=32 or summary.get('error_codes')!=[] or summary.get('abort_failures')!=0
            or summary.get('acceptance_order_sha256')!=digest(list(range(32)))
            or summary.get('generation_settings')!=settings or proof.get('generation_settings')!=settings
            or not isinstance(summary.get('request_order_sha256'),str)
            or not re.fullmatch('[0-9a-f]{64}',summary['request_order_sha256'])
            or any(proof.get(k)!=summary.get(k) for k in ('dataset_sha256','request_order_sha256'))):
        raise ValueError('generation response identity incomplete')
    record_generation_comparison(proof,proof)
    expected=digest([[row['request_index'],row['output_ids']] for row in sorted(proof['records'],key=lambda row:row['request_index'])])
    if summary.get('generated_ids_sha256')!=expected:raise ValueError('generation sequence digest differs')
    return True


def record_generation_comparison(reference,candidate):
    def normalize(value):
        rows=value.get('records',[])
        if len(rows)!=32:raise ValueError('generation proof requires 32 actual request sequences')
        result={}
        for row in rows:
            index=row.get('request_index');ids=row.get('output_ids')
            if (type(index) is not int or index not in range(32) or index in result
                    or row.get('request_id')!=f'weave-c32-acceptance-{index:04d}'
                    or not isinstance(ids,list) or len(ids)!=8 or any(type(x) is not int or x<0 for x in ids)
                    or row.get('output_tokens')!=8 or row.get('output_ids_sha256')!=digest(ids)):
                raise ValueError('invalid actual generation response identities or IDs')
            result[index]=ids
        return result
    before,after=normalize(reference),normalize(candidate)
    matched=sum(before[i]==after[i] for i in range(32))
    tokens=sum(a==b for i in range(32) for a,b in zip(before[i],after[i]))
    return {'schema_version':1,'status':'recorded','numerical_policy':'record_only',
        'generation_comparison_policy':'record_only','valid_requests':32,'generated_tokens':256,
        'matched_requests':matched,'mismatched_requests':32-matched,
        'matched_generated_tokens':tokens,'mismatched_generated_tokens':256-tokens,
        'reference_sequence_sha256':digest(before),'candidate_sequence_sha256':digest(after),
        'logprob_policy':'record_only'}


def validate_assembly_reports(rows, *, candidate_source_sha256,bf16_source_sha256,assembly_source_sha256,image_id):
    if len(rows)!=4 or {row.get('rank') for row in rows}!={0,1,2,3}:raise ValueError('four assembly rank reports required')
    hashes={'candidate_source_sha256':candidate_source_sha256,'bf16_source_sha256':bf16_source_sha256,
            'assembly_source_sha256':assembly_source_sha256}
    for row in rows:
        ensure_public(row)
        fields={'schema_version','numerical_policy','status','rank','world_size','cases','case_count','stop_reason',
            'runtime','candidate_source_sha256','bf16_source_sha256','assembly_source_sha256','source_hashes_after',
            'ordered_finalizer','additional_slot_collectives','stable_addresses','binding_lifetime',
            'empty_local_rank_exercised','empty_origin_rank_exercised','resources'}
        if set(row)!=fields or row.get('stop_reason')!='none' or public_resources(row['resources'])!=row['resources']:
            raise ValueError('assembly report public schema differs')
        if (row.get('schema_version')!=1 or row.get('status')!='passed' or row.get('numerical_policy')!='record_only'
                or row.get('world_size')!=4 or row.get('source_hashes_after')!=hashes
                or any(row.get(k)!=v for k,v in hashes.items())):raise ValueError('assembly source/status identity differs')
        if (row.get('case_count')!=6 or row.get('additional_slot_collectives')!=6
                or any(row.get(k) is not True for k in ('stable_addresses','binding_lifetime','empty_local_rank_exercised','empty_origin_rank_exercised'))):
            raise ValueError('complete assembly buffer/empty-rank/collective evidence required')
        cases=row.get('cases',[])
        if [(x.get('case'),x.get('batch')) for x in cases]!=list(ASSEMBLY_CASES):raise ValueError('assembly batch coverage incomplete')
        for case in cases:
            if set(case)!={'case','batch','additional_slot_collectives','assembly_exact','finite','stable_addresses','binding_alive','local_slots','origin_rows'}:
                raise ValueError('assembly case public schema differs')
            if (case.get('additional_slot_collectives')!=1
                    or any(case.get(k) is not True for k in ('assembly_exact','finite','stable_addresses','binding_alive'))
                    or any(type(case.get(k)) is not int or case[k]<0 for k in ('local_slots','origin_rows'))):
                raise ValueError('assembly execution invariant failed')
        if row.get('ordered_finalizer')!={'status':'passed','case_count':4,'batches':[1,7,17,32],'exact':True}:
            raise ValueError('ordered finalizer deterministic fixture failed')
        runtime=row.get('runtime',{})
        if (runtime.get('image_id')!=image_id or runtime.get('capability')!=[9,0]
                or runtime.get('device_count')!=4 or 'H100' not in runtime.get('device_name','')):
            raise ValueError('assembly runtime identity differs')
    return {'schema_version':1,'status':'passed','numerical_policy':'record_only','ranks':4,
        'cases_per_rank':6,'ordered_finalizer_cases_per_rank':4,**hashes,'image_id':image_id}
