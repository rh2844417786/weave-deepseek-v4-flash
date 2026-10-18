"""Strict offline GPQA Main contract and immutable prior compute evidence."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from .gpqa import load_prepared_records, ordered_indices
from .experiment_export import ensure_public

ROOT=Path(__file__).resolve().parents[2]
RECORD_SHA256='cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888'

@dataclass(frozen=True)
class GPQAWorkload:
    records: tuple
    identity: dict


def load_gpqa_workload(records_path, model_path=None):
    path=Path(records_path)
    audit=json.loads((ROOT/'data/gpqa/deepseek_length_audit.json').read_text())
    records,manifest=load_prepared_records(path)
    if set(manifest)!=set(audit):raise ValueError('exact prepared GPQA manifest fields required')
    if audit['record_sha256']!=RECORD_SHA256 or hashlib.sha256(path.read_bytes()).hexdigest()!=RECORD_SHA256:
        raise ValueError('pinned original GPQA Main records SHA256 required')
    for key in ('model_id','model_revision','pinned_model_metadata_sha256','gpqa_main_sha256','order_seed',
                'choice_seed','prompt_instruction','reasoning_mode','thinking_mode','reasoning_effort',
                'max_new_tokens','normal_eos_allowed','record_sha256','count','min','median','p95','max','at_least_8192'):
        if type(manifest.get(key)) is not type(audit[key]) or manifest.get(key)!=audit[key]:
            raise ValueError('GPQA manifest pin differs: '+key)
    if len(records)!=448 or [r['row_index'] for r in records]!=ordered_indices(448):
        raise ValueError('all 448 natural GPQA Main rows in seed42 order required')
    if any(not re.fullmatch('[0-9a-f]{64}',str(r.get('prompt_sha256',''))) for r in records):
        raise ValueError('prompt SHA256 commitments required')
    lengths=sorted(r['input_tokens'] for r in records)
    position=(len(lengths)-1)*.95;low=int(position)
    actual={'min':lengths[0],'median':statistics.median(lengths),'p95':lengths[low]+(lengths[low+1]-lengths[low])*(position-low),
            'max':lengths[-1],'at_least_8192':sum(n>=8192 for n in lengths)}
    if any(not math.isclose(actual[k],audit[k],abs_tol=1e-9) for k in actual):raise ValueError('natural GPQA lengths differ')
    observed=manifest.get('observed_model_metadata_sha256',{})
    pinned=audit['pinned_model_metadata_sha256']
    if set(observed)!=set(pinned) or any(not re.fullmatch('[0-9a-f]{64}',str(v)) for v in observed.values()):
        raise ValueError('observed model metadata required')
    if any(observed[k]!=v for k,v in pinned.items() if k!='config.json'):
        raise ValueError('tokenizer/encoder must match pinned preparation')
    if manifest.get('config_matches_pinned') is not (observed['config.json']==pinned['config.json']):
        raise ValueError('observed config policy inconsistent')
    if model_path is not None:
        for name,sha in observed.items():
            file=Path(model_path)/name
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=sha:
                raise ValueError('actual readonly model metadata differs: '+name)
    identity={**manifest,'ignore_eos':False,'temperature':0,'concurrency':32,'seconds':1200,
              'manifest_sha256':hashlib.sha256((path.parent/'manifest.json').read_bytes()).hexdigest(),
              'numerical_policy':'record_only','accuracy_evaluated':False}
    return GPQAWorkload(tuple(records),identity)


def _proof_allowlist():
    # Inventory of the accepted public schema, independent of supplied manifest.
    core={'runtime.json','checkpoint_identity.json'}|{f'{prefix}{r}.json' for prefix in ('rank','checkpoint-rank') for r in range(4)}
    from .hopper_serving_evidence import REPLAY_FILES
    names={'experiment.json','comparison.json','source-manifest.json','serving-source-manifest.json','replay-proof-check.json'}
    names|={'core/'+n for n in core}|{'replay-proof/'+n for n in REPLAY_FILES|{'manifest.json'}}
    names|={'online-gate/'+n for n in ('runtime.json','container_lifecycle.json','gate-check.json')}
    names|={f'online-gate/{prefix}-rank{r}.json' for prefix in ('hopper-serving','hopper-prefill') for r in range(4)}
    common={'group.json','launch.json','runtime.json','server-lifecycle.json','acceptance-lifecycle.json','measurement-lifecycle.json',
            'acceptance-summary.json','acceptance-check.json','client-summary.json','load-window.json','load-progress.json','scheduler-summary.json'}
    names|={f'{g}/{n}' for g in ('native','hopper_fp8','hopper_prefill') for n in common}
    names|={f'{g}/{n}' for g in ('hopper_fp8','hopper_prefill') for n in ('coverage-before.json','coverage-after.json','coverage-check.json')}
    return frozenset(names)

PREFILL_FILES=_proof_allowlist()

def prefill_bundle_bytes(path):
    root=Path(path)
    expected=PREFILL_FILES|{'manifest.json'}
    if root.is_symlink() or not root.is_dir():raise ValueError('prefill bundle directory required')
    entries=list(root.rglob('*'))
    actual={p.relative_to(root).as_posix() for p in entries if p.is_file() or p.is_symlink()}
    if actual!=expected or any(p.is_symlink() for p in entries):raise ValueError('complete original 98-file manifest inventory required')
    files={name:(root/name).read_bytes() for name in expected}
    manifest=json.loads(files['manifest.json'])
    ensure_public(manifest)
    if set(manifest)!={'schema_version','status','raw_content_exported','files'}:
        raise ValueError('exact public prefill manifest schema required')
    if (manifest.get('schema_version')!=1 or manifest.get('status')!='comparison_complete'
            or manifest.get('raw_content_exported') is not False or set(manifest.get('files',{}))!=PREFILL_FILES):
        raise ValueError('complete public prefill manifest required')
    for name in PREFILL_FILES:
        if hashlib.sha256(files[name]).hexdigest()!=manifest['files'][name]:raise ValueError('prefill file SHA differs: '+name)
        if name.endswith('.json'):
            value=json.loads(files[name]);ensure_public(value)
            if any(k in json.dumps(value) for k in ('"records":','"output_token_ids":','"input_token_ids":')):raise ValueError('private proof payload')
        elif any(line and not line.lstrip().startswith(('ptxas info','ptxas warning')) for line in files[name].decode().splitlines()):
            raise ValueError('compiler resource lines only')
    return files


def validate_prefill_bundle(path,model_path,image_id):
    from scripts.calibrate_weave import validate_core_evidence
    from .ops import kernel_source_sha256
    from .hopper_candidate import source_sha256
    from .hopper_decode import assembly_source_sha256
    from .hopper_prefill_candidate import source_sha256 as prefill_source
    from .hopper_prefill import runtime_source_sha256
    from .hopper_serving_evidence import validate_replay_bundle, validate_assembly_reports
    from .hopper_prefill_evidence import validate_prefill_reports
    from .replay_trace import tokenizer_manifest_sha256
    root=Path(path);files=prefill_bundle_bytes(root)
    read=lambda name:json.loads(files[name])
    source=kernel_source_sha256()
    core,checkpoint=validate_core_evidence(root/'core',Path(model_path),source,{'resolved_image_id':image_id})
    identity={'source_sha256':source,'candidate_source_sha256':source_sha256(),'assembly_source_sha256':assembly_source_sha256(),
              'prefill_source_sha256':prefill_source(),'prefill_runtime_source_sha256':runtime_source_sha256(),
              'checkpoint_content_sha256':core['checkpoint_content_sha256'],'image_id':image_id,
              'tokenizer_manifest_sha256':tokenizer_manifest_sha256(model_path)}
    report=read('experiment.json');comparison=read('comparison.json')
    if report.get('status')!='comparison_complete' or comparison.get('status')!='passed':raise ValueError('prior prefill comparison incomplete')
    for value in (report,comparison):
        if value.get('numerical_policy')!='record_only' or any(value.get('identity',{}).get(k)!=v for k,v in identity.items()):
            raise ValueError('prefill native/runtime/checkpoint/image identity differs')
    from .c32_comparison import digest
    if digest(read('source-manifest.json'))!=source:
        raise ValueError('prior native source manifest differs')
    if digest(read('serving-source-manifest.json'))!=report['identity'].get('serving_source_sha256'):
        raise ValueError('historical serving Python manifest differs')
    for name in ('native','hopper_fp8','hopper_prefill'):
        group=read(name+'/group.json')
        if (group.get('status')!='passed' or group.get('stage')!='complete'
                or group.get('identity')!=report['identity']):
            raise ValueError('prior serving group incomplete: '+name)
    runtime=read('online-gate/runtime.json');lifecycle=read('online-gate/container_lifecycle.json')
    if (runtime.get('runtime_status')!='passed' or runtime.get('resolved_image_id')!=image_id
            or runtime.get('kernel_source_sha256')!=source or lifecycle.get('exit_code')!=0
            or lifecycle.get('cleanup_status') not in ('not_required','removed')):
        raise ValueError('prefill gate runtime or cleanup failed')
    replay=validate_replay_bundle(root/'replay-proof',candidate_source_sha256=identity['candidate_source_sha256'],bf16_source_sha256=source,
        checkpoint_content_sha256=identity['checkpoint_content_sha256'],image_id=image_id,tokenizer_manifest_sha256=identity['tokenizer_manifest_sha256'])
    if replay!=read('replay-proof-check.json'):raise ValueError('prior replay check differs')
    assembly=validate_assembly_reports([read(f'online-gate/hopper-serving-rank{r}.json') for r in range(4)],
        candidate_source_sha256=identity['candidate_source_sha256'],bf16_source_sha256=source,assembly_source_sha256=identity['assembly_source_sha256'],image_id=image_id)
    assembly['prefill']=validate_prefill_reports([read(f'online-gate/hopper-prefill-rank{r}.json') for r in range(4)],
        **{k:identity[k] for k in ('prefill_source_sha256','prefill_runtime_source_sha256','candidate_source_sha256','checkpoint_content_sha256','image_id')},bf16_source_sha256=source)
    if assembly!=read('online-gate/gate-check.json'):raise ValueError('prior assembly/prefill check differs')
    return {'status':'passed','identity':identity,'core_gate':core,'manifest_sha256':hashlib.sha256(files['manifest.json']).hexdigest(),
            'manifest_file_count':len(PREFILL_FILES),'numerical_policy':'record_only',
            'source_policy':'historical Python preserved; current native, decode, prefill runtime, checkpoint, image and tokenizer match'}


def copy_verified_prefill_bundle(source,destination,proof=None):
    files=prefill_bundle_bytes(source)
    if proof is not None and hashlib.sha256(files['manifest.json']).hexdigest()!=proof['manifest_sha256']:raise ValueError('proof changed before copy')
    destination=Path(destination)
    if destination.exists():raise FileExistsError('fresh proof destination required')
    destination.mkdir(parents=True)
    for name,raw in files.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o664)
    if prefill_bundle_bytes(destination)!=files:raise ValueError('proof copy changed')
    return json.loads(files['manifest.json'])


def validate_client_summary(value):
    """Require the fixed whole-window streaming contract, including partial rows."""
    ensure_public(value)
    for key,expected in {'schema_version':1,'mode':'gpqa_measurement','status':'passed','concurrency':32,
                         'requested_seconds':1200,'window_seconds':1200,'peak_client_inflight':32,'numerical_policy':'record_only',
                         'accuracy_evaluated':False,'raw_content_exported':False,'reached_deadline':True,'cleanup_complete':True}.items():
        if value.get(key)!=expected:raise ValueError('GPQA client contract differs: '+key)
    if value.get('generation_settings')!={'max_new_tokens':32768,'temperature':0,'ignore_eos':False}:
        raise ValueError('natural GPQA EOS/output generation settings required')
    if value.get('workload',{}).get('record_sha256')!=RECORD_SHA256:raise ValueError('GPQA source identity differs')
    start,end=value.get('started_monotonic'),value.get('deadline_monotonic')
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in (start,end)) or not math.isclose(end-start,1200,abs_tol=1e-6):
        raise ValueError('fixed GPQA 1200-second window required')
    for name in ('requests_started','requests_completed','requests_unfinished','requests_failed','requests_unattempted',
                 'decode_tokens_in_window','output_tokens_in_window'):
        if type(value.get(name)) is not int or value[name]<0:raise ValueError('invalid GPQA count: '+name)
    if (value['requests_failed'] or value.get('error_codes') or value['requests_started']>448
            or value['requests_started']+value['requests_unattempted']!=448
            or value['requests_completed']+value['requests_unfinished']!=value['requests_started']):raise ValueError('incomplete/invalid GPQA request accounting')
    completed=value.get('completed_row_indices')
    if (not isinstance(completed,list) or any(type(x) is not int or not 0<=x<448 for x in completed)
            or completed!=sorted(set(completed)) or len(completed)!=value['requests_completed']):raise ValueError('completed original GPQA row IDs required')
    decode=value['decode_tokens_in_window'];output=value['output_tokens_in_window']
    if not 0<=output-decode<=value['requests_started'] or decode<=0:raise ValueError('all streaming tokens less each first output token required')
    if not math.isclose(value.get('decode_tokens_per_second',-1),decode/1200,rel_tol=1e-9):raise ValueError('whole-window token throughput required')
    if type(value.get('dataset_exhausted')) is not bool:raise ValueError('dataset exhaustion state required')
    if not {'last_completed_monotonic','dataset_exhausted_monotonic','idle_seconds'}<=set(value):
        raise ValueError('explicit completion, exhaustion and idle timing evidence required')
    last=value['last_completed_monotonic'];exhausted_at=value['dataset_exhausted_monotonic'];idle=value['idle_seconds']
    if value['requests_completed']:
        if type(last) not in (int,float) or not math.isfinite(last) or not start<=last<end:
            raise ValueError('actual final completed response timestamp required')
    elif last is not None:
        raise ValueError('no completed response may have a completion timestamp')
    exhausted=value['requests_completed']==448
    if value['dataset_exhausted']!=exhausted:
        raise ValueError('dataset exhaustion differs from completed row accounting')
    if type(idle) not in (int,float) or not math.isfinite(idle) or idle<0:
        raise ValueError('finite observed idle interval required')
    if exhausted:
        if (value['requests_unfinished'] or value['requests_unattempted'] or
                type(exhausted_at) not in (int,float) or exhausted_at!=last or
                not math.isclose(idle,end-last,rel_tol=1e-9,abs_tol=1e-6)):
            raise ValueError('exhaustion timing/counts inconsistent')
    elif exhausted_at is not None or idle!=0:
        raise ValueError('nonexhausted dataset cannot claim an idle exhaustion interval')
    if type(value.get('refill_exhausted')) is not bool or 'refill_exhausted_monotonic' not in value:
        raise ValueError('explicit refill exhaustion state and timing required')
    refill_at=value['refill_exhausted_monotonic']
    if value['refill_exhausted']:
        if (value['requests_started']!=448 or value['requests_unattempted'] or
                value['requests_completed']<417 or type(refill_at) not in (int,float) or
                not math.isfinite(refill_at) or not start<=refill_at<end):
            raise ValueError('refill exhaustion timing/counts inconsistent')
    elif refill_at is not None:
        raise ValueError('available refill cannot have an exhaustion timestamp')
    return value


def compare_groups(groups):
    if set(groups)!={'native','hopper_prefill'}:raise ValueError('exact native/hopper_prefill groups required')
    summaries={key:validate_client_summary(item.get('client_summary',item)) for key,item in groups.items()}
    if summaries['native']['workload']!=summaries['hopper_prefill']['workload']:
        raise ValueError('GPQA group workload identities differ')
    identities=[item.get('identity') for item in groups.values() if 'identity' in item]
    if identities and (len(identities)!=2 or identities[0]!=identities[1]):
        raise ValueError('GPQA group execution identities differ')
    exhausted=any(x['dataset_exhausted'] for x in summaries.values())
    refill_exhausted=any(x['refill_exhausted'] for x in summaries.values())
    suppressed=exhausted or refill_exhausted
    rates={key:item['decode_tokens_in_window']/1200 for key,item in summaries.items()}
    common=sorted(set(summaries['native']['completed_row_indices'])&set(summaries['hopper_prefill']['completed_row_indices']))
    return {'schema_version':1,'status':'passed','formal_gpqa':True,'accuracy_evaluated':False,
            'scope':'GPQA Main natural Think High 32 concurrent requests, 1200 seconds per group',
            'numerical_policy':'record_only','generation_comparison_policy':'record_only','raw_content_exported':False,
            'decode_tokens_per_second':rates,'common_completed_row_indices':common,'common_completed_count':len(common),
            'dataset_exhausted':exhausted,'refill_exhausted':refill_exhausted,'sustained_c32_ratio_available':not suppressed,
            'ratio_suppression_reason':('dataset_exhausted_before_deadline' if exhausted else
                                        'refill_exhausted_before_deadline' if refill_exhausted else None),
            'prefill_vs_native_ratio':None if suppressed else rates['hopper_prefill']/rates['native']}
