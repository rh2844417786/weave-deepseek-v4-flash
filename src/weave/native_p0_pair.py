"""Record-only paired generation evidence for native SGLang and native P0."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from .graph_evidence import validate_graph_delta
from .native_p0_evidence import validate_launch_trace, validate_native_p0_smoke

P0_ENV={'WEAVE_NATIVE_P0','WEAVE_NATIVE_P0_STRIDE','WEAVE_NATIVE_P0_CAPACITY','WEAVE_NATIVE_P0_PRIVATE_DIR'}


def normalize_launch(argv):
    """Allow only declared P0 env additions and owned container/output names."""
    result=[];index=0
    while index<len(argv):
        value=argv[index]
        if value=='-e' and index+1<len(argv) and argv[index+1].split('=',1)[0] in P0_ENV:
            index+=2;continue
        if value in ('--name','--output-dir'):
            if index+1>=len(argv):raise ValueError('malformed launch argv')
            result.extend([value,'<owned-name>' if value=='--name' else '<owned-output>']);index+=2;continue
        result.append(value);index+=1
    return result


def _hex(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _validate_client(report):
    if (report.get('status')!='passed' or report.get('request_count')!=32 or report.get('requests_completed')!=32
            or report.get('cleanup_complete')is not True or report.get('error_codes')
            or report.get('numerical_policy')!='record_only'):
        raise ValueError('32 complete valid responses and cleanup required')
    if report.get('generation_settings')!=dict(max_new_tokens=32768,temperature=0,ignore_eos=False):
        raise ValueError('paired generation settings differ from natural Think High contract')
    if report.get('warmup',{}).get('requests_completed')!=32 or report.get('post_warmup_flush',{}).get('status')!='flushed':
        raise ValueError('same warmup and idle cache flush required')
    selection=report.get('selection',{});expected=selection.get('requests',[]);rows=report.get('rows',[])
    if len(rows)!=32 or len(expected)!=32 or len({v.get('original_row_index') for v in expected})!=32:
        raise ValueError('complete unique paired cohort required')
    for row,identity in zip(rows,expected):
        if any(row.get(k)!=identity.get(k) for k in ('original_row_index','prompt_sha256','input_tokens','request_payload_sha256')):
            raise ValueError('paired request identity differs')
        if not all(_hex(row.get(k)) for k in ('prompt_sha256','request_payload_sha256','output_text_sha256')):
            raise ValueError('valid input/output commitments required')
        count=row.get('output_tokens');reason=row.get('finish_reason')
        if row.get('status')!='completed' or type(count)is not int or not 0<count<=32768 or not (reason in ('stop','eos') or reason=='length' and count==32768):
            raise ValueError('invalid complete generation response')
        if type(row.get('output_text_bytes'))is not int or row['output_text_bytes']<0:
            raise ValueError('invalid output byte count')
        elapsed=row.get('elapsed_seconds')
        if type(elapsed)not in (int,float) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('invalid response elapsed time')
        points=row.get('prefix_digests',[]);counts=[p.get('completion_tokens') for p in points]
        if any(type(n)is not int or n<1 or n>count for n in counts) or counts!=sorted(set(counts)) or any(not _hex(p.get('sha256')) for p in points):
            raise ValueError('invalid observed prefix commitments')
    return rows


def compare_clients(native,p0):
    a,b=_validate_client(native),_validate_client(p0)
    for key in ('selection','source_workload_identity','generation_settings'):
        if native.get(key)!=p0.get(key):raise ValueError('paired request/source configuration differs: '+key)
    rows=[]
    for x,y in zip(a,b):
        pa={p['completion_tokens']:p['sha256'] for p in x['prefix_digests']};pb={p['completion_tokens']:p['sha256'] for p in y['prefix_digests']}
        prefixes=[]
        for count in sorted(set(pa)|set(pb)):
            status=('unavailable_in_native' if count not in pa else 'unavailable_in_p0' if count not in pb else 'matched' if pa[count]==pb[count] else 'mismatched')
            prefixes.append(dict(completion_tokens=count,status=status))
        rows.append(dict(original_row_index=x['original_row_index'],request_payload_sha256=x['request_payload_sha256'],
            native_output_tokens=x['output_tokens'],p0_output_tokens=y['output_tokens'],length_delta=y['output_tokens']-x['output_tokens'],
            p0_to_native_length_ratio=y['output_tokens']/x['output_tokens'],native_finish_reason=x['finish_reason'],p0_finish_reason=y['finish_reason'],
            finish_reason_matched=x['finish_reason']==y['finish_reason'],output_text_digest_matched=x['output_text_sha256']==y['output_text_sha256'],
            native_output_text_sha256=x['output_text_sha256'],p0_output_text_sha256=y['output_text_sha256'],
            prefix_comparisons=prefixes,first_observed_divergent_prefix_tokens=next((p['completion_tokens'] for p in prefixes if p['status']=='mismatched'),None)))
    def lengths(values):return dict(count=len(values),min=min(values),max=max(values),mean=statistics.mean(values),median=statistics.median(values))
    matched=sum(r['output_text_digest_matched'] for r in rows)
    prefix_summary={}
    for row in rows:
        for item in row['prefix_comparisons']:
            key=str(item['completion_tokens']);counts=prefix_summary.setdefault(key,{})
            counts[item['status']]=counts.get(item['status'],0)+1
    return dict(schema_version=1,status='paired_check_complete',scope='same_32_natural_GPQA_requests_native_vs_P0',
        numerical_policy='record_only',generation_comparison_policy='record_only',formal_performance=False,accuracy_evaluated=False,
        selection=native['selection'],request_count=32,output_text_digest_matched=matched,output_text_digest_mismatched=32-matched,
        length_matched=sum(r['length_delta']==0 for r in rows),finish_reason_matched=sum(r['finish_reason_matched'] for r in rows),
        lengths={'native':lengths([r['output_tokens'] for r in a]),'p0':lengths([r['output_tokens'] for r in b])},
        finish_reasons={'native':dict(Counter(r['finish_reason'] for r in a)),'p0':dict(Counter(r['finish_reason'] for r in b))},
        rows=rows,prefix_summary=prefix_summary,interpretation=[
            'All output length, finish-reason and digest differences are record-only; no accuracy or agreement cutoff.',
            'Prefix digests refer to accumulated incremental UTF-8 text at observed token counts, not token-ID equality.',
            'First observed divergent prefix is not necessarily the first divergent token.',
            'Same request cohort does not guarantee identical arrival or decode batch trajectories.',
            'A reproduced difference associates with the P0 intervention; it does not establish a specific event, buffer or sampler root cause.'])


def paired_graph_check(before,after,*,p0_enabled,identity):
    result=validate_graph_delta(before,after,expected_observer_source=identity['observer_source_sha256'],expected_sources=identity['graph_sources'])
    ranks={}
    for old,new in zip(sorted(before,key=lambda r:r['rank']),sorted(after,key=lambda r:r['rank'])):
        if not p0_enabled:
            if old.get('stage_profile') is not None or new.get('stage_profile') is not None:raise ValueError('uninstrumented arm has a stage profiler')
            continue
        for outer in (old,new):
            p=outer.get('stage_profile',{})
            expected=dict(kind='native_p0',rank=outer['rank'],profile_source_sha256=identity['native_p0_source_sha256'],external_events=True,numerical_policy='record_only')
            if any(p.get(k)!=v for k,v in expected.items()) or p.get('configuration')!=dict(stride=128,capacity=64,layers=43,captured_batch_size=32):
                raise ValueError('actual P0 profile identity/configuration differs')
            seen=p.get('b32_seen');sampled=p.get('sampled_total')
            if (type(seen)is not int or type(sampled)is not int or seen<0 or sampled!=seen//128
                    or seen!=outer['raw_batch_histogram'].get('32',0) or p.get('aborted_samples')!=0):
                raise ValueError('actual P0 sampling counters differ')
            metadata=p.get('layer_metadata',{})
            if set(metadata)!={str(i) for i in range(43)}:raise ValueError('43 captured P0 layers required')
            for layer in metadata.values():
                if (not layer.get('captured_path') or not {'moe','routed','gate_up','down'}.issubset(layer.get('captured_spans',[]))
                        or [r.get('role') for r in layer.get('marlin_calls',[])]!=['gate_up','down']):
                    raise ValueError('actual P0 capture hooks missing')
            if sampled:validate_launch_trace(p.get('launch_trace') or {},metadata,outer['rank'])
        delta=new['stage_profile']['sampled_total']-old['stage_profile']['sampled_total']
        if delta<0:raise ValueError('P0 sample counter decreased')
        ranks[str(new['rank'])]=dict(runtime_sample_count=delta,b32_replays=new['stage_profile']['b32_seen']-old['stage_profile']['b32_seen'])
    counts={r['runtime_sample_count'] for r in ranks.values()}
    if len(counts)>1:raise ValueError('cross-rank P0 runtime sample counts differ')
    sampled=bool(counts and next(iter(counts))>0)
    result.update(p0_enabled=p0_enabled,p0_runtime_sampled=sampled,p0_sampling_by_rank=ranks,
        p0_execution_scope=('captured_instrumentation_and_runtime_sampling' if sampled else 'captured_instrumentation_without_runtime_sampling') if p0_enabled else 'native_without_P0',
        numerical_policy='record_only')
    return result


def validate_prior_smoke(root,identity):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());files={}
    names=['native-p0-proof-check.json','native-p0-gate/runtime.json']+[f'native-p0-gate/{prefix}-rank{rank}.json' for prefix in ('native-p0-event','graph-cleanup') for rank in range(4)]
    for name in names:
        file=root/name
        if file.is_symlink():raise ValueError('prior P0 proof symlink forbidden')
        raw=file.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if manifest['files'].get(name)!=sha:raise ValueError('prior P0 smoke manifest differs')
        files[name]=sha
    proof=validate_native_p0_smoke(root/'native-p0-gate',identity)
    if proof!=json.loads((root/'native-p0-proof-check.json').read_text()):raise ValueError('prior P0 proof revalidation differs')
    runtime=json.loads((root/'native-p0-gate/runtime.json').read_text())
    if runtime.get('runtime_status')!='passed' or runtime.get('resolved_image_id')!=identity['image_id']:raise ValueError('prior smoke runtime/image differs')
    return dict(status='passed',scope='reuse_unchanged_source_and_image_P0_event_smoke',files=files,proof=proof)
