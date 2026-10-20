"""Validate native-only P0 evidence without depending on candidate-compute gates."""
import hashlib
import json
import math
import re
from pathlib import Path
from .checkpoint_identity import checkpoint_identity, checkpoint_stats, sha256_file
from .graph_cleanup import CLEANUP_PHASES
from .graph_stage_evidence import stats
from .native_p0 import derive_layer_metrics


def native_checkpoint_proof(baseline, model):
    baseline, model = Path(baseline), Path(model)
    name = 'prefill-proof/core/checkpoint_identity.json'
    path = baseline / name
    manifest = json.loads((baseline / 'manifest.json').read_text())
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != manifest['files'].get(name):
        raise ValueError('native baseline checkpoint attestation digest differs')
    old = json.loads(raw)
    if hashlib.sha256(json.dumps(old['files_sha256'], sort_keys=True, separators=(',', ':')).encode()).hexdigest() != old['sha256']:
        raise ValueError('checkpoint content manifest digest differs')
    current = checkpoint_stats(model)
    if (current == old['verified_stats'] and set(current) == set(old['files_sha256'])
            and all(sha256_file(model / key) == old['files_sha256'][key]
                    for key in ('config.json', 'model.safetensors.index.json'))):
        checked, policy = old, 'reuse_content_attestation_with_unchanged_file_stats_and_config_index_hashes'
    else:
        checked, policy = checkpoint_identity(model), 'rehash_all_local_checkpoint_files'
        if checked['sha256'] != old['sha256']:
            raise ValueError('local model differs from native baseline checkpoint')
    return dict(status='passed', checkpoint_content_sha256=checked['sha256'],
                verified_stats=checked['verified_stats'], checkpoint_verification=policy,
                baseline_attestation_sha256=hashlib.sha256(raw).hexdigest(),
                candidate_numerical_gate_required=False, numerical_policy='record_only')


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid native timing')
    return value


def validate_native_sample(sample, layers=43):
    if sample.get('batch_size') != 32 or type(sample.get('sequence')) is not int or sample['sequence'] < 1:
        raise ValueError('native actual B32 sequence required')
    whole = _time(sample.get('whole_graph_ms'))
    if whole <= 0: raise ValueError('positive native graph span required')
    rows = sample.get('layers', {})
    if set(rows) != {str(i) for i in range(layers)}:
        raise ValueError('native layer timing coverage incomplete')
    totals = {'whole_graph_ms': whole}
    previous = 0.
    for layer in range(layers):
        row = rows[str(layer)]; intervals = row.get('intervals', {})
        if not {'moe','routed','gate_up','activation','down','grouping'}.issubset(intervals):
            raise ValueError('native routed event coverage incomplete')
        moe = intervals['moe']
        for name, interval in intervals.items():
            lo, hi, duration = (_time(interval.get(k)) for k in ('start_ms','end_ms','duration_ms'))
            if hi < lo or hi > whole + .005 or abs(hi-lo-duration) > .005:
                raise ValueError('native interval window invalid')
            if lo < moe['start_ms']-.005 or hi > moe['end_ms']+.005:
                raise ValueError('native span outside layer')
            totals[name+'_ms'] = totals.get(name+'_ms', 0.) + duration
        if moe['start_ms'] < previous-.005:raise ValueError('native layer order invalid')
        previous = moe['end_ms']
        for name, value in derive_layer_metrics(intervals).items():
            if abs(_time(row.get(name))-value)>.005:raise ValueError('native derived metric differs')
            if name not in ('moe_ms','branches_done_ms'):
                totals[name]=totals.get(name,0.)+value
        group=row.get('grouping')
        if not isinstance(group,dict):raise ValueError('native grouping aggregate missing')
        required=('valid_local_pairs','valid_groups','active_experts','max_rows','num_tokens_post_pad','block_size','valid_pair_capacity')
        if any(type(group.get(k)) is not int or group[k]<0 for k in required):raise ValueError('native grouping counters invalid')
        hist=group.get('row_histogram',{})
        if (any(not str(k).isdigit() or int(k)<=0 or type(v)is not int or v<=0 for k,v in hist.items())
                or sum(hist.values())!=group['active_experts']
                or sum(int(k)*v for k,v in hist.items())!=group['valid_local_pairs']
                or max((int(k) for k in hist),default=0)!=group['max_rows']
                or group['valid_local_pairs']>group['valid_pair_capacity']):
            raise ValueError('native grouping histogram inconsistent')
    totals['outside_moe_ms']=max(0.,whole-totals['moe_ms'])
    return totals


def validate_launch_trace(trace, metadata, rank):
    """Reparse actual resource rows, then match the pinned Python call contract."""
    from .native_p0_trace import parse_chrome_trace, NativeP0TraceError
    if trace.get('status')!='complete' or trace.get('rank')!=rank or trace.get('marlin_kernel_count')!=86 or trace.get('expected_layers')!=43:
        raise ValueError('native launch resource trace incomplete')
    events=[]
    for row in trace.get('kernels',[]):
        args={'grid':row.get('grid'),'block':row.get('block'),'registers per thread':row.get('registers_per_thread')}
        for public,raw in (('shared_memory_bytes','shared memory'),('static_shared_memory_bytes','static shared memory'),('dynamic_shared_memory_bytes','dynamic shared memory')):
            if row.get(public) is not None:args[raw]=row[public]
        events.append(dict(ph='X',cat='kernel',name=row.get('name'),ts=row.get('start_us'),dur=row.get('duration_us'),args=args))
    try:parsed=parse_chrome_trace(events)
    except NativeP0TraceError as exc:raise ValueError(str(exc)) from exc
    if parsed['kernels']!=trace.get('kernels') or parsed['kernel_count']!=trace.get('kernel_count'):
        raise ValueError('native sanitized resource/template records differ')
    for kernel in parsed['kernels']:
        if 'template' not in kernel:continue
        role=kernel['role'];call=metadata[str(kernel['layer'])]['marlin_calls'][0 if role=='gate_up' else 1]
        template=kernel['template']
        if any(type(call.get(k))is not int or call[k]<=0 for k in ('size_m','size_n','size_k','moe_block_size','top_k')):
            raise ValueError('native Marlin call dimensions invalid')
        if (template['tile_m']!=call['moe_block_size'] or call['size_n']%template['tile_n'] or call['size_k']%template['tile_k']
                or template['is_ep']!=call.get('is_ep') or template['has_bias']!=call.get('bias_present')
                or template['is_zp_float']!=call.get('is_zp_float')):
            raise ValueError('native kernel template and Python Marlin call disagree')
        scalar_id=re.search(r'(\d+)[uUlL]*$',template['weight_type_id'])
        if scalar_id is None or int(scalar_id.group(1))!=call.get('b_q_type_id'):
            raise ValueError('native Marlin quantization type identity differs')
    return parsed


def summarize_native_p0(before, after, *, profile_source, stride, capacity):
    def indexed(rows):
        if len(rows)!=4 or {r.get('rank') for r in rows}!={0,1,2,3}:raise ValueError('four native ranks required')
        return {r['rank']:r for r in rows}
    a,b=indexed(before),indexed(after)
    retained={}; totals={}; ranks={}
    for rank in range(4):
        p=b[rank].get('stage_profile',{}); old=a[rank].get('stage_profile',{})
        for row in (old,p):
            expected=dict(kind='native_p0',rank=rank,profile_source_sha256=profile_source,
                          measurement_mode='instrumented_diagnostic',numerical_policy='record_only',external_events=True)
            if any(row.get(k)!=v for k,v in expected.items()) or row.get('configuration')!=dict(stride=stride,capacity=capacity,layers=43,captured_batch_size=32):
                raise ValueError('native profile identity/configuration differs')
            if row.get('aborted_samples')!=0:raise ValueError('native sampled replay aborted')
        for key in ('b32_seen','sampled_total'):
            if type(old.get(key))is not int or type(p.get(key))is not int or not 0<=old[key]<=p[key]:raise ValueError('native counters invalid')
        for outer,row in ((a[rank],old),(b[rank],p)):
            if row['sampled_total']!=row['b32_seen']//stride:raise ValueError('native sampling cadence differs')
            if row['b32_seen']!=outer.get('raw_batch_histogram',{}).get('32',0):raise ValueError('native sampler and actual B32 replay counters differ')
        rows=p.get('samples',[])
        if len(rows)!=min(capacity,p['sampled_total']):raise ValueError('native retained ring incomplete')
        seq=[r.get('sequence') for r in rows]
        if any(type(s)is not int for s in seq) or seq!=sorted(set(seq)):raise ValueError('native sequences invalid')
        ordinals=list(range((p['sampled_total']-len(rows)+1)*stride,(p['sampled_total']+1)*stride,stride))
        if [r.get('b32_ordinal') for r in rows]!=ordinals or any(s<o or s>b[rank]['successful_replays'] for s,o in zip(seq,ordinals)):
            raise ValueError('native retained replay freshness differs')
        metadata=p.get('layer_metadata',{})
        if set(metadata)!={str(i) for i in range(43)}:raise ValueError('native capture metadata incomplete')
        for item in metadata.values():
            if not item.get('captured_path') or [x.get('role') for x in item.get('marlin_calls',[])]!=['gate_up','down']:
                raise ValueError('actual native branch/Marlin calls missing')
        trace=p.get('launch_trace') or {}
        validate_launch_trace(trace,metadata,rank)
        selected=[r for r in rows if r['sequence']>a[rank]['successful_replays'] and not r.get('launch_trace_active',False)]
        if not selected:raise ValueError('no untraced native B32 samples in measurement window')
        retained[rank]={r['sequence']:r for r in selected}
        totals[rank]={r['sequence']:validate_native_sample(r) for r in selected}
        if rank and list(retained[rank])!=list(retained[0]):raise ValueError('native cross-rank samples not aligned')
        keys=set.intersection(*(set(v) for v in totals[rank].values()))
        layer_summary={}
        for layer in range(43):
            lr=[r['layers'][str(layer)] for r in selected]
            spans=set.intersection(*(set(r['intervals']) for r in lr))
            derived=set.intersection(*(set(r)-{'intervals','grouping'} for r in lr))
            layer_summary[str(layer)]={'intervals_relative_to_graph_ms':{k:{field:stats([r['intervals'][k][field] for r in lr]) for field in ('start_ms','end_ms','duration_ms')} for k in sorted(spans)},
                'derived_ms':{k:stats([r[k] for r in lr]) for k in sorted(derived)},
                'durations_ms':{k:stats([r['intervals'][k]['duration_ms'] for r in lr]) for k in sorted(spans)},
                'grouping':{k:stats([r['grouping'][k] for r in lr]) for k in ('valid_local_pairs','valid_groups','active_experts','max_rows')}}
        ranks[str(rank)]=dict(totals={k:stats([v[k] for v in totals[rank].values()]) for k in sorted(keys)},
                             layers=layer_summary,layer_metadata=metadata,launch_trace=trace,kernel_to_python_agreement=True)
    sequences=list(retained[0]); critical=[];hist={}
    for seq in sequences:
        r=max(range(4),key=lambda rank:totals[rank][seq]['whole_graph_ms'])
        hist[str(r)]=hist.get(str(r),0)+1;critical.append(totals[r][seq])
    keys=set.intersection(*(set(v) for v in critical))
    return dict(schema_version=1,status='diagnostic_complete',measurement_mode='instrumented_diagnostic',
        formal_performance=False,numerical_policy='record_only',profile_source_sha256=profile_source,
        sample_count=len(sequences),sequences=sequences,per_rank=ranks,critical_rank_histogram=hist,
        critical_rank={k:stats([v[k] for v in critical]) for k in sorted(keys)},
        whole_graph_rank_spread_ms=stats([max(totals[r][s]['whole_graph_ms'] for r in range(4))-min(totals[r][s]['whole_graph_ms'] for r in range(4)) for s in sequences]),
        interpretation=['Branch overlap is interval intersection, not additive latency or causal proof of resource contention.',
                        'All-reduce intervals include stream/rank waiting; they are not pure link bandwidth.',
                        'outside_moe_ms includes attention and other graph work; absent spans are not zero.',
                        'Each critical-rank sample uses all stage values from the same rank.',
                        'One CUPTI launch sample is excluded from ordinary timing statistics. No CTA budget was changed.'])


def validate_native_p0_smoke(folder, identity):
    from tests.gpu.check_native_p0_events import hashes
    expected_hashes=hashes(); files={}
    if expected_hashes['profile_source_sha256']!=identity['native_p0_source_sha256']:raise ValueError('native P0 smoke source differs')
    for rank in range(4):
        path=Path(folder)/f'native-p0-event-rank{rank}.json'
        if path.is_symlink():raise ValueError('native smoke symlink forbidden')
        raw=path.read_bytes();row=json.loads(raw);files[path.name]=hashlib.sha256(raw).hexdigest()
        expected=dict(status='passed',gpu_checks_status='passed',rank=rank,world_size=4,image_id=identity['image_id'],
                      case_count=4,replay_count=4,graph_capture_count=1,external_events=True,two_streams=True,
                      nccl_all_reduce=True,generic_launch_metadata_verified=True,numerical_policy='record_only',model_execution_evidence=False)
        if any(row.get(k)!=v for k,v in expected.items()) or any(row.get(k)!=v for k,v in expected_hashes.items()) or row.get('source_hashes_after')!=expected_hashes:
            raise ValueError('native event/CUPTI smoke evidence incomplete')
        cleanup=row.get('cleanup',{})
        if cleanup.get('status')!='complete' or cleanup.get('graphs_reset')!=1 or cleanup.get('completed_phases')!=list(CLEANUP_PHASES):
            raise ValueError('native smoke cleanup incomplete')
        cases=row.get('cases',[])
        if len(cases)!=4 or [c.get('sequence') for c in cases]!=[1,2,3,4]:raise ValueError('native smoke case coverage incomplete')
        runtime=row.get('runtime',{})
        if (runtime.get('image_id')!=identity['image_id'] or runtime.get('capability')!=[9,0]
                or runtime.get('device_count')!=4 or 'H100' not in runtime.get('device_name','')):
            raise ValueError('native smoke H100 runtime incomplete')
        profile=row.get('native_p0_profile',{})
        if (profile.get('rank')!=rank or profile.get('profile_source_sha256')!=identity['native_p0_source_sha256']
                or profile.get('configuration')!=dict(stride=1,capacity=4,layers=1,captured_batch_size=32)
                or profile.get('sampled_total')!=4 or profile.get('b32_seen')!=4 or profile.get('aborted_samples')!=0
                or profile.get('samples')!=[c.get('sample') for c in cases]):
            raise ValueError('native smoke sampler source/coverage differs')
        for case in cases:
            if any(case.get(k)is not True for k in ('finite','fresh_events_verified','join_order_verified')):raise ValueError('native event replay freshness missing')
            sample=case.get('sample',{})
            if sample.get('sequence')!=case['sequence'] or sample.get('b32_ordinal')!=case['sequence'] or sample.get('batch_size')!=32:
                raise ValueError('native smoke sample sequence differs')
            whole=_time(sample.get('whole_graph_ms'));intervals=sample.get('layers',{}).get('0',{}).get('intervals',{})
            if whole<=0 or not {'moe','routed','shared','shared_merge','all_reduce'}.issubset(intervals):raise ValueError('native smoke intervals incomplete')
            for interval in intervals.values():
                lo,hi,duration=(_time(interval.get(k)) for k in ('start_ms','end_ms','duration_ms'))
                if hi<lo or hi>whole+.005 or abs(hi-lo-duration)>.005:raise ValueError('native smoke intervals invalid')
            if intervals['shared_merge']['start_ms']+.005<max(intervals[k]['end_ms'] for k in ('routed','shared')):
                raise ValueError('native smoke join precedes branch completion')
        trace=profile.get('launch_trace') or {}
        if trace.get('status')!='complete' or not trace.get('kernels'):raise ValueError('native smoke launch metadata missing')
    return dict(status='passed',rank_count=4,cases_per_rank=4,files=files,numerical_policy='record_only',model_execution_evidence=False)
