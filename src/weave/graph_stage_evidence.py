"""Scalar graph-stage diagnostic validation and aligned four-rank accounting."""
import hashlib
import json
import math
from pathlib import Path
from .graph_cleanup import CLEANUP_PHASES

STAGES = ('expert_pipeline_ms', 'slot_sum_ms', 'finalize_ms')
TOTALS = ('whole_graph_ms', *STAGES, 'routed_total_ms', 'residual_ms')


def stats(values):
    ordered = sorted(values)
    def percentile(q):
        position = (len(ordered)-1)*q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)
    return {'count':len(values), 'mean':sum(values)/len(values),
            'p50':percentile(.5), 'p95':percentile(.95), 'max':ordered[-1]}


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid nonnegative stage time')
    return value


def _close(a, b):
    # CUDA events have finite timestamp resolution. This validates time-window
    # accounting, never expert-output numerical accuracy.
    return abs(a-b) <= max(.005, max(abs(a),abs(b))*1e-5)


def validate_sample(sample, *, layers):
    if type(sample.get('sequence')) is not int or sample['sequence'] < 1 or sample.get('batch') != 32:
        raise ValueError('actual B32 replay sequence required')
    whole = _time(sample.get('whole_graph_ms'))
    if whole <= 0: raise ValueError('positive graph interval required')
    rows = sample.get('layers', [])
    if len(rows) != layers or [r.get('layer') for r in rows] != list(range(layers)):
        raise ValueError('complete ordered layer timings required')
    sums = {name:0. for name in STAGES}
    for row in rows:
        for name in STAGES: sums[name] += _time(row.get(name))
        if not _close(sum(row[name] for name in STAGES), _time(row.get('routed_total_ms'))):
            raise ValueError('layer timing closure differs')
    routed = sum(sums.values())
    if not _close(routed, _time(sample.get('routed_total_ms'))):
        raise ValueError('routed timing closure differs')
    residual = sample.get('residual_ms')
    if type(residual) not in (int,float) or not math.isfinite(residual) or not _close(residual,whole-routed):
        raise ValueError('remaining graph timing differs')
    if routed > whole and not _close(routed,whole):
        raise ValueError('routed intervals exceed graph window')
    return dict(sums,whole_graph_ms=whole,routed_total_ms=routed,residual_ms=whole-routed)


def summarize_graph_stages(before, after, *, kind, profile_source, stride, capacity):
    if kind not in ('native','weave'): raise ValueError('unknown timing kind')
    def index(rows):
        if len(rows)!=4 or {r.get('rank') for r in rows}!={0,1,2,3}:raise ValueError('four stage ranks required')
        return {r['rank']:r for r in rows}
    a,b=index(before),index(after);layers=43 if kind=='weave' else 0
    retained={}; totals={}; result={'schema_version':1,'status':'passed',
        'measurement_mode':'instrumented_diagnostic','formal_performance':False,
        'numerical_policy':'record_only','kind':kind,'profile_source_sha256':profile_source,
        'stride':stride,'capacity':capacity,'per_rank':{},
        'boundaries':{'expert_pipeline_ms':'workspace.run including input copies, grouping, quantization and expert compute',
                     'slot_sum_ms':'slot SUM stream interval including NCCL and rank/stream waiting',
                     'finalize_ms':'device counter and ordered finalizer',
                     'residual_ms':'remaining graph span: attention, shared experts, outer TP and launch gaps; not attributed separately'}}
    for rank in range(4):
        for row in (a[rank],b[rank]):
            p=row.get('stage_profile',{})
            expected={'schema_version':1,'measurement_mode':'instrumented_diagnostic','kind':kind,
                      'rank':rank,'profile_source_sha256':profile_source,'stride':stride,'capacity':capacity,
                      'layers_expected':layers,'external_events':True,'numerical_policy':'record_only'}
            if any(p.get(k)!=v for k,v in expected.items()) or p.get('registered_layers')!=list(range(layers)):
                raise ValueError('stage configuration/source/layer identity differs')
        p=b[rank]['stage_profile'];old=a[rank]['stage_profile']
        for key in ('b32_seen','sampled_total'):
            if type(p.get(key)) is not int or type(old.get(key)) is not int or not 0<=old[key]<=p[key]:
                raise ValueError('stage counters invalid')
        if p['sampled_total']!=p['b32_seen']//stride:raise ValueError('sample cadence differs')
        rows=p.get('samples',[])
        if len(rows)!=min(capacity,p['sampled_total']):raise ValueError('retained timing ring incomplete')
        sequences=[s.get('sequence') for s in rows]
        if any(type(s) is not int for s in sequences) or sequences!=sorted(set(sequences)):
            raise ValueError('unique ordered sample sequences required')
        ordinals=[s.get('b32_ordinal') for s in rows]
        expected_ordinals=list(range((p['sampled_total']-len(rows)+1)*stride,(p['sampled_total']+1)*stride,stride))
        if ordinals!=expected_ordinals:raise ValueError('retained B32 sampling cadence differs')
        if any(s<o for s,o in zip(sequences,ordinals)):raise ValueError('B32 ordinal exceeds total replay sequence')
        if sequences and sequences[-1]>b[rank]['successful_replays']:raise ValueError('sample beyond actual replay')
        selected=[s for s in rows if s['sequence']>a[rank]['successful_replays']]
        if not selected:raise ValueError('no B32 samples inside measurement window')
        retained[rank]={s['sequence']:s for s in selected}
        totals[rank]={s['sequence']:validate_sample(s,layers=layers) for s in selected}
        if rank and list(retained[rank])!=list(retained[0]):raise ValueError('cross-rank sample sequence mismatch')
        result['per_rank'][str(rank)]={
            'totals':{k:stats([v[k] for v in totals[rank].values()]) for k in TOTALS},
            'layers':{str(layer):{k:stats([s['layers'][layer][k] for s in selected]) for k in (*STAGES,'routed_total_ms')}
                      for layer in range(layers)}}
    sequences=list(retained[0]);critical=[];hist={}
    for seq in sequences:
        rank=max(range(4),key=lambda r:totals[r][seq]['whole_graph_ms'])
        hist[str(rank)]=hist.get(str(rank),0)+1;critical.append(totals[rank][seq])
    result.update(sample_count=len(sequences),sequences=sequences,critical_rank_histogram=hist,
        critical_rank={k:stats([s[k] for s in critical]) for k in TOTALS},
        whole_graph_rank_spread_ms=stats([max(totals[r][s]['whole_graph_ms'] for r in range(4))-
                                         min(totals[r][s]['whole_graph_ms'] for r in range(4)) for s in sequences]))
    result['critical_rank_mean_fraction']={k:result['critical_rank'][k]['mean']/result['critical_rank']['whole_graph_ms']['mean']
                                         for k in (*STAGES,'residual_ms')}
    return result


def validate_stage_event_smoke(folder, identity):
    """Require actual event nodes + finite records + clean process resources."""
    files={}
    source_fields=('candidate_source_sha256','assembly_source_sha256','graph_runtime_source_sha256',
                   'prefill_source_sha256','prefill_runtime_source_sha256')
    for rank in range(4):
        path=Path(folder)/f'stage-event-rank{rank}.json'
        if path.is_symlink():raise ValueError('stage proof symlink forbidden')
        raw=path.read_bytes();row=json.loads(raw);files[path.name]=hashlib.sha256(raw).hexdigest()
        expected={'schema_version':1,'status':'passed','gpu_checks_status':'passed','rank':rank,'world_size':4,
                  'measurement_mode':'instrumented_diagnostic','numerical_policy':'record_only',
                  'raw_content_exported':False,'case_count':4,'replay_count':4,'graph_capture_count':2,
                  'external_events':True,'replay_device_collective_delta':4,
                  'profile_source_sha256':identity['graph_stage_source_sha256'],'image_id':identity['image_id'],
                  'shape':dict(batch=32,hidden=128,intermediate=128,experts=32,topk=6,layers=1)}
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('complete stage event GPU proof required')
        if any(row.get(k)!=identity[k] for k in source_fields) or row.get('bf16_source_sha256')!=identity['source_sha256']:
            raise ValueError('stage smoke source mismatch')
        source_keys=(*source_fields,'bf16_source_sha256','profile_source_sha256')
        if row.get('source_hashes_after')!={k:row[k] for k in source_keys}:raise ValueError('stage sources changed')
        runtime=row.get('runtime',{})
        if runtime.get('image_id')!=identity['image_id'] or runtime.get('capability')!=[9,0] or runtime.get('device_count')!=4 or 'H100' not in runtime.get('device_name',''):
            raise ValueError('four H100 stage runtime required')
        cleanup=row.get('cleanup',{})
        if (cleanup.get('status')!='complete' or cleanup.get('phase')!='complete' or cleanup.get('graphs_owned')!=2
            or cleanup.get('graphs_reset')!=2 or cleanup.get('completed_phases')!=list(CLEANUP_PHASES)):
            raise ValueError('stage graph cleanup incomplete')
        start,end=cleanup.get('started_monotonic'),cleanup.get('finished_monotonic')
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in (start,end)) or end<start:
            raise ValueError('stage cleanup timestamps invalid')
        cases=row.get('cases',[])
        if len(cases)!=4 or [(c.get('sequence'),c.get('batch'),c.get('variant')) for c in cases]!=[(i,32,i) for i in range(1,5)]:
            raise ValueError('stage replay case coverage differs')
        for case in cases:
            if any(case.get(k) is not True for k in ('finite','metadata_exact','assembly_exact','stable_addresses','input_update_verified','fresh_events_verified')) or case.get('device_collective_delta')!=1:
                raise ValueError('stage replay freshness evidence missing')
            sample=case['stage_sample'];validate_sample(sample,layers=1)
            if sample.get('sequence')!=case['sequence'] or sample.get('b32_ordinal')!=case['sequence']:
                raise ValueError('stage sample case mismatch')
            metrics=case.get('numerical',{})
            if set(metrics)!={'original_graph','original_eager','fp8_oracle','bf16_oracle','bf16_reference'}:
                raise ValueError('record-only reference metrics missing')
            for metric in metrics.values():
                if metric.get('finite') is not True:raise ValueError('nonfinite stage reference')
                for key in ('relative_l2','max_abs'):_time(metric.get(key))
        profile=row.get('stage_profile',{})
        if (profile.get('profile_source_sha256')!=identity['graph_stage_source_sha256'] or profile.get('rank')!=rank
            or profile.get('kind')!='weave' or profile.get('stride')!=1 or profile.get('layers_expected')!=1
            or profile.get('sampled_total')!=4 or profile.get('b32_seen')!=4 or profile.get('aborted_samples')!=0
            or profile.get('samples')!=[case['stage_sample'] for case in cases]):
            raise ValueError('actual stage smoke snapshot differs')
    return {'status':'passed','measurement_mode':'instrumented_diagnostic','numerical_policy':'record_only',
            'rank_count':4,'cases_per_rank':4,'files':files}
