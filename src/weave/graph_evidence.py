"""Strict idle evidence for actual C32 decode graph execution (numerics record only)."""
import hashlib
import json
import math
from pathlib import Path
from .graph_cleanup import CLEANUP_PHASES


def _index(rows):
    if len(rows)!=4 or {r.get('rank') for r in rows}!={0,1,2,3}:raise ValueError('four distinct graph ranks required')
    return {r['rank']:r for r in rows}


def _delta(old,new,key):
    a,b=old.get(key),new.get(key)
    if type(a) is not int or type(b) is not int or a<0 or b<a:raise ValueError('invalid graph counter '+key)
    return b-a


def _histogram(old,new,key):
    a,b=old.get(key,{}),new.get(key,{})
    if any(str(k) not in {str(i) for i in range(1,33)} or type(v) is not int or v<0 for row in (a,b) for k,v in row.items()):raise ValueError('invalid graph histogram')
    result={k:b.get(k,0)-a.get(k,0) for k in set(a)|set(b)}
    if any(v<0 for v in result.values()):raise ValueError('graph histogram regressed')
    return {k:v for k,v in result.items() if v}


def validate_graph_delta(before,after,*,expected_observer_source=None,expected_sources=None):
    a,b=_index(before),_index(after);result={'status':'passed','numerical_policy':'record_only','graph_mode':'full','ranks':{}}
    common=None
    for rank in range(4):
        x,y=a[rank],b[rank]
        for row in (x,y):
            if (row.get('active') is not False or row.get('graph_mode')!='full' or row.get('backend')!='FullCudaGraphBackend'
                or row.get('numerical_policy')!='record_only' or row.get('captured_batches')!=list(range(1,33))
                or row.get('capture_successes')!=32 or row.get('capture_attempts')!=32 or row.get('capture_failures')!=0):raise ValueError('actual complete idle graph capture required')
            if not row.get('source_sha256') or not row.get('observer_source_sha256'):raise ValueError('source-bound observer required')
            if expected_observer_source is not None and row['observer_source_sha256']!=expected_observer_source:raise ValueError('observer source differs')
            if expected_sources is not None and row['source_sha256']!=expected_sources:raise ValueError('pinned graph sources differ')
        if any(x[k]!=y[k] for k in ('source_sha256','observer_source_sha256')):raise ValueError('graph source drift')
        for key in ('capture_attempts','capture_successes','capture_failures','failed_replays'):
            if _delta(x,y,key):raise ValueError('recapture or failed replay during measurement')
        for key in ('capture','warmup','eager_decode','other'):
            if _delta(x['model_forwards'],y['model_forwards'],key):raise ValueError('non-replay decode forward during measurement')
        count=_delta(x,y,'successful_replays');raw=_histogram(x,y,'raw_batch_histogram');captured=_histogram(x,y,'captured_batch_histogram')
        if count<1 or raw!=captured or sum(raw.values())!=count or raw.get('32',0)<1:raise ValueError('unpadded actual graph B32 replay required')
        identity=(x['source_sha256'],x['observer_source_sha256'],count,raw)
        if common is not None and identity!=common:raise ValueError('cross-rank graph replay/source drift')
        common=identity;result['ranks'][str(rank)]={'replays':count,'decode_batch_histogram':raw}
    result.update(replays=common[2],decode_batch32_forwards_rank0=common[3]['32'],max_running_requests=32)
    return result


def validate_graph_weave_delta(before,after,graph_check,identity):
    a,b=_index(before),_index(after);replays=graph_check['replays'];result={'status':'passed','numerical_policy':'record_only','ranks':{},'max_running_requests':32}
    for rank in range(4):
        x,y=a[rank],b[rank]
        for row in (x,y):
            if (row.get('graph_enabled') is not True or row.get('kernel')!='tiled' or row.get('hot_cache') is not True
                or row.get('fixed_plan')!=[1,1,0] or row.get('profile_bank_sha256') is not None
                or row.get('execution_profile_configuration') is not None or row.get('decode_implementation')!='hopper_fp8'
                or row.get('prefill_implementation')!='hopper_fp8' or row.get('numerical_policy')!='record_only'
                or row.get('model_batch_counts',{}).get('active') is not False or set(row.get('layers',{}))!=set(map(str,range(43)))):raise ValueError('all43 fixed cached Hopper graph contract required')
            fields={'source_sha256':'source_sha256','profile_checkpoint_content_sha256':'checkpoint_content_sha256',**{k:k for k in ('candidate_source_sha256','assembly_source_sha256','prefill_source_sha256','prefill_runtime_source_sha256','graph_runtime_source_sha256')}}
            if any(not identity.get(expected) or row.get(actual)!=identity[expected] for actual,expected in fields.items()):raise ValueError('graph runtime/checkpoint source differs')
        layers={};prefill=0
        if _delta(x['model_batch_counts'],y['model_batch_counts'],'failed_forwards'):raise ValueError('failed model forward')
        for layer in map(str,range(43)):
            p,q=x['layers'][layer],y['layers'][layer]
            if len(p.get('device_calls',[]))!=3 or len(q.get('device_calls',[]))!=3:raise ValueError('actual device counters required')
            delta=[end-start for start,end in zip(p['device_calls'],q['device_calls'])]
            if any(type(v) is not int or v<0 for v in delta) or delta[1]!=replays or delta[0]<1 or delta[2]!=0:raise ValueError('device replay/prefill execution differs')
            for key in ('hopper_decode','staged_decode','persistent_decode','persistent_prefill','persistent_other'):
                if _delta(p['implementation_calls'],q['implementation_calls'],key):raise ValueError('eager implementation fallback')
            if _delta(p['implementation_calls'],q['implementation_calls'],'hopper_prefill')!=delta[0]:raise ValueError('actual Hopper prefill differs')
            if _delta(p,q,'capture_calls'):raise ValueError('layer recapture')
            if _delta(p,q,'python_decode_calls'):raise ValueError('eager Python decode fallback')
            prefill+=delta[0];layers[layer]={'device_delta':delta}
        histogram=_histogram(x['model_batch_counts'],y['model_batch_counts'],'decode_valid_batch_histogram')
        if histogram!=graph_check['ranks'][str(rank)]['decode_batch_histogram']:raise ValueError('model/observer replay histogram differs')
        if _delta(x,y,'device_slot_collectives')!=43*replays:raise ValueError('actual graph device collective count differs')
        if _delta(x,y,'additional_prefill_slot_collectives')!=prefill:raise ValueError('separate actual prefill collective count differs')
        result['ranks'][str(rank)]={'layers':layers,'device_slot_collectives':43*replays,'additional_prefill_slot_collectives':prefill}
    result['decode_batch32_forwards_rank0']=graph_check['decode_batch32_forwards_rank0'];return result


def validate_graph_smoke(folder,identity):
    rows=[];files={}
    expected=[{'case':'all_buckets','batch':b,'handle':0,'variant':1} for b in range(1,33)]+[{'case':'cross_bucket','batch':b,'handle':j%2,'variant':j+2} for j,b in enumerate([32,1,17,7,32])]+[{'case':'after_prefill','batch':32,'handle':0,'variant':9}]
    for rank in range(4):
        path=Path(folder)/f'hopper-graph-rank{rank}.json'
        if path.is_symlink():raise ValueError('graph proof symlink forbidden')
        raw=path.read_bytes();row=json.loads(raw);files[path.name]=hashlib.sha256(raw).hexdigest()
        if (row.get('rank')!=rank or row.get('world_size')!=4 or row.get('status')!='passed' or row.get('numerical_policy')!='record_only'
            or row.get('raw_content_exported') is not False or row.get('graph_capture_count')!=64 or row.get('replay_count')!=38
            or row.get('replay_device_collective_delta')!=38):raise ValueError('complete four-rank graph smoke required')
        cleanup = row.get('cleanup', {})
        if (row.get('schema_version') != 2 or row.get('gpu_checks_status') != 'passed'
            or cleanup.get('status') != 'complete' or cleanup.get('phase') != 'complete'
            or cleanup.get('graphs_owned') != 64 or cleanup.get('graphs_reset') != 64
            or cleanup.get('completed_phases') != list(CLEANUP_PHASES)):
            raise ValueError('completed graph reset and communicator cleanup required')
        start, end = cleanup.get('started_monotonic'), cleanup.get('finished_monotonic')
        if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in (start, end)) or end < start:
            raise ValueError('valid graph cleanup timestamps required')
        for key in ('candidate_source_sha256','assembly_source_sha256','graph_runtime_source_sha256','prefill_source_sha256','prefill_runtime_source_sha256'):
            if row.get(key)!=identity.get(key) or not identity.get(key):raise ValueError('graph proof source differs')
        if row.get('runtime',{}).get('image_id')!=identity['image_id']:raise ValueError('graph smoke image differs')
        cases=row.get('cases',[])
        if [{k:c.get(k) for k in ('case','batch','handle','variant')} for c in cases]!=expected:raise ValueError('complete graph replay matrix required')
        for case in cases:
            if any(case.get(k) is not True for k in ('finite','metadata_exact','assembly_exact','stable_addresses','binding_alive')) or case.get('device_collective_delta')!=1 or case.get('device_counter_after',0)-case.get('device_counter_before',0)!=1 or case.get('input_update_verified') is not True or case.get('oracle_tokens')!=min(case['batch'],2):raise ValueError('invalid graph replay evidence')
            if any(type(value) in (float,int) and not math.isfinite(value) for metric in case.get('numerical',{}).values() for value in metric.values()):raise ValueError('nonfinite numerical record')
            if set(case.get('numerical',{}))!={'eager_graph','fp8_oracle','bf16_oracle','bf16_reference'} or any(v.get('finite') is not True for v in case['numerical'].values()):raise ValueError('finite record-only numerical evidence required')
            for metric in case['numerical'].values():
                for field in ('relative_l2','max_abs'):
                    value=metric.get(field)
                    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                        raise ValueError('finite nonnegative numerical record required: '+field)
        prefill=row.get('prefill_interleave',{})
        if prefill!={'batch':65,'finite':True,'additional_slot_collectives':1,'separate_workspace':True}:raise ValueError('separate prefill interleave proof required')
        if row.get('capture_batches')!=list(range(32,0,-1)) or row.get('capture_handles')!=[0,1] or row.get('captures')!=[{'batch':b,'handle':h} for h in (0,1) for b in range(32,0,-1)] or row.get('capture_device_collective_delta')!=0:raise ValueError('actual capture matrix differs')
        source_keys=('candidate_source_sha256','assembly_source_sha256','graph_runtime_source_sha256','prefill_source_sha256','prefill_runtime_source_sha256','bf16_source_sha256')
        if row.get('source_hashes_after')!={key:row.get(key) for key in source_keys}:raise ValueError('graph proof source changed')
        if row.get('bf16_source_sha256')!=identity['source_sha256']:raise ValueError('graph BF16 source differs')
        if row.get('device_slot_collectives')!=140 or row.get('eager_slot_collectives')!=102:raise ValueError('actual smoke collectives differ')
        if row.get('runtime',{}).get('capability')!=[9,0] or row['runtime'].get('device_count')!=4 or 'H100' not in row['runtime'].get('device_name',''):raise ValueError('four H100 graph proof required')
        if row.get('case_count')!=38:raise ValueError('graph proof case count differs')
        rows.append(row)
    return {'status':'passed','numerical_policy':'record_only','files':files,'rank_count':4,'case_count_per_rank':38}
