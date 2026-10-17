"""CPU validation for matched eager C32 service experiments."""
import hashlib
import json
import math
import re
from collections import Counter

VARIANTS=('native','persistent_cached','staged_direct')
HOPPER_VARIANTS=('native','staged_direct','hopper_fp8')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_online_gate(ranks,source,checkpoint):
    if len(ranks)!=4 or {row.get('rank') for row in ranks}!={0,1,2,3}:
        raise ValueError('four online staged numerical rank reports required')
    expected={(-1,v,b) for v,b in enumerate((1,7,17,32,7,1))}|{(layer,layer,32) for layer in range(43)}
    for row in ranks:
        if (row.get('status')!='online_staged_checks_passed' or row.get('source_sha256')!=source
                or row.get('checkpoint_content_sha256')!=checkpoint or row.get('layers')!=list(range(43))):
            raise ValueError('online gate source/checkpoint/all43 identity differs')
        cases=row.get('cases',[])
        if len(cases)!=49 or {(x.get('layer'),x.get('variant'),x.get('batch')) for x in cases}!=expected:
            raise ValueError('online gate requires six changing synthetic cases and all43 real layers atB32')
        for case in cases:
            if (case.get('metadata')!='exact_p1_match'
                    or case.get('assembly')!='exact_fp32_slots_then_ordered_sum_bf16_origin_mask'):
                raise ValueError('exact online grouping and assembly evidence required')
            for kind in ('local_slots','reference','persistent'):
                metrics=case.get(kind,{})
                relative,maximum=metrics.get('relative_l2'),metrics.get('max_abs')
                if (type(relative) not in (float,int) or not math.isfinite(relative) or not 0<=relative<=.025
                        or type(maximum) not in (float,int) or not math.isfinite(maximum) or maximum<0):
                    raise ValueError('online numerical comparison failed or incomplete')
    return {'status':'passed','ranks':4,'cases_per_rank':49,'all43_real_batch':32,
            'source_sha256':source,'checkpoint_content_sha256':checkpoint}


def compare_acceptance(reference,candidate):
    def normalize(value):
        rows=value.get('records',[])
        if len(rows)!=32:raise ValueError('acceptance requires32 actual request sequences')
        result={}
        for row in rows:
            index=row.get('request_index');ids=row.get('output_ids');request_id=row.get('request_id')
            if (type(index) is not int or index<0 or index in result or not isinstance(request_id,str)
                    or not isinstance(ids,list) or len(ids)!=8 or any(type(x) is not int or x<0 for x in ids)):
                raise ValueError('invalid acceptance sequence metadata')
            result[index]=(request_id,ids)
        return result
    before,after=normalize(reference),normalize(candidate)
    if before!=after:raise ValueError('generated token sequences differ from native acceptance')
    return {'status':'passed','matched_requests':32,'matched_generated_tokens':256,
            'sequence_commitment_sha256':digest(before),'logprob_policy':'record_only'}


def validate_coverage_delta(before,after,mode,source,checkpoint, *, candidate_source_sha256=None,assembly_source_sha256=None):
    if mode not in ('persistent','staged_direct','hopper_fp8'):raise ValueError('unsupported decode implementation')
    def index(values):
        if len(values)!=4 or {v.get('rank') for v in values}!={0,1,2,3}:
            raise ValueError('four distinct ranks required')
        for v in values:
            if mode=='hopper_fp8' and (not candidate_source_sha256 or not assembly_source_sha256
                    or v.get('candidate_source_sha256')!=candidate_source_sha256
                    or v.get('assembly_source_sha256')!=assembly_source_sha256
                    or v.get('numerical_policy')!='record_only' or v.get('hopper_max_batch')!=32
                    or v.get('hopper_slot_collective')!='torch.distributed.all_reduce/SUM/tp_device_group'):
                raise ValueError('Hopper candidate/assembly source or runtime contract differs')
            if (v.get('source_sha256')!=source or v.get('profile_checkpoint_content_sha256')!=checkpoint
                    or v.get('kernel')!='tiled' or v.get('graph_enabled') is not False
                    or v.get('hot_cache') is not True or v.get('decode_implementation')!=mode
                    or v.get('fixed_plan')!=[1,1,0] or v.get('profile_bank_sha256') is not None
                    or v.get('execution_profile_configuration') is not None
                    or set(v.get('layers',{}))!={str(x) for x in range(43)}):
                raise ValueError('service differs from all43 cached/fixed/eager source contract')
            if v.get('model_batch_counts',{}).get('active') is not False:
                raise ValueError('coverage must be collected while service is idle')
        return {v['rank']:v for v in values}
    a,b=index(before),index(after);result={'status':'passed','mode':mode,'ranks':{}}
    for rank in range(4):
        old,new=a[rank],b[rank];layers={};collectives=0
        if new['model_batch_counts']['failed_forwards']!=old['model_batch_counts']['failed_forwards']:
            raise ValueError('model forward failed during measured execution')
        ah=old['model_batch_counts']['decode_valid_batch_histogram'];bh=new['model_batch_counts']['decode_valid_batch_histogram']
        histogram={k:bh.get(k,0)-ah.get(k,0) for k in set(ah)|set(bh)}
        if any(type(v) is not int or v<0 for v in histogram.values()) or histogram.get('32',0)<1:
            raise ValueError('actual decode batch32 evidence required')
        for layer in map(str,range(43)):
            x,y=old['layers'][layer],new['layers'][layer]
            delta=[end-start for start,end in zip(x['device_calls'],y['device_calls'])]
            if len(delta)!=3 or min(delta)<0 or delta[1]<1:raise ValueError('missing all43 decode execution')
            keys=('staged_decode','persistent_decode','persistent_prefill','persistent_other')
            if mode=='hopper_fp8' or 'hopper_decode' in x['implementation_calls'] or 'hopper_decode' in y['implementation_calls']:keys+=('hopper_decode',)
            calls={k:y['implementation_calls'][k]-x['implementation_calls'][k] for k in keys}
            wanted={'staged_direct':'staged_decode','persistent':'persistent_decode','hopper_fp8':'hopper_decode'}[mode]
            other=[k for k in ('staged_decode','persistent_decode','hopper_decode') if k!=wanted]
            if any(type(v) is not int or v<0 for v in calls.values()) or calls[wanted]!=delta[1] or any(calls.get(k,0)!=0 for k in other):
                raise ValueError('decode silently changed implementation or coverage is incomplete')
            collectives+=calls['staged_decode']+calls.get('hopper_decode',0);layers[layer]={'device_delta':delta,'implementation_delta':calls}
        if new['additional_slot_collectives']-old['additional_slot_collectives']!=collectives:
            raise ValueError('exactly one slot all-reduce per staged layer call required')
        result['ranks'][str(rank)]={'layers':layers,'decode_batch_histogram':histogram,
            'additional_slot_collectives':collectives}
    result['decode_batch32_forwards_rank0']=result['ranks']['0']['decode_batch_histogram']['32']
    return result


def scheduler_counts(log):
    counts=Counter()
    for line in log.splitlines():
        if 'Decode batch' not in line:continue
        match=re.search(r'#running-req\s*[:=]\s*(\d+)',line)
        if match:counts[int(match.group(1))]+=1
    return {'scope':'sampled scheduler decode log running requests; not GPU event timing',
            'histogram':{str(k):v for k,v in sorted(counts.items())},
            'max_running_requests':max(counts,default=0),'samples':sum(counts.values())}


def validate_client_summary(value,seconds,dataset_sha256):
    if (value.get('status')!='passed' or value.get('dataset_sha256')!=dataset_sha256
            or not re.fullmatch('[0-9a-f]{64}',value.get('request_order_sha256',''))
            or value.get('generation_settings')!={'max_new_tokens':1024,'temperature':0,'ignore_eos':True}
            or value.get('concurrency')!=32 or value.get('peak_client_inflight')!=32
            or value.get('full_window_observed') is not True or value.get('error_codes')!=[]
            or value.get('abort_failures')!=0 or value.get('requests_failed')!=0):
        raise ValueError('client did not complete the requested healthy C32 stream window')
    for key in ('requested_seconds','window_seconds'):
        duration=value.get(key)
        if type(duration) not in (float,int) or not math.isfinite(duration) or not math.isclose(duration,seconds,rel_tol=1e-9):
            raise ValueError('client measurement window differs')
    for key in ('requests_started','requests_completed','requests_failed','requests_unfinished',
                'output_tokens_in_window','decode_tokens_in_window'):
        if type(value.get(key)) is not int or value[key]<0:raise ValueError('invalid client token/request counter')
    if (sum(value[k] for k in ('requests_completed','requests_failed','requests_unfinished'))!=value['requests_started']
            or value['requests_started']<32 or not 0<value['decode_tokens_in_window']<=value['output_tokens_in_window']
            or not 0<=value['output_tokens_in_window']-value['decode_tokens_in_window']<=value['requests_started']):
        raise ValueError('inconsistent stream counters')
    rate=value.get('decode_tokens_per_second')
    if type(rate) not in (int,float) or not math.isfinite(rate) or not math.isclose(rate,value['decode_tokens_in_window']/seconds,rel_tol=1e-9):
        raise ValueError('decode TPS must include all in-window streamed tokens with the declared denominator')


def summarize_comparison(groups,comparison_suite='staged'):
    if comparison_suite not in ('staged','hopper'):raise ValueError('unknown comparison suite')
    variants=HOPPER_VARIANTS if comparison_suite=='hopper' else VARIANTS
    if set(groups)!=set(variants):raise ValueError('all three eager groups required')
    identities=[groups[v]['identity'] for v in variants]
    if any(x!=identities[0] for x in identities[1:]) or identities[0].get('decode_graph')!='disabled':
        raise ValueError('groups differ in workload/runtime/source/execution mode')
    rates={}
    for variant in variants:
        group=groups[variant];client=group['client'];rate=client.get('decode_tokens_per_second')
        if (group['status']!='passed' or client.get('status')!='passed'
                or group['acceptance'].get('status')!=('recorded' if comparison_suite=='hopper' else 'passed')
                or group['batch_evidence'].get('max_running_requests')!=32
                or type(rate) not in (int,float) or not math.isfinite(rate) or rate<=0):
            raise ValueError('complete accepted C32 measurement required before ratio')
        rates[variant]=rate
    if comparison_suite=='hopper':
        return {'schema_version':2,'status':'passed','scope':'Hopper C32 eager serving comparison on fixed custom workload',
            'metric':'client-observed decode tokens/s; cumulative stream counts inside window; first token excluded',
            'formal_gpqa':False,'identity':identities[0],'numerical_policy':'record_only',
            'generation_comparison_policy':'record_only','decode_tokens_per_second':rates,
            'hopper_vs_native_ratio':rates['hopper_fp8']/rates['native'],
            'hopper_vs_bf16_staged_ratio':rates['hopper_fp8']/rates['staged_direct'],
            'communication_note':'BF16 staged and Hopper include one slot all-reduce plus unchanged outer shared/TP reduction'}
    return {'schema_version':1,'status':'passed','scope':'C32 eager serving comparison on fixed custom workload',
        'metric':'client-observed decode tokens/s; cumulative stream counts inside window; first token excluded',
        'formal_gpqa':False,'identity':identities[0],'decode_tokens_per_second':rates,
        'staged_vs_native_ratio':rates['staged_direct']/rates['native'],
        'staged_vs_persistent_ratio':rates['staged_direct']/rates['persistent_cached'],
        'persistent_role':'optimization control; native versus staged is serving comparison',
        'communication_note':'staged includes slot all-reduce plus unchanged outer shared/TP reduction'}
