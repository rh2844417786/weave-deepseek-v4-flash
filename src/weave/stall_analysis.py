"""Diagnostic interval analysis; overlap is observation and carries no causality."""
import math
import statistics


def _number(v):
    if isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v):
        raise ValueError('stall_invalid_numeric')
    return v


def _span(a,b):
    a,b=_number(a),_number(b)
    if b<a: raise ValueError('stall_invalid_interval')
    return b-a


def _stats(values):
    return dict(max=max(values),median=statistics.median(values),spread=max(values)-min(values))


def _overlap(intervals,start,end):
    merged=[]
    for a,b in sorted((max(a,start),min(b,end)) for a,b in intervals if b>start and a<end):
        if merged and a<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],b)
        else: merged.append([a,b])
    return sum(b-a for a,b in merged)


def analyze_stalls(rank_reports, client_report, window):
    out=dict(schema_version=1,status='failed',error_codes=[],instrumented_only=True,
             instrumented_diagnostic=True,used_for_formal_performance=False,numerical_policy='record_only',
             interpretation='Observed instrumented intervals and overlap; no causal attribution. CUDA stream spans include waits and launch starvation; routed spans include dispatch/combine.',
             clock_contract='Same-host Linux CLOCK_MONOTONIC; CUDA intervals mapped approximately using rank anchor bracket.')
    try:
        allowed_client={'schema_version','status','error_codes','clock','clock_contract','instrumented_only','instrumented_diagnostic','used_for_formal_performance','numerical_policy','raw_content_exported','requests','top_gaps'}
        if set(client_report)-allowed_client: raise ValueError('stall_client_unexpected_fields')
        required_contract={'schema_version':1,'status':'passed','instrumented_only':True,'instrumented_diagnostic':True,
                           'used_for_formal_performance':False,'numerical_policy':'record_only','raw_content_exported':False,'error_codes':[]}
        if any(type(client_report.get(k)) is not type(v) or client_report.get(k)!=v for k,v in required_contract.items()):
            raise ValueError('stall_client_contract')
        if not isinstance(client_report.get('clock_contract'),str) or 'CLOCK_MONOTONIC' not in client_report['clock_contract']:
            raise ValueError('stall_clock_contract')
        start=_number(window['started_monotonic'])*1e9; end=_number(window['deadline_monotonic'])*1e9
        if _span(start,end)<=0: raise ValueError('stall_invalid_window')
        requests=client_report['requests']
        if not isinstance(requests,list) or not 32<=len(requests)<=16000: raise ValueError('stall_client_request_coverage')
        request_by_index={}
        for request in requests:
            if set(request)!={'index','start_ns','first_ns','last_ns','count','finished','server_timing'}:
                raise ValueError('stall_client_request_schema')
            index=request['index']; count=request['count']
            if type(index) is not int or index<0 or index in request_by_index: raise ValueError('stall_request_index')
            if type(count) is not int or not 0<=count<=1024 or type(request['finished']) is not bool:
                raise ValueError('stall_request_count')
            if request['finished'] and count!=1024: raise ValueError('stall_request_finish')
            dispatched=_number(request['start_ns'])
            if not start-1<=dispatched<end: raise ValueError('stall_request_window')
            if count:
                first,last=_number(request['first_ns']),_number(request['last_ns'])
                if not dispatched<=first<=last<end: raise ValueError('stall_request_timestamp')
            elif request['first_ns'] is not None or request['last_ns'] is not None or request['finished']:
                raise ValueError('stall_request_empty_output')
            if not isinstance(request['server_timing'],dict): raise ValueError('stall_request_timing')
            for timestamp in request['server_timing'].values():
                if _number(timestamp)<=0: raise ValueError('stall_request_timing')
            request_by_index[index]=request
        for request in client_report.get('requests',[]):
            if 'index' in request and (type(request['index']) is not int or request['index']<0): raise ValueError('stall_request_index')
            for field in ('start_ns','first_ns','last_ns','count'):
                if request.get(field) is not None and _number(request[field])<0: raise ValueError('stall_request_numeric')
            if set(request)-{'index','start_ns','first_ns','last_ns','count','finished','server_timing'}: raise ValueError('stall_client_unexpected_fields')
            if set(request.get('server_timing',{}))-{'created_time','tokenize_finish_time','api_server_dispatch_finish_time','first_token_time','wait_queue_entry_time','forward_entry_time','prefill_finished_time'}: raise ValueError('stall_client_unexpected_timing')
        if len(rank_reports)!=4 or sorted(r['rank'] for r in rank_reports)!=list(range(4)):
            raise ValueError('stall_rank_coverage')
        if client_report.get('clock')!='CLOCK_MONOTONIC': raise ValueError('stall_clock_mismatch')
        if client_report.get('status','passed')!='passed': raise ValueError('stall_client_failed')
        start=_number(window['started_monotonic'])*1e9; end=_number(window['deadline_monotonic'])*1e9
        _span(start,end)
        identity=None; signatures=None; call_signatures=None; intervals={'prefill':[],'decode':[]}; uncertainties=[]
        aligned=[]; per_rank=[]
        for r in sorted(rank_reports,key=lambda r:r['rank']):
            if r.get('clock')!='CLOCK_MONOTONIC' or not r.get('source_verified'):
                raise ValueError('stall_rank_identity')
            ident=(r['source_sha256'],r['native_source_sha256'],r['identity'])
            if identity is not None and ident!=identity: raise ValueError('stall_source_mismatch')
            identity=ident
            if r.get('failures') or any(r.get('overflow',{}).values()): raise ValueError('stall_collector_failed')
            if len(r['calls'])>16000 or len(r['samples'])>256: raise ValueError('stall_report_overflow')
            if not r['calls'] or not r['samples']: raise ValueError('stall_empty_coverage')
            sig=[(s['sequence'],s['phase'],s['rows']) for s in r['samples']]
            sampled_indices=[c['sample_index'] for c in r['calls'] if c['sample_index'] is not None]
            if any(type(i) is not int for i in sampled_indices) or sampled_indices!=list(range(len(sig))):
                raise ValueError('stall_sample_bijection')
            if len({s['sequence'] for s in r['samples']})!=len(sig): raise ValueError('stall_duplicate_sample_sequence')
            csig=[(s['sequence'],s['phase'],s['rows'],s['sample_index']) for s in r['calls']]
            if signatures is not None and sig!=signatures: raise ValueError('stall_crossrank_sample_drift')
            if call_signatures is not None and csig!=call_signatures: raise ValueError('stall_crossrank_call_drift')
            signatures,call_signatures=sig,csig
            for phase in ('prefill','decode'):
                expected=sum(c['phase']==phase for c in r['calls'])
                if r.get('coverage',{}).get(phase)!=[expected]*43: raise ValueError('stall_full_layer_coverage')
            before=r['anchor']['host_before_ns']; after=r['anchor']['host_after_ns']
            uncertainties.append(_span(before,after)); mid=(before+after)/2
            layers={phase:{i:0 for i in range(43)} for phase in ('prefill','decode')}
            totals={'prefill':0,'decode':0}; clipped={'prefill':0,'decode':0}; residual={'prefill':0,'decode':0}
            decode_count=0
            for expected_sequence,call in enumerate(r['calls']):
                if type(call['sequence']) is not int or type(call['rows']) is not int or call['rows']<=0: raise ValueError('stall_call_numeric')
                if call['sequence']!=expected_sequence or call['phase'] not in ('prefill','decode'): raise ValueError('stall_call_sequence')
                expected_sample=call['phase']=='prefill' or decode_count%32==0
                if call['phase']=='decode': decode_count+=1
                if expected_sample != (call['sample_index'] is not None): raise ValueError('stall_sample_selection')
                _span(call['host_start_ns'],call['host_end_ns'])
                if call['phase']=='prefill' and call['sample_index'] is None: raise ValueError('stall_missing_prefill_sample')
                if call['sample_index'] is not None:
                    ix=call['sample_index']
                    if not isinstance(ix,int) or not 0<=ix<len(sig) or sig[ix]!=(call['sequence'],call['phase'],call['rows']): raise ValueError('stall_sample_link')
            for j,s in enumerate(r['samples']):
                if s['phase'] not in totals: raise ValueError('stall_invalid_phase')
                _span(s['gpu_start_ms'],s['gpu_end_ms'])
                duration=_number(s['gpu_duration_ms'])
                if [x['layer'] for x in s['routed']]!=list(range(43)): raise ValueError('stall_layer_coverage')
                if len(aligned)<=j: aligned.append(dict(sequence=s['sequence'],phase=s['phase'],rows=s['rows'],values=[]))
                if duration<=0 or s['rows']<=0: raise ValueError('stall_invalid_sample')
                aligned[j]['values'].append(duration); totals[s['phase']]+=duration
                model_a=mid+s['gpu_start_ms']*1e6; model_b=mid+s['gpu_end_ms']*1e6
                clipped[s['phase']]+=max(0,min(end,model_b)-max(start,model_a))/1e6
                routed_duration=0
                previous_end=s['gpu_start_ms']
                intervals[s['phase']].append((mid+s['gpu_start_ms']*1e6,mid+s['gpu_end_ms']*1e6))
                for layer in s['routed']:
                    if not previous_end<=_number(layer['start_ms'])<=_number(layer['end_ms'])<=s['gpu_end_ms']: raise ValueError('stall_routed_outside_model')
                    layer_duration=_number(layer['duration_ms'])
                    if layer_duration<0: raise ValueError('stall_invalid_interval')
                    previous_end=layer['end_ms']
                    layers[s['phase']][layer['layer']]+=layer_duration
                    routed_duration+=layer_duration
                residual[s['phase']]+=duration-routed_duration
            per_rank.append(dict(rank=r['rank'],sampled_phase_gpu_ms=totals,approximate_window_sampled_phase_gpu_ms=clipped,routed_layer_gpu_ms=layers,model_minus_routed_union_ms=residual))
        for row in aligned: row['gpu_ms']=_stats(row.pop('values'))
        stages={name:dict(durations_ms=[],missing=0) for name in ('queue','prefill','api_tokenization','api_dispatch','ttft')}
        pairs={'queue':('wait_queue_entry_time','forward_entry_time'),'prefill':('forward_entry_time','prefill_finished_time'),
               'api_tokenization':('created_time','tokenize_finish_time'),'api_dispatch':('tokenize_finish_time','api_server_dispatch_finish_time'),
               'ttft':('created_time','first_token_time')}
        if len(client_report.get('requests',[]))>16000 or len(client_report.get('top_gaps',[]))>64: raise ValueError('stall_client_overflow')
        for request in client_report.get('requests',[]):
            t=request.get('server_timing',{})
            for name,(a,b) in pairs.items():
                if a not in t or b not in t: stages[name]['missing']+=1
                else: stages[name]['durations_ms'].append(_span(t[a],t[b])*1000)
        gaps=[]
        for gap in client_report.get('top_gaps',[]):
            if set(gap)-{'index','start_ns','end_ns','delta_ns','after_token_position','before_token_position'}: raise ValueError('stall_gap_unexpected_fields')
            if type(gap['index']) is not int or gap['index']<0: raise ValueError('stall_gap_index')
            if set(gap)!={'index','start_ns','end_ns','delta_ns','after_token_position','before_token_position'}:
                raise ValueError('stall_gap_schema')
            request=request_by_index.get(gap['index'])
            if request is None or not request['count']: raise ValueError('stall_gap_request')
            a,b=gap['start_ns'],gap['end_ns']; duration=_span(a,b)
            if duration<=0 or _number(gap['delta_ns'])!=duration: raise ValueError('stall_gap_duration')
            if not start<=a<b<end or not request['first_ns']<=a<b<=request['last_ns']:
                raise ValueError('stall_gap_timestamp')
            after,before=gap['after_token_position'],gap['before_token_position']
            if type(after) is not int or type(before) is not int or not 1<=after<before<=request['count']:
                raise ValueError('stall_gap_position')
            a,b=max(a,start),min(b,end)
            if b<=a: continue
            gaps.append(dict(index=gap['index'],start_ns=a,end_ns=b,
                             approximate_prefill_overlap_ns=_overlap(intervals['prefill'],a,b),
                             approximate_sampled_decode_overlap_ns=_overlap(intervals['decode'],a,b)))
        phase_totals={phase:_stats([r['sampled_phase_gpu_ms'][phase] for r in per_rank]) for phase in ('prefill','decode')}
        layer_totals={phase:{i:_stats([r['routed_layer_gpu_ms'][phase][i] for r in per_rank]) for i in range(43)} for phase in ('prefill','decode')}
        clipped_totals={phase:_stats([r['approximate_window_sampled_phase_gpu_ms'][phase] for r in per_rank]) for phase in ('prefill','decode')}
        observability={name:('missing' if not value['durations_ms'] else 'partial' if value['missing'] else 'available') for name,value in stages.items()}
        out.update(request_stage_observability=observability,approximate_window_sampled_phase_gpu_ms=clipped_totals,totals_scope='Full recorded trace unless labeled approximate_window; CUDA mapping uses anchor midpoint. Model minus routed union subtracts summed direct durations of sequential nonoverlapping routed slots from direct model duration; it includes waits and launch starvation and may have small signed event rounding error. Anchor offsets are approximate and used only for window clipping and overlap.',sampled_phase_gpu_ms=phase_totals,routed_layer_gpu_ms=layer_totals,status='passed',aligned_samples=aligned,per_rank=per_rank,request_stages=stages,gap_overlap=gaps,
                   anchor_uncertainty_ns_by_rank=uncertainties,window_ns=dict(start=start,end=end),
                   coverage='All prefill; sampled decode only. Missing stages remain missing. Parallel ranks are never summed.')
    except (KeyError,TypeError,ValueError,IndexError) as error:
        code=str(error) if isinstance(error,ValueError) and str(error).startswith('stall_') else 'stall_malformed_report'
        out['error_codes']=[code]
    return out
