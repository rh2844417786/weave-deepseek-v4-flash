"""Aggregate rank-local diagnostic intervals; never a formal-window profiler."""
from collections import defaultdict
import math
import statistics


def _merged(intervals):
    out=[]
    for begin,end in sorted(intervals):
        if not (0<begin<=end):raise ValueError('invalid diagnostic task interval')
        if begin==end:continue
        if out and begin<=out[-1][1]:out[-1]=(out[-1][0],max(end,out[-1][1]))
        else:out.append((begin,end))
    return out


def interval_overlap_us(left,right):
    left,right=_merged(left),_merged(right);i=j=0;duration=0
    while i<len(left) and j<len(right):
        a,b=left[i],right[j];duration+=max(0,min(a[1],b[1])-max(a[0],b[0]))
        if a[1]<=b[1]:i+=1
        else:j+=1
    return duration/1000.


def distribution(values):
    values=[float(v) for v in values if v is not None]
    if not values:return None
    if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite timing observation')
    return dict(count=len(values),min=min(values),median=statistics.median(values),max=max(values))


def publication_lags(snapshot,chunks):
    """Signed delays on ONE GPU. Old reports only have finalize-end timestamps."""
    times=snapshot['chunk_times_ns'][:chunks]
    ends=[] if snapshot.get('local_groups')==0 else [row[1] for row in times if row[1]>0]
    last=max(ends) if ends else None
    comm=[row[i] for row in times for i in (4,6) if 0<row[i]<(1<<64)-1]
    published=snapshot.get('chunk_published_ns',[None])[0]
    return dict(chunk0_finalize_end_after_last_down_us=(times[0][3]-last)/1000. if last else None,
        chunk0_publish_after_last_down_us=(published-last)/1000. if last and published else None,
        first_peer_task_after_last_down_us=(min(comm)-last)/1000. if last and comm else None)


def snapshot_timing(snapshot,*,chunks,tiles,capacity):
    events=snapshot['events'];later=[];early=[]
    for task,(_,begin,end,kind) in enumerate(events[:capacity*tiles]):
        if kind==1 and task%tiles>=tiles//chunks:later.append((begin,end))
    for task,(_,begin,end,kind) in enumerate(events[capacity*tiles:capacity*tiles+5*chunks]):
        chunk=task if task<chunks else (task-chunks)//4
        if kind in (2,3) and chunk==0:early.append((begin,end))
    return dict(measurement_scope='same_gpu_diagnostic_replay',time_unit='us',
        planner_us=(snapshot['planner_end_ns']-snapshot['planner_begin_ns'])/1000.,
        prepare_us=((snapshot['planner_begin_ns']-snapshot['start_ns'])+
                    (snapshot['planned_ns']-snapshot['planner_end_ns']))/1000.,
        finish_us=(snapshot['finish_ns']-snapshot['completion_begin_ns'])/1000.,
        chunk0_communication_overlap_us=interval_overlap_us(later,early),
        **publication_lags(snapshot,chunks))


def summarize_cases(cases):
    groups=defaultdict(list)
    for case in cases:
        if case.get('timing'):
            groups[f"B{case['batch']}/{case['mode']}/{case['pattern']}"].append(case['timing'])
    return {name:dict(cases=len(rows),**{key:distribution(row.get(key) for row in rows)
                for key in rows[0] if key.endswith('_us')}) for name,rows in sorted(groups.items())}


def summarize_profiles(profiles,batch=32):
    """Read existing calibration; no extra GPU work and no invented phase times."""
    rows=[]
    for rank,profile in sorted(profiles.items()):
        for key,table in sorted(profile['tables'].items()):
            if not key.endswith(':'+str(batch)):continue
            anchors=table['measurements']['anchors'];candidates=[]
            for index,plan in enumerate(table['plans']):
                observed=[c['diagnostic_observations'] for a in anchors for i,c in enumerate(a['candidates']) if i==index]
                observed=[o for samples in observed for o in samples]
                curves=[o['curve'] for o in observed]
                lags=[publication_lags(o['rank_local_timestamps'],plan['chunks']) for o in observed
                      if o.get('rank_local_timestamps',{}).get('chunk_times_ns')]
                result=dict(index=index,plan=plan,diagnostic_samples=len(curves),
                    full_moe_median_us=distribution(a['candidates'][index]['full_moe']['median_us'] for a in anchors))
                for name in ('front_us','prepare_us','finish_us','planner_us'):
                    result['fixed_'+name if name=='planner_us' else name]=distribution(c[name] for c in curves)
                for name in ('down_us','finalize_us','fetch_reduce_us','fetch_result_us'):
                    result[name]=[distribution(c[name][j] for c in curves) for j in range(plan['chunks'])]
                for name in ('chunk0_finalize_end_after_last_down_us','chunk0_publish_after_last_down_us','first_peer_task_after_last_down_us'):
                    result[name]=distribution(row[name] for row in lags)
                candidates.append(result)
            heldout=[]
            for h in table['heldout']:
                fixed=[c['full_moe']['median_us'] for c in h['fixed_candidates']]
                heldout.append(dict(index=h['index'],native_us=h['native_full_moe']['median_us'],
                    selected_us=h['selected_full_moe']['median_us'],best_fixed_us=min(fixed),
                    best_fixed_index=min(range(len(fixed)),key=fixed.__getitem__),
                    chosen_indices=h['chosen_indices'],prediction=h['prediction']))
            rows.append(dict(rank=int(rank),table=key,resources=table['executor_resources'],
                search_planner_us=distribution(o['planner_us'] for a in anchors for o in a['search_planner']['observations']),
                candidates=candidates,heldout=heldout))
    return dict(schema_version=1,time_unit='us',batch=batch,numerical_policy='record_only',
        timing_scope='existing_same_GPU_diagnostic_replays_and_separate_uninstrumented_MLP_objectives',
        limitations=['Do not sum overlapping chunk spans or times across ranks.',
            'Calibration patterns are not a decomposition of the formal GPQA window.',
            'Old first-peer intervals include successful-claim polling; v2 starts after the claim.',
            'Old reports have no chunk publication timestamp; unavailable values stay null.'],
        profiles={str(r):dict(profile_sha256=p['profile_sha256'],source_sha256=p['identity']['tail_source_sha256']) for r,p in profiles.items()},rows=rows)
