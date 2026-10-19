"""Source-bound paper-mechanism evidence, never promote an operator by name."""
import hashlib
import json
import math
from pathlib import Path
from .hopper_persistent import IMPLEMENTATION,MECHANISM_FIELDS,source_sha256,source_manifest,validate_profile_bank


def _load(path,files):
    if path.is_symlink():raise ValueError('paper evidence symlink forbidden')
    raw=path.read_bytes();files[path.as_posix()]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def _finite_metrics(metrics):
    if set(metrics)!={'original_hopper','fp8_oracle','bf16_oracle'}:raise ValueError('record-only references missing')
    for row in metrics.values():
        if row.get('finite') is not True:raise ValueError('nonfinite paper/reference')
        for key in ('relative_l2','max_abs'):
            v=row.get(key)
            if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('invalid numerical record')


def validate_paper_bundle(folder,identity):
    folder=Path(folder);files={};profiles={};source=source_sha256();activity={'stolen_tiles':0,'return_compute_overlap':0,'dispatch_compute_overlap':0}
    from scripts.run_c32_comparison import serving_source_manifest
    serving=_load(folder/'serving-source-manifest.json',files)
    if serving!=serving_source_manifest():raise ValueError('serving Python source manifest differs')
    serving_sha=hashlib.sha256(json.dumps(serving,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest=_load(folder/'paper-source-manifest.json',files)
    if manifest!=source_manifest():raise ValueError('paper source manifest differs')
    for stage in ('smoke','calibrate','checkpoint'):
        lifecycle=_load(folder/stage/'container-lifecycle.json',files)
        if lifecycle.get('exit_code')!=0 or lifecycle.get('termination_signal') is not None:raise ValueError('paper GPU container did not exit successfully')
        runtime=_load(folder/stage/'runtime.json',files)
        if runtime.get('runtime_status')!='passed' or runtime.get('resolved_image_id')!=identity['image_id']:
            raise ValueError('paper GPU runtime differs')
    for rank in range(4):
        for stage in ('smoke','calibrate','checkpoint'):
            row=_load(folder/stage/f'paper-{stage}-rank{rank}.json',files)
            expected=dict(status='passed',gpu_checks_status='passed',implementation=IMPLEMENTATION,rank=rank,world_size=4,
                numerical_policy='record_only',raw_content_exported=False,source_sha256=source,source_sha256_after=source,
                image_id=identity['image_id'],checkpoint_content_sha256=identity['checkpoint_content_sha256'],
                serving_source_sha256=serving_sha,serving_source_sha256_after=serving_sha)
            if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('paper GPU evidence incomplete or foreign')
            expected_shape=dict(hidden=128,intermediate=128,experts=32,capacity=64) if stage=='smoke' else dict(hidden=4096,intermediate=2048,experts=256,capacity=2048)
            if row.get('shape')!=expected_shape:raise ValueError('paper proof shape differs')
            cleanup=row.get('cleanup',{})
            if cleanup.get('status')!='complete' or cleanup.get('graphs_reset')!=cleanup.get('graphs_owned'):
                raise ValueError('paper cleanup incomplete')
            r=row['resources']
            if r.get('implementation')!=IMPLEMENTATION or r.get('active_blocks_per_sm')!=1 or r.get('threads_per_block')!=128:
                raise ValueError('resident Hopper compute proof required')
            if stage=='smoke':
                if [c.get('case') for c in row['cases']]!=['serial','dispatch_pipeline','pipeline','stealing','dynamic','dynamic_skew','graph_3','graph_4']:
                    raise ValueError('paper mechanism case coverage differs')
            if stage=='checkpoint':
                if row.get('profile_bank_sha256')!=profiles[str(rank)]:raise ValueError('checkpoint profile differs from calibrated service profile')
                expected_cases=[(layer,'decode',32) for layer in range(43)]+[(0,'prefill',128)]
                if [(c.get('layer'),c.get('phase'),c.get('batch')) for c in row['cases']]!=expected_cases:
                    raise ValueError('all43 real weights and prefill required')
            for case in row['cases']:
                m=case['mechanisms']
                if case.get('unique_worker_smids') is not True or case.get('routed_collectives')!=0 or m.get('calls')!=1:
                    raise ValueError('persistent execution evidence missing')
                if any(type(m.get(k)) is not int or m[k]<0 for k in MECHANISM_FIELDS):raise ValueError('invalid mechanism counters')
                if m['computed_tiles']<m['stolen_tiles'] or m['remote_unique']>m['remote_pairs']:raise ValueError('invalid work accounting')
                if stage=='smoke':
                    for key in activity:activity[key]+=m[key]
                    if case['case'] in ('dispatch_pipeline','pipeline','stealing') and m['k_sum']<=1:raise ValueError('controlled chunk pipeline not selected')
                    if case['case']=='dispatch_pipeline' and m['dispatch_compute_overlap']<1:raise ValueError('controlled dispatch overlap not observed')
                    if case['case']=='pipeline' and m['return_compute_overlap']<1:raise ValueError('controlled chunk return overlap not observed')
                    if case['case']=='stealing' and m['stolen_tiles']<1:raise ValueError('controlled bubble stealing not observed')
                    if case['case'].startswith(('dynamic','graph_')) and (m['dynamic_calls']!=1 or m['profile_candidates']<4):
                        raise ValueError('device dynamic selection not observed')
                if stage=='checkpoint':
                    _finite_metrics(case['numerical'])
                    scope={'original_hopper':{'rows':case['batch'],'scope':'full_global_batch'},
                           'fp8_oracle':{'rows':2,'global_rows':[0,1]},'bf16_oracle':{'rows':2,'global_rows':[0,1]}}
                    if case.get('numerical_coverage')!=scope:raise ValueError('explicit oracle coverage required')
            if stage in ('smoke','calibrate'):
                bank_path=folder/stage/f'paper-profile-bank-rank{rank}.json';bank=_load(bank_path,files)
                validate_profile_bank(bank,source_sha256=source,device_key=bank['device_key'],hidden_size=expected_shape['hidden'],
                    intermediate_size=expected_shape['intermediate'],experts=expected_shape['experts'],top_k=6,rank=rank,workers=r['grid_blocks'],capacity=expected_shape['capacity'],
                    image_id=identity['image_id'],checkpoint_content_sha256=identity['checkpoint_content_sha256'],runtime=bank['runtime'])
                expected=[(b['max_local_tokens'],c) for b in bank['profiles'] for c in b['candidates']]
                if len(row['calibration'])!=len(expected):raise ValueError('measured cost coverage missing')
                for record,(limit,candidate) in zip(row['calibration'],expected):
                    if record['max_local_tokens']!=limit or record['candidate']!=candidate:raise ValueError('cost row differs')
                    for key in ('compute_ms','communication_ms','padded_flops','communication_bytes'):
                        value=record[key]
                        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('nonpositive calibration measurement')
                    rates=[record['padded_flops']/(record['compute_ms']/1000),record['communication_bytes']/(record['communication_ms']/1000)]
                    if rates!=candidate[2:4]:raise ValueError('profile rates not derived from measurements')
                digest=hashlib.sha256(bank_path.read_bytes()).hexdigest()
                if row['profile_bank_sha256']!=digest:raise ValueError('loaded calibration profile differs')
                if stage=='calibrate':profiles[str(rank)]=digest
    if any(activity[k]<1 for k in activity):
        raise ValueError('controlled probes did not demonstrate stealing and chunk overlap; do not claim verified mechanisms')
    # Paths in the exported proof are portable relative paths.
    return dict(status='passed',method_identity=IMPLEMENTATION,method_status='kernel_mechanisms_verified',
        complete_weave_claim=False,numerical_policy='record_only',paper_source_sha256=source,
        profile_sha256=profiles,mechanism_activity=activity,
        files={str(Path(path).relative_to(folder)):digest for path,digest in files.items()})


def validate_paper_coverage(before,after,graph,identity,proof):
    def indexed(rows):
        if len(rows)!=4 or {r['rank'] for r in rows}!={0,1,2,3}:raise ValueError('four paper ranks required')
        return {r['rank']:r for r in rows}
    a,b=indexed(before),indexed(after);result={'status':'passed','method_identity':IMPLEMENTATION,
        'method_status':'paper_service_execution_verified','complete_weave_claim':False,'numerical_policy':'record_only','ranks':{}}
    for rank in range(4):
        x,y=a[rank],b[rank]
        for row in (x,y):
            expected=dict(method_identity=IMPLEMENTATION,source_sha256=proof['paper_source_sha256'],
                profile_bank_sha256=proof['profile_sha256'][str(rank)],schedule_mode='measured_dynamic',fixed_plan=None,
                decode_implementation='hopper_persistent',prefill_implementation='hopper_persistent',graph_enabled=True,
                additional_slot_collectives=0,additional_prefill_slot_collectives=0,hot_cache=True,kernel='tiled',
                graph_mode='full',numerical_policy='record_only',profile_checkpoint_content_sha256=identity['checkpoint_content_sha256'])
            if any(row.get(k)!=v for k,v in expected.items()) or set(row['layers'])!=set(map(str,range(43))):
                raise ValueError('paper service identity/implementation differs')
            mechanism=row['mechanism_evidence'];resources=row['paper_resources'];smids=mechanism['worker_smids']
            if (mechanism.get('fields')!=list(MECHANISM_FIELDS) or resources.get('active_blocks_per_sm')!=1
                or len(smids)!=resources.get('grid_blocks') or len(set(smids))!=len(smids)
                or any(type(s) is not int or s<0 for s in smids)
                or row.get('model_batch_counts',{}).get('active') is not False):
                raise ValueError('paper resident/idle evidence missing')
        counters={key:0 for key in MECHANISM_FIELDS};plans={}
        for layer in map(str,range(43)):
            old,new=x['layers'][layer],y['layers'][layer]
            delta=[n-o for o,n in zip(old['device_calls'],new['device_calls'])]
            if len(delta)!=3 or delta[0]<1 or delta[1]!=graph['replays'] or delta[2]!=0:raise ValueError('actual paper layer execution missing')
            if new['capture_calls']!=old['capture_calls'] or new['python_decode_calls']!=old['python_decode_calls']:
                raise ValueError('recapture or eager decode in paper window')
            if 'paper_persistent_prefill' not in new['implementation_calls'] or 'paper_persistent_decode' not in new['implementation_calls']:
                raise ValueError('paper implementation counters missing')
            for key in new['implementation_calls']:
                d=new['implementation_calls'][key]-old['implementation_calls'].get(key,0)
                if d!=(delta[0] if key=='paper_persistent_prefill' else 0):raise ValueError('mixed operator fallback')
            for phase,count in [('prefill',delta[0]),('decode',delta[1])]:
                prev=x['mechanism_evidence']['phases'][phase][layer];now=y['mechanism_evidence']['phases'][phase][layer]
                ds={key:now['counters'][key]-prev['counters'][key] for key in MECHANISM_FIELDS}
                if any(v<0 for v in ds.values()) or ds['calls']!=count or ds['dynamic_calls']!=count or ds['profile_candidates']<count*4:
                    raise ValueError('device scheduler coverage missing')
                for key,v in ds.items():counters[key]+=v
                oldplans={(v['c'],v['k']):v['calls'] for v in prev['plan_histogram']}
                differences={(v['c'],v['k']):v['calls']-oldplans.get((v['c'],v['k']),0) for v in now['plan_histogram']}
                if any(v<0 for v in differences.values()) or sum(differences.values())!=count:raise ValueError('paper actual plan histogram differs')
                for key,v in differences.items():plans[str(key)]=plans.get(str(key),0)+v
        result['ranks'][str(rank)]={'counters':counters,'actual_plan_histogram':plans,
            'stealing_observed':counters['stolen_tiles']>0,'chunk_overlap_observed':counters['return_compute_overlap']>0}
    return result
