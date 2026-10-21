"""Measured full-MoE CTA-pair surface, strict artifact binding, and CPU oracle.

No model, routes, activation values or fabricated timing defaults are stored here.
Finite numerical differences in calibration are evidence only (record_only).
"""
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

from .native_budget import ROOT, source_pins

METHOD = 'native_marlin_measured_cta_v2'
POLICY_VERSION = 'measured_moe_surface_v2'
CANDIDATES = ((8,8),(8,4),(8,2),(4,8),(4,4),(4,2),(2,8),(2,4),(2,2))
PLAN_SIZE = 32
TABLE_SHAPE = (9,6,8)
FEATURES = ('local_rows','active_experts','local_blocks','sum_squared_rows','max_rows')
SOURCE_FILES = ('src/weave/native_cost.py','src/weave/native_cost_patch.py',
                'src/weave/native_cost_runtime.py','src/weave/native_cost_calibration.py',
                'src/weave/native_budget.py','src/weave/native_budget_patch.py',
                'csrc/native_cost.h','configs/native_budget_sources.json',
                'src/weave/graph_hooks.py','src/weave/graph_observer.py',
                'configs/sglang_graph_sources.json','configs/runtime.json',
                'timing_hooks/sitecustomize.py')


def native_cost_source_sha256():
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        path = ROOT / name
        if not path.is_file():
            raise ValueError('incomplete native cost source delivery: ' + name)
        digest.update(name.encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def _number(value):
    return type(value) in (float,int) and math.isfinite(value)


def _features(features, total_rows, context):
    if (not isinstance(features,(list,tuple)) or len(features)!=5
            or any(not _number(v) or v<0 for v in features)):
        raise ValueError(context + ': need five finite nonnegative workload features')
    rows,experts,blocks,squares,largest = features
    if (rows>total_rows or experts>min(64,rows) or blocks>rows or largest>rows
            or squares>rows*rows or ((rows==0) != (experts==blocks==squares==largest==0))):
        raise ValueError(context + ': workload features exceed measured EP4/B32 geometry')


def validate_table(table, total_rows):
    if type(total_rows) is not int or not 1<=total_rows<=192:
        raise ValueError('native cost surface requires 1..192 total routed rows')
    if not isinstance(table,(list,tuple)) or len(table)!=9:
        raise ValueError('native cost table must have shape 9 x 6 x 8')
    for index,candidate in enumerate(table):
        if not isinstance(candidate,(list,tuple)) or len(candidate)!=6:
            raise ValueError('native cost table must have shape 9 x 6 x 8')
        for anchor,row in enumerate(candidate):
            where=f'candidate {index} anchor {anchor}'
            if not isinstance(row,(list,tuple)) or len(row)!=8:
                raise ValueError('native cost table must have shape 9 x 6 x 8')
            _features(row[:5],total_rows,where)
            if any(not _number(v) or v<=0 for v in row[5:]):
                raise ValueError(where + ': measured times must be finite and positive')
            if row[5]-row[6]+row[7]<=0:
                raise ValueError(where + ': corrected full MoE time must be positive; repeat calibration')
    return table


def workload_difference(reference, observed):
    """Record cross-candidate variation; no equality or magnitude acceptance gate."""
    return dict(policy='record_only', reference_candidate_index=0,
                features_match=observed['features']==reference['features'],
                feature_delta=[b-a for a,b in zip(reference['features'],observed['features'])],
                native_routing_sha256_match=observed['native_routing_sha256']==reference['native_routing_sha256'],
                global_blocks_delta=observed['global_blocks']-reference['global_blocks'])


def predict_cost(table, features, total_rows):
    """Nearest-two normalized-L1 interpolation; deterministic anchor/pair ties.

    Exact duplicate anchors use the first two observations. Their measured costs
    are averaged, so duplicate patterns do not silently discard an observation.
    Each candidate owns its measured feature anchors; they need not equal
    another candidate's anchors. Calibrators persist the float32 upload roundtrip; both CPU and GPU perform
    cost interpolation in double from those stored values.
    """
    validate_table(table,total_rows)
    _features(features,total_rows,'prediction')
    scales=(total_rows,64,total_rows,total_rows**2,total_rows)
    predictions=[]; distances=[]
    for candidate in table:
        # Python 3.12+ sum() uses compensated floating-point accumulation.
        # Explicit additions preserve the device planner's five-term order.
        ranked=[]
        for anchor,row in enumerate(candidate):
            distance=0.
            for j in range(5):distance+=abs(features[j]-row[j])/scales[j]
            ranked.append((distance,anchor))
        nearest=sorted(ranked)[:2]
        (d0,a0),(d1,a1)=nearest
        c0=candidate[a0][5]-candidate[a0][6]+candidate[a0][7]
        c1=candidate[a1][5]-candidate[a1][6]+candidate[a1][7]
        value=(c0+c1)/2 if d1==0 else c0 if d0==0 else (c0*d1+c1*d0)/(d0+d1)
        predictions.append(value);distances.append(d0)
    selected=min(range(9),key=predictions.__getitem__)
    return dict(candidate_index=selected,pair=CANDIDATES[selected],predicted_us=predictions[selected],
                predictions_us=predictions,nearest_distance=distances[selected])


def profile_digest(profile):
    """Canonical content digest: self digest excluded, identity and samples included."""
    if not isinstance(profile,dict):raise ValueError('native cost profile must be an object')
    payload={k:v for k,v in profile.items() if k!='profile_sha256'}
    try:
        data=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
    except (TypeError,ValueError) as exc:
        raise ValueError('native cost profile must contain finite JSON values') from exc
    return hashlib.sha256(data.encode()).hexdigest()


def _match(actual, expected, path='identity'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict):raise ValueError('native cost identity mismatch: '+path)
        for key,value in expected.items():
            if key not in actual:raise ValueError('native cost identity missing: '+path+'.'+key)
            _match(actual[key],value,path+'.'+key)
    elif type(actual)!=type(expected) or actual!=expected:
        raise ValueError('native cost identity mismatch: '+path+'; regenerate with run_native_cost.py')


def _sha(value, length=64):
    return isinstance(value,str) and len(value)==length and all(c in '0123456789abcdef' for c in value)


def _f32(value):
    try:return struct.unpack('f',struct.pack('f',value))[0]
    except (OverflowError,struct.error) as exc:
        raise ValueError('native cost measured time cannot be represented by the float32 table') from exc


def _timing(stats, repeats, where):
    if not isinstance(stats,dict):raise ValueError(where+': timing samples missing')
    samples=stats.get('samples_us')
    if (not isinstance(samples,list) or len(samples)!=repeats
            or any(not _number(v) or v<=0 for v in samples)):
        raise ValueError(where+': complete finite positive timing samples required')
    expected=dict(median_us=statistics.median(samples),mean_us=statistics.mean(samples),
                  stddev_us=statistics.pstdev(samples),min_us=min(samples),max_us=max(samples))
    if any(stats.get(key)!=value for key,value in expected.items()):
        raise ValueError(where+': timing summaries disagree with raw samples')
    return stats['median_us']


def _measured_case(case, total, index, where):
    if not isinstance(case,dict) or case.get('index')!=index or case.get('total_rows')!=total:
        raise ValueError(where+': measured workload identity missing')
    _features(case.get('features'),total,where)
    if (type(case.get('global_blocks')) is not int or not 0<=case['global_blocks']<=total
            or type(case.get('seed')) is not int
            or case.get('activation_pattern')!=('repeated_row' if index in (4,5,7) else 'spread')
            or not all(_sha(case.get(k)) for k in ('native_routing_sha256','input_sha256','input_ids_sha256'))):
        raise ValueError(where+': workload evidence malformed')


def _measured_workload(workload, total, where):
    if not isinstance(workload,dict) or workload.get('total_rows')!=total:
        raise ValueError(where+': observed workload missing')
    _features(workload.get('features'),total,where)
    if (type(workload.get('global_blocks')) is not int or not 0<=workload['global_blocks']<=total
            or not _sha(workload.get('native_routing_sha256'))):
        raise ValueError(where+': observed workload malformed')


def _candidate_workloads(candidates, reference, total, where):
    for candidate in candidates:
        workload=candidate.get('workload')
        _measured_workload(workload,total,where)
        # Verify the record says what happened, never require a match.
        if candidate.get('workload_difference')!=workload_difference(reference,workload):
            raise ValueError(where+': workload difference record disagrees with observations')


def _measured_pairs(candidates, repeats, where, stages=False):
    if not isinstance(candidates,list) or len(candidates)!=9:
        raise ValueError(where+': all nine measured candidate pairs required')
    medians=[]
    for index,entry in enumerate(candidates):
        if not isinstance(entry,dict) or entry.get('pair')!=list(CANDIDATES[index]):
            raise ValueError(where+': measured candidate order differs')
        medians.append([_timing(entry.get(key),repeats,where+'/'+key)
                        for key in ('full_moe','fixed_planner','search_planner')])
        if medians[-1][0]-medians[-1][1]+medians[-1][2]<=0:
            raise ValueError(where+': nonpositive corrected measured full MoE time')
        if stages:
            spans=entry.get('stages')
            if not isinstance(spans,dict) or not {'gate_up','down','all_reduce'}.issubset(spans):
                raise ValueError(where+': complete native stage measurements required')
            for name,stats in spans.items():
                if name not in ('gate_up','down','shared','all_reduce'):
                    raise ValueError(where+': unexpected stage measurement')
                _timing(stats,repeats,where+'/'+name)
    return medians


def _validate_measurements(entry, batch):
    measured=entry.get('measurements',{})
    if (not isinstance(measured,dict) or measured.get('complete') is not True
            or measured.get('units')!='microseconds' or type(measured.get('repeats')) is not int
            or measured['repeats']<1 or measured.get('objective_graph_component_events') is not False
            or measured.get('objective')!='unmodified_native_MLP_routing_shared_join_collective_post_add'
            or measured.get('pattern_protocol')!='six_native_routed_synthetic_v1_four_spread_two_repeated'
            or measured.get('routing_comparison_policy')!='record_only'
            or type(measured.get('planner_calls_per_timed_graph')) is not int
            or measured['planner_calls_per_timed_graph']<1):
        raise ValueError('native cost complete measured full-MoE protocol missing')
    repeats=measured['repeats'];total=batch*6
    overhead=measured.get('event_overhead_us',{})
    samples=overhead.get('samples_us') if isinstance(overhead,dict) else None
    if (not isinstance(samples,list) or len(samples)!=repeats
            or any(not _number(v) or v<0 for v in samples)
            or overhead.get('median_us')!=statistics.median(samples)):
        raise ValueError('native cost event overhead samples missing or malformed')
    geometry=measured.get('geometry')
    if not isinstance(geometry,list) or len(geometry)!=2:
        raise ValueError('native cost exact two-role GEMM geometry missing')
    for index,role in enumerate(('gate_up','down')):
        shape=geometry[index]
        if (not isinstance(shape,dict) or shape.get('role')!=role or shape.get('is_ep') is not True
                or shape.get('size_m')!=batch*(6 if index else 1) or shape.get('top_k')!=(1 if index else 6)
                or any(type(shape.get(k)) is not int or shape[k]<1 for k in
                       ('size_m','top_k','size_n','size_k','moe_block_size','native_grid','n_tiles','k_tiles'))
                or shape['moe_block_size'] not in (8,16,32,48,64)
                or shape['size_n']%shape['n_tiles'] or shape['size_k']%shape['k_tiles']):
            raise ValueError('native cost exact role geometry/native grid missing or malformed')
    anchors=measured.get('anchors')
    if not isinstance(anchors,list) or len(anchors)!=6:
        raise ValueError('native cost six measured workload anchors required')
    for index,anchor in enumerate(anchors):
        where=f'native cost anchor {index}'
        _measured_case(anchor,total,index,where)
        medians=_measured_pairs(anchor.get('candidates'),repeats,where,stages=True)
        _candidate_workloads(anchor['candidates'],anchor,total,where)
        search_workload=anchor.get('search_workload')
        _measured_workload(search_workload,total,where+'/search_planner')
        if anchor.get('search_workload_difference')!=workload_difference(anchor,search_workload):
            raise ValueError(where+': search workload difference record disagrees with observations')
        for candidate in range(9):
            row=entry['rows'][candidate][index]
            if row!=anchor['candidates'][candidate]['workload']['features']+[_f32(v) for v in medians[candidate]]:
                raise ValueError(where+': table is not the float32 measured features/medians')
    validation=entry.get('validation',{})
    if (not isinstance(validation,dict) or validation.get('complete') is not True
            or validation.get('numerical_policy')!='record_only'
            or validation.get('routing_comparison_policy')!='record_only'):
        raise ValueError('native cost heldout validation incomplete')
    heldout=validation.get('heldout')
    if not isinstance(heldout,list) or len(heldout)!=2:
        raise ValueError('native cost two independently measured heldout cases required')
    for index,case in enumerate(heldout,6):
        where=f'native cost heldout {index}'
        _measured_case(case,total,index,where)
        medians=_measured_pairs(case.get('fixed_candidates'),repeats,where)
        reference=case['fixed_candidates'][0].get('workload')
        _measured_workload(reference,total,where)
        _candidate_workloads(case['fixed_candidates'],reference,total,where)
        if case.get('cost_workload_difference')!=workload_difference(reference,case):
            raise ValueError(where+': cost workload difference record disagrees with observations')
        _timing(case.get('cost_full_moe'),repeats,where+'/cost_full_moe')
        prediction=case.get('prediction',{})
        selected=prediction.get('selected_index') if isinstance(prediction,dict) else None
        if (type(selected) is not int or not 0<=selected<9
                or prediction.get('numerical_policy')!='record_only'
                or case.get('selected_pair')!=list(CANDIDATES[selected])):
            raise ValueError(where+': heldout prediction/selection missing')
        for key in ('predicted_us','actual_us','absolute_error_us','relative_error'):
            values=prediction.get(key)
            if (not isinstance(values,list) or len(values)!=9
                    or any(not _number(v) or v<0 or (key.endswith('_us') and key!='absolute_error_us' and v==0)
                           for v in values)):
                raise ValueError(where+': complete finite heldout prediction/error vectors required')
        for key in ('selected_actual_us','best_actual_us','regret_us','regret_fraction'):
            if not _number(prediction.get(key)) or prediction[key]<0:
                raise ValueError(where+': complete finite heldout regret evidence required')
        expected=predict_cost(entry['rows'],case['features'],total)
        expected_index=expected['candidate_index']
        if (selected!=expected_index or type(case.get('cpu_selected_index')) is not int
                or case['cpu_selected_index']!=expected_index or case.get('cpu_gpu_selection_agree') is not True):
            raise ValueError(where+': deterministic CPU/GPU planner selection disagrees with measured table')
        # These are evidence-consistency checks, not accuracy/quality cutoffs.
        # Every finite error and regret remains admissible when it is the value
        # obtained from this table and these independent measured repetitions.
        predicted=list(map(float,expected['predictions_us']))
        actual=[float(full-fixed+search) for full,fixed,search in medians]
        best=min(actual)
        errors=[abs(p-a) for p,a in zip(predicted,actual)]
        derived=dict(predicted_us=predicted,actual_us=actual,absolute_error_us=errors,
                     relative_error=[error/a for error,a in zip(errors,actual)],
                     selected_actual_us=actual[selected],best_actual_us=best,
                     regret_us=actual[selected]-best,regret_fraction=(actual[selected]-best)/best)
        for key,value in derived.items():
            if prediction[key]!=value:
                raise ValueError(where+': '+key+' disagrees with measured table and heldout timing samples')
        numerical=case.get('native_numerical',{})
        if (not isinstance(numerical,dict) or numerical.get('finite') is not True
                or numerical.get('numerical_policy')!='record_only'
                or any(not _number(numerical.get(k)) or numerical[k]<0 for k in ('relative_l2','max_abs'))):
            raise ValueError(where+': finite native numerical evidence required (record_only)')


def validate_profile(profile, expected_identity=None):
    """Validate delivery, full class/batch coverage and requested exact identity.

    expected_identity is a recursively compared subset, allowing the controller
    to bind its known deployment fields and the runtime to bind the full object.
    No prediction-error, numerical-error, or throughput cutoff is imposed.
    """
    if not isinstance(profile,dict):raise ValueError('native cost profile must be an object')
    if profile.get('schema_version')!=1 or type(profile.get('schema_version')) is not int:
        raise ValueError('unsupported native cost profile schema_version')
    for key,value in (('method',METHOD),('policy_version',POLICY_VERSION),('numerical_policy','record_only')):
        if profile.get(key)!=value:raise ValueError('native cost profile '+key+' mismatch')
    if profile.get('calibration_complete') is not True:
        raise ValueError('native cost calibration is incomplete; run run_native_cost.py')
    if profile.get('profile_sha256')!=profile_digest(profile):
        raise ValueError('native cost profile digest mismatch; regenerate with run_native_cost.py')
    identity=profile.get('identity')
    if not isinstance(identity,dict):raise ValueError('native cost identity missing')
    for key in ('cost_source_sha256','checkpoint_content_sha256','model_config_sha256'):
        if not _sha(identity.get(key)):raise ValueError('native cost identity invalid '+key)
    if not _sha(identity.get('sglang_commit'),40):raise ValueError('native cost identity invalid sglang_commit')
    image=identity.get('image_id','')
    if not isinstance(image,str) or not _sha(image.removeprefix('sha256:')):
        raise ValueError('native cost identity invalid image_id')
    upstream=identity.get('upstream_sha256')
    if not isinstance(upstream,dict) or not upstream or not all(_sha(v) for v in upstream.values()):
        raise ValueError('native cost identity missing upstream source hashes')
    if type(identity.get('rank')) is not int or not 0<=identity['rank']<4:
        raise ValueError('native cost identity invalid EP4 rank')
    gpu=identity.get('gpu',{})
    if (not isinstance(gpu,dict) or not isinstance(gpu.get('name'),str) or not gpu['name']
            or gpu.get('capability')!=[9,0]
            or any(type(gpu.get(k)) is not int or gpu[k]<=0 for k in ('sm_count','total_memory'))):
        raise ValueError('native cost identity missing Hopper GPU properties')
    runtime=identity.get('runtime',{})
    if not isinstance(runtime,dict) or any(not isinstance(runtime.get(k),str) or not runtime[k]
            for k in ('python','torch','cuda','sglang','sgl_kernel')):
        raise ValueError('native cost identity missing runtime versions')
    _match(identity,dict(precision=dict(activations='bfloat16',routed_weights='mxfp4_e8m0',shared_weights='fp8'),
                        model_geometry=dict(hidden_size=4096,intermediate=2048,experts=256,top_k=6,layers=43),
                        graph_contract=dict(tp=4,ep=4,moe_tp=1,a2a_backend='none',decode_batches=list(range(1,33)))))
    if expected_identity is not None:_match(identity,expected_identity)
    groups=profile.get('groups');tables=profile.get('tables')
    if not isinstance(groups,dict) or not groups or not isinstance(tables,dict) or set(groups)!=set(tables):
        raise ValueError('native cost class/table coverage mismatch')
    partition=[]
    for name,group in groups.items():
        if (not isinstance(name,str) or not isinstance(group,dict)
                or not isinstance(group.get('layers'),list) or not group['layers']
                or any(type(layer) is not int for layer in group['layers'])
                or group.get('representative_layer') not in group['layers']
                or not isinstance(group.get('signature'),dict) or not group['signature']):
            raise ValueError('native cost structural class malformed: '+str(name))
        partition+=group['layers']
        cases=tables[name]
        if not isinstance(cases,dict) or set(cases)!={str(i) for i in range(1,33)}:
            raise ValueError('native cost batch coverage must be exactly 1..32 for '+name)
        for batch,entry in cases.items():
            if not isinstance(entry,dict):raise ValueError('malformed native cost table entry')
            validate_table(entry.get('rows'),int(batch)*6)
            _validate_measurements(entry,int(batch))
    if sorted(partition)!=list(range(43)):
        raise ValueError('native cost structural groups must partition all 43 layers exactly once')
    return profile


def load_profile(path, expected_identity=None):
    try:
        profile=json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError('native cost profile unavailable or malformed: '+str(path)+
                         '; generate it with scripts/run_native_cost.py') from exc
    return validate_profile(profile,expected_identity)
