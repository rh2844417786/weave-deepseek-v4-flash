"""Real native-weight tail calibration inside distributed capture_one warmup.

CUDA graphs with diagnostic events/timestamps are separate from the uninstrumented
full-MLP objective graph. Every candidate retains its own observed workload; no
cross-candidate routing equality gate exists. Absolute timestamps are compared
only within one rank. Finite error, regret and regression are record-only.
"""
from contextlib import contextmanager
import ast
import hashlib
import math
from pathlib import Path
import statistics
import types

from .calibration import atomic_json
from .native_cost_calibration import (private_arguments, _pattern, _measure,
    _numerical, _output_tensor, _tensor_digest, timing_summary, _event_overhead)
from .native_tail_cost import (METHOD, POLICY, SCALARS, STAGES, select_joint,
    profile_digest, validate_surface)
from .native_tail_front import _compile_function
from .native_tail_patch import source_pins


def _finite_nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('missing or invalid measured native tail ' + name)
    return float(value)


def _interval(a, b, name):
    a, b = _finite_nonnegative(a, name + ' start'), _finite_nonnegative(b, name + ' end')
    if a == 0 or b < a:
        raise ValueError('missing or reversed rank-local native tail interval: ' + name)
    return (b - a) / 1000.


def curve_from_snapshot(plan, snapshot, *, front_us):
    """Map observed same-GPU intervals to the documented approximate chunk DAG."""
    result = dict(plan=plan.to_dict(), features=list(snapshot['features']),
        front_us=_finite_nonnegative(front_us, 'front_us'),
        prepare_us=(_interval(snapshot.get('start_ns'), snapshot.get('planner_begin_ns'), 'features')
            + _interval(snapshot.get('planner_end_ns'), snapshot.get('planned_ns'), 'prepare join')),
        planner_us=_interval(snapshot.get('planner_begin_ns'), snapshot.get('planner_end_ns'), 'planner'),
        finish_us=_interval(snapshot.get('completion_begin_ns'), snapshot.get('finish_ns'), 'finish acknowledgement'),
        role_switch_us=_finite_nonnegative(snapshot.get('role_switch_ns'), 'role switch') / 1000.)
    times = snapshot['chunk_times_ns']
    if len(times) < plan.chunks:
        raise ValueError('missing measured chunk intervals')
    for stage, offset in zip(STAGES, (0, 2, 4, 6)):
        result[stage] = []
        for chunk in range(plan.chunks):
            row = times[chunk]
            if len(row) != 8:
                raise ValueError('malformed measured chunk intervals')
            if stage == 'down_us' and snapshot['local_groups'] == 0:
                # No down task exists. This is an observed empty task set, not
                # a missing timing replaced with a fabricated measurement.
                duration = 0.
            else:
                duration = _interval(row[offset], row[offset + 1], stage)
            result[stage].append(duration)
    return result


def workload_difference(reference, observed):
    return dict(features_delta=[float(b) - float(a) for a,b in zip(reference['features'],observed['features'])],
        routing_digest_changed=reference.get('native_routing_sha256') != observed.get('native_routing_sha256'),
        policy='record_only')


def prediction_record(predicted_us, fixed_actual_us, *, actual_selected_us):
    predicted = _finite_nonnegative(predicted_us, 'prediction')
    fixed = [_finite_nonnegative(v, 'fixed objective') for v in fixed_actual_us]
    selected = _finite_nonnegative(actual_selected_us, 'selected objective')
    if not fixed or min(fixed) <= 0 or selected <= 0:
        raise ValueError('complete positive full-MoE objective measurements required')
    return dict(predicted_us=predicted, actual_selected_us=selected,
        absolute_error_us=abs(predicted-selected), relative_error=abs(predicted-selected)/selected,
        best_measured_homogeneous_us=min(fixed), regret_us=selected-min(fixed),
        regret_fraction=(selected-min(fixed))/min(fixed), numerical_policy='record_only',
        regret_scope='selected_joint_plan_vs_measured_homogeneous_candidates')


def verify_current_plan(surfaces, snapshots, plans, total_rows):
    if len(snapshots) != 4:
        raise RuntimeError('GPU planner validation requires all four current replay workloads')
    predicted = select_joint(surfaces, [s['features'] for s in snapshots], plans, total_rows)
    for rank, state in enumerate(snapshots):
        if (state['chosen_indices'] != predicted['indices']
                or state['selected_index'] != predicted['indices'][rank]):
            raise RuntimeError('GPU planner differs from CPU specification on current replay workloads')
    return predicted


class _Points:
    def __init__(self, torch, names):
        self.torch = torch
        self.events = {n:torch.cuda.Event(enable_timing=True, external=True) for n in names}
        self.seen = set()

    def mark(self, name):
        if name in self.seen or name not in self.events:
            raise RuntimeError('repeated or unknown native tail diagnostic event: ' + name)
        self.seen.add(name)
        self.events[name].record(self.torch.cuda.current_stream())

    def us(self, start, end):
        if not {start,end}.issubset(self.seen):
            raise RuntimeError('native tail diagnostic path missed ' + start + '/' + end)
        return _finite_nonnegative(float(self.events[start].elapsed_time(self.events[end]))*1000., start+'/'+end)


def _observe_join(function, marker_name='_weave_mark'):
    """Instrument the one explicit join in the pinned native dual-stream method."""
    count = 0
    body = [ast.parse(f'{marker_name}("entry")').body[0]]
    for node in function.body:
        if (isinstance(node, ast.Expr) and ast.unparse(node.value) == 'current_stream.wait_stream(self.alt_stream)'):
            body.extend(ast.parse(f'{marker_name}("before_join")').body)
            body.append(node)
            body.extend(ast.parse(f'{marker_name}("after_join")').body)
            count += 1
        else:
            body.append(node)
    if count != 1:
        raise RuntimeError('native shared join diagnostic anchor differs')
    function.body = body
    function.decorator_list = []
    return function


@contextmanager
def _native_join_observer(runtime, mlp, points):
    source_path = 'srt/models/deepseek_v2.py'
    source = (Path(runtime.package_root) / source_path).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != source_pins()[source_path]:
        raise RuntimeError('native tail native-join source mismatch')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'DeepseekV2MoE')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_normal_dual_stream')
    function = _compile_function(_observe_join(method),
        dict(vars(runtime.front.native), _weave_mark=points.mark), '<weave-native-join-diagnostic>')
    original = mlp.forward_normal_dual_stream
    mlp.forward_normal_dual_stream = types.MethodType(function, mlp)
    try:
        yield
    finally:
        mlp.forward_normal_dual_stream = original


def _gather(torch, value, group):
    values = [None] * 4
    torch.distributed.all_gather_object(values, value, group=group)
    return values


def _route_record(down, snapshot, torch):
    padded = int(down.num_tokens_post_padded.item())
    if padded < 0 or padded > down.sorted_token_ids.numel() or padded % down.geometry.block_size_m:
        raise RuntimeError('native tail malformed padded routing')
    digest = hashlib.sha256()
    for tensor in (down.sorted_token_ids[:padded], down.local_expert_ids[:padded//down.geometry.block_size_m]):
        digest.update(bytes.fromhex(_tensor_digest(tensor, torch)))
    return dict(features=list(snapshot['features']), native_routing_sha256=digest.hexdigest())


def _median_curve(plan, samples):
    if not samples:
        raise RuntimeError('native tail missing diagnostic repeated measurements')
    totals = [sum(s[k] for k in SCALARS) + sum(sum(s[k]) for k in STAGES) for s in samples]
    middle = statistics.median(totals)
    index = min(range(len(samples)), key=lambda i:(abs(totals[i]-middle), i))
    # Keep one actual observation and its workload together. Component-wise
    # medians plus a preceding candidate's features would fabricate an anchor
    # when runtime grouping/workload differs across repetitions.
    return dict(samples[index]), index


def _checked_snapshot(engine, layer, batch, *, private=False):
    state = engine.snapshot_context(layer, batch, private=private)
    if state.get('protocol_valid') is not True or state.get('canary_ok') is not True:
        raise RuntimeError('native tail calibration protocol/coverage/ack evidence is invalid or missing')
    return state


def _measure_tail(graph, torch, repeats, engine, layer, batch):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay(); torch.cuda.synchronize()
    samples, states = [], []
    for _ in range(repeats):
        # Align ranks outside the timed interval after all preceding diagnostic
        # readbacks. The native reference uses the identical boundary.
        engine.dist.barrier(group=engine.group)
        start.record(); graph.replay(); end.record(); end.synchronize()
        samples.append(float(start.elapsed_time(end))*1000.)
        # Reads are outside the objective interval and occur only in calibration.
        states.append(_checked_snapshot(engine, layer, batch))
    return timing_summary(samples), states


def _measure_native(graph, torch, repeats, engine):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    graph.replay();torch.cuda.synchronize()
    samples=[]
    for _ in range(repeats):
        engine.dist.barrier(group=engine.group)
        start.record();graph.replay();end.record();end.synchronize()
        samples.append(float(start.elapsed_time(end))*1000.)
    return timing_summary(samples)


def calibrate_layer(runtime, layer, batch, original, args, kwargs):
    """Populate one structural-class/batch table from real native-weight graphs."""
    from sglang.srt.model_executor.runner import get_is_capture_mode
    torch, engine = runtime.torch, runtime.executor
    observer = getattr(runtime.model, 'weave_graph_observer', None)
    if (observer is None or not observer.capturing or torch.cuda.is_current_stream_capturing()
            or not get_is_capture_mode() or batch not in range(1, 33)):
        raise RuntimeError('native tail calibration requires native capture_one noncapturing decode warmup')
    group = runtime.layer_groups[layer]
    key = f'{group}:{batch}'
    completed = getattr(runtime, 'tail_calibration_completed', set())
    if key in completed:
        return
    if layer != runtime.groups[group]['representative_layer']:
        raise RuntimeError('native tail class was reached before its representative calibration')
    if getattr(runtime, 'tail_calibrating', False):
        raise RuntimeError('recursive native tail calibration')
    runtime.tail_calibration_completed = completed
    runtime.tail_calibrating = True
    bound = private_arguments(original, args, kwargs)
    hidden = bound.arguments['hidden_states']
    if hidden.ndim != 2 or hidden.shape[0] != batch:
        runtime.tail_calibrating = False
        raise RuntimeError('native tail calibration shape differs from exact decode batch')
    mlp = runtime.model.model.layers[layer].mlp
    config = getattr(mlp, 'config', getattr(runtime.model, 'config', None))
    vocab_size = int(getattr(config, 'vocab_size', 0))
    if vocab_size <= 0:
        runtime.tail_calibrating = False
        raise RuntimeError('native tail calibration requires loaded checkpoint vocabulary')
    for name in ('input_ids','input_ids_global'):
        if name in bound.signature.parameters and bound.arguments.get(name) is None:
            bound.arguments[name] = torch.zeros(batch, dtype=torch.int64, device=hidden.device)
    repeats = runtime.repeats
    if repeats < 2:
        runtime.tail_calibrating = False
        raise ValueError('native tail needs at least two measured repetitions')
    graphs, outputs = [], []
    stream = torch.cuda.Stream(device=hidden.device)
    previous_mode = None
    owner_count = None
    succeeded = False

    def native(_capturing):
        return runtime.invoke_native(layer, original, bound.args, bound.kwargs), None

    def tail(capturing, points=None):
        inputs = dict(bound.arguments)
        activation = inputs.pop('hidden_states')
        seen = []
        def invoke(down, shared, **kw):
            seen.append(down)
            return engine(down, shared, **kw)
        value = runtime.front(mlp, activation, invoke, **inputs,
                              observer=points if capturing else None)
        if len(seen) != 1:
            raise RuntimeError('native tail calibration requires one front and one tail')
        return value, seen[0]

    def capture(call):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):call(False)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(); graphs.append(graph)
        with torch.cuda.graph(graph, stream=stream):
            output, down = call(True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        outputs.append(output)
        return graph, _output_tensor(output, torch), down

    try:
        _pattern(bound, 0, batch, vocab_size, torch)
        tail(False)  # Allocate/compile the real geometry before any graph capture.
        ctx = engine.contexts[layer,batch]
        owner_count = len(ctx.owners)
        previous_mode = ctx.mode.clone()
        plans = list(ctx.plans)
        all_plans = _gather(torch, [p.to_dict() for p in plans], runtime.group)
        if any(p != all_plans[0] for p in all_plans):
            raise RuntimeError('native tail ranks disagree on executable candidate geometry')
        engine.set_fixed(layer,batch,0,diagnostic=False)
        full_graph, full_output, full_down = capture(tail)
        native_graph, native_output, _ = capture(native)
        new_points = _Points(torch, ('entry','before_join','after_join','tail_entry'))
        engine.set_fixed(layer,batch,0,diagnostic=True)
        diagnostic_graph, diagnostic_output, diagnostic_down = capture(lambda c:tail(c,new_points))
        native_points = _Points(torch, ('entry','before_join','after_join'))
        def native_diagnostic(capturing):
            if not capturing:
                return native(False)
            with _native_join_observer(runtime, mlp, native_points):
                return native(True)
        native_stage_graph, _, _ = capture(native_diagnostic)
        total_rows = full_down.geometry.pair_rows
        event_overhead=_event_overhead(torch,repeats)
        diagnostic_start,diagnostic_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        surface = [[] for _ in plans]
        anchors = []
        # Two spread and two repeated-row inputs, with real loaded router/expert
        # weights. Held-out seeds are distinct; no previous cost table is used.
        for anchor_index in (0,1,4,5):
            anchor = _pattern(bound,anchor_index,batch,vocab_size,torch)
            native_time = _measure_native(native_graph,torch,repeats,engine)
            native_join = []
            for _ in range(repeats):
                engine.dist.barrier(group=engine.group)
                native_stage_graph.replay();torch.cuda.synchronize()
                native_join.append(native_points.us('before_join','after_join'))
            anchor.update(native_full_moe=native_time,native_shared_join_wait_us=native_join,candidates=[])
            for index, plan in enumerate(plans):
                engine.set_fixed(layer,batch,index,diagnostic=False)
                full, full_states = _measure_tail(full_graph,torch,repeats,engine,layer,batch)
                if not bool(torch.isfinite(full_output).all()):
                    raise RuntimeError('nonfinite native tail full-MoE calibration output')
                objective_route = _route_record(full_down,full_states[-1],torch)
                engine.set_fixed(layer,batch,index,diagnostic=True)
                curves, diagnostic_records, joins, diagnostic_totals = [], [], [], []
                for _ in range(repeats):
                    engine.dist.barrier(group=engine.group)
                    diagnostic_start.record();diagnostic_graph.replay();diagnostic_end.record();diagnostic_end.synchronize()
                    diagnostic_totals.append(float(diagnostic_start.elapsed_time(diagnostic_end))*1000.)
                    state = _checked_snapshot(engine,layer,batch,private=True)
                    route = _route_record(diagnostic_down,state,torch)
                    curve = curve_from_snapshot(plan,state,front_us=new_points.us('entry','tail_entry'))
                    curves.append(curve)
                    joins.append(new_points.us('before_join','after_join'))
                    diagnostic_records.append(dict(workload=route,curve=curve,
                        rank_local_timestamps={k:v for k,v in state.items() if k.endswith('_ns')},
                        protocol_valid=True))
                chosen_curve, representative = _median_curve(plan,curves)
                chosen_route = diagnostic_records[representative]['workload']
                if index == 0:
                    reference_route = chosen_route
                surface[index].append(chosen_curve)
                anchor['candidates'].append(dict(plan=plan.to_dict(),full_moe=full,
                    objective_workload=objective_route,objective_replay_workloads=[
                        dict(features=s['features'],chosen_indices=s['chosen_indices']) for s in full_states],
                    workload=chosen_route,workload_difference=workload_difference(reference_route,chosen_route),
                    representative_observation=representative,
                    diagnostic_observations=diagnostic_records,new_shared_join_wait_us=joins,
                    diagnostic_full_moe=timing_summary(diagnostic_totals),
                    diagnostic_overhead_median_us=statistics.median(diagnostic_totals)-full['median_us'],
                    shared_join_wait_delta_us=statistics.median(joins)-statistics.median(native_join)))
            anchors.append(anchor)
        validate_surface(surface,plans,total_rows)
        table = dict(plans=[p.to_dict() for p in plans],total_rows=total_rows,surface=surface,
            executor_resources=ctx.resources, native_geometry=ctx.geometry,
            measurements=dict(complete=True,repeats=repeats,anchors=anchors,event_pair_overhead=event_overhead,
                objective='real_loaded_native_MLP_front_shared_join_tail_global_reduce',
                objective_graph_component_events=False,diagnostic_graph_separate=True,
                rank_alignment='CPU_group_barrier_before_each_timed_or_diagnostic_replay_not_in_event_interval',
                curve_estimator='actual_observation_nearest_median_local_stage_cost',
                pattern_protocol=dict(training_indices=[0,1,4,5],heldout_indices=[6,7],
                    training_layout='two_spread_two_repeated_rows'),
                timing_scope='per_rank_only_no_cross_GPU_absolute_clock_subtraction',
                role_switch_model='first_communication_after_both_tile_end_and_local_observed_readiness_excludes_ready_wait',
                prepare_model='feature_scan_and_post_planner_join_zeroing_overlaps_planner',
                routing_comparison_policy='record_only'),heldout=[])
        rank_tables = _gather(torch,table,runtime.group)
        engine.install_surfaces(group,batch,rank_tables)
        # The fixed calibration's planner interval measures fixed-plan exchange,
        # not search. Replace that component only with actual search execution
        # on this input, retaining its own workload and every measured sample.
        for position, anchor in enumerate(anchors):
            _pattern(bound,anchor['index'],batch,vocab_size,torch)
            engine.set_search(layer,batch,diagnostic=True)
            search_samples = []
            for _ in range(repeats):
                engine.dist.barrier(group=engine.group)
                diagnostic_graph.replay();torch.cuda.synchronize()
                state = _checked_snapshot(engine,layer,batch,private=True)
                search_samples.append(dict(planner_us=_interval(state.get('planner_begin_ns'),
                    state.get('planner_end_ns'),'measured search planner'),
                    workload=_route_record(diagnostic_down,state,torch),
                    chosen_indices=state['chosen_indices']))
            middle = statistics.median(s['planner_us'] for s in search_samples)
            selected_sample = min(range(repeats),key=lambda i:(abs(search_samples[i]['planner_us']-middle),i))
            selected_search = search_samples[selected_sample]
            for curve_anchors in surface:
                curve = curve_anchors[position]
                curve['fixed_planner_us'] = curve['planner_us']
                curve['planner_us'] = selected_search['planner_us']
            anchor['search_planner'] = dict(observations=search_samples,
                representative_observation=selected_sample,
                workload_difference=workload_difference(anchor['candidates'][0]['workload'],selected_search['workload']))
        table['measurements']['curve_estimator'] += '_with_separately_measured_search_planner'
        validate_surface(surface,plans,total_rows)
        rank_tables = _gather(torch,table,runtime.group)
        engine.install_surfaces(group,batch,rank_tables)
        surfaces = [t['surface'] for t in rank_tables]
        heldout = []
        for pattern_index in (6,7):
            item = _pattern(bound,pattern_index,batch,vocab_size,torch)
            native_time = _measure_native(native_graph,torch,repeats,engine)
            fixed, fixed_states = [], []
            for index, plan in enumerate(plans):
                engine.set_fixed(layer,batch,index,diagnostic=False)
                measured, states = _measure_tail(full_graph,torch,repeats,engine,layer,batch)
                if not bool(torch.isfinite(full_output).all()):
                    raise RuntimeError('nonfinite held-out native tail output')
                fixed.append(dict(plan=plan.to_dict(),full_moe=measured,
                    workload=_route_record(full_down,states[-1],torch)))
                fixed_states.append(states[-1])
            engine.set_search(layer,batch,diagnostic=False)
            measured, states = _measure_tail(full_graph,torch,repeats,engine,layer,batch)
            current_state = states[-1]
            current_route = _route_record(full_down,current_state,torch)
            rank_states = _gather(torch,current_state,runtime.group)
            cpu = verify_current_plan(surfaces,rank_states,plans,total_rows)
            actual = full_output.clone()
            native_graph.replay();torch.cuda.synchronize()
            numerics = _numerical(actual,native_output,torch)
            rank_measurements = _gather(torch,dict(selected=measured['median_us'],
                fixed=[v['full_moe']['median_us'] for v in fixed]),runtime.group)
            actual_selected = max(v['selected'] for v in rank_measurements)
            actual_fixed = [max(v['fixed'][i] for v in rank_measurements) for i in range(len(plans))]
            item.update(native_full_moe=native_time,fixed_candidates=fixed,selected_full_moe=measured,
                workload=current_route,workload_difference=workload_difference(fixed[0]['workload'],current_route),
                current_rank_workloads=[s['features'] for s in rank_states],
                chosen_indices=cpu['indices'],cpu_gpu_selection_agree=True,
                prediction=prediction_record(cpu['predicted_us'],actual_fixed,actual_selected_us=actual_selected),
                gpu_prediction_us=rank_states[0]['leader_prediction_us'],gpu_prediction_publisher_rank=0,
                native_numerical=numerics,
                numerical_policy='record_only',routing_comparison_policy='record_only')
            heldout.append(item)
        table['heldout'] = heldout
        rank_tables = _gather(torch,table,runtime.group)
        engine.install_surfaces(group,batch,rank_tables)
        if getattr(engine,'profile',None) is None:
            engine.profile = dict(method=METHOD,policy=POLICY,numerical_policy='record_only',
                status='in_progress',identity=engine.identity,groups=runtime.groups,tables={})
        engine.profile['tables'][key] = table
        completed.add(key)
        expected = {f'{g}:{b}' for g in runtime.groups for b in range(1,getattr(engine,'max_batch',32)+1)}
        engine.profile['status'] = 'complete' if completed == expected else 'in_progress'
        engine.profile['profile_sha256'] = profile_digest(engine.profile)
        atomic_json(Path(runtime.directory)/f'native-tail-rank{runtime.rank}.json',engine.profile)
        if engine.profile['status'] == 'complete':
            engine.profiles = {str(i):p for i,p in enumerate(_gather(torch,engine.profile,runtime.group))}
        succeeded = True
    finally:
        # Destroy only private calibration graphs after their streams complete.
        # Keep model/IPC workspaces and the native graph pool alive.
        torch.cuda.synchronize()
        for graph in graphs:graph.reset()
        torch.cuda.synchronize()
        if owner_count is not None:
            del ctx.owners[owner_count:]
        if succeeded:
            engine.set_search(layer,batch,diagnostic=False)
        elif previous_mode is not None:
            ctx.mode.copy_(previous_mode)
        outputs.clear();graphs.clear()
        runtime.tail_calibrating = False
