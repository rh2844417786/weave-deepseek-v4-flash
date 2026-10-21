"""Measure real native MoE CUDA graphs during SGLang's decode warmup.

No service request data is retained or exported. The supplied runtime enters the
original loaded MLP, preserving native router, shared stream, join and collective.
Only this calibration module creates timing events; formal replay does not.
"""
from contextlib import contextmanager
import copy
import hashlib
import inspect
import math
import statistics


def timing_summary(samples):
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('CUDA timing samples must be finite positive microseconds; rerun calibration')
    return dict(samples_us=values, median_us=statistics.median(values),
                mean_us=statistics.mean(values), stddev_us=statistics.pstdev(values),
                min_us=min(values), max_us=max(values))


def prediction_record(predicted, actual, selected):
    predicted, actual = list(map(float, predicted)), list(map(float, actual))
    if (len(predicted) != len(actual) or not actual or selected not in range(len(actual))
            or any(not math.isfinite(v) or v <= 0 for v in predicted + actual)):
        raise ValueError('prediction validation requires complete finite positive costs')
    best = min(actual)
    return dict(predicted_us=predicted, actual_us=actual, selected_index=selected,
                absolute_error_us=[abs(p - a) for p, a in zip(predicted, actual)],
                relative_error=[abs(p - a) / a for p, a in zip(predicted, actual)],
                selected_actual_us=actual[selected], best_actual_us=best,
                regret_us=actual[selected] - best,
                regret_fraction=(actual[selected] - best) / best,
                numerical_policy='record_only')


def require_planner_selection(prediction, selected, invalid_count):
    """Check exact algorithm execution, independently of prediction quality."""
    if selected not in range(len(prediction['predictions_us'])) or invalid_count != 0:
        raise RuntimeError('GPU planner produced an invalid measured plan')
    if selected != prediction['candidate_index']:
        raise RuntimeError('GPU planner choice differs from the deterministic CPU specification on the same float32 table')


def private_arguments(original, args, kwargs):
    """Bind by native names, isolate tensors, and avoid consuming the model allocator."""
    bound = inspect.signature(original).bind_partial(*args, **kwargs)
    if 'hidden_states' not in bound.arguments:
        raise RuntimeError('native calibration requires the original MLP hidden_states argument')
    bound.arguments['hidden_states'] = bound.arguments['hidden_states'].clone()
    for name in ('input_ids', 'input_ids_global'):
        value = bound.arguments.get(name)
        if value is not None:
            bound.arguments[name] = value.clone()
    if 'gemm_output_zero_allocator' in bound.signature.parameters:
        bound.arguments['gemm_output_zero_allocator'] = None
    forward_batch = bound.arguments.get('forward_batch')
    if forward_batch is not None:
        forward_batch = copy.copy(forward_batch)
        if getattr(forward_batch, 'input_ids', None) is not None:
            forward_batch.input_ids = forward_batch.input_ids.clone()
        bound.arguments['forward_batch'] = forward_batch
    return bound


def _tensor_digest(tensor, torch):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _output_tensor(value, torch):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise RuntimeError('native MLP output is neither a tensor nor a tensor-leading tuple')


def _numerical(actual, reference, torch):
    if actual.shape != reference.shape or not bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()):
        raise RuntimeError('native calibration output shape or finite-value check failed')
    delta = actual.double() - reference.double()
    relative = float((delta.norm() / reference.double().norm().clamp_min(1e-12)).item())
    maximum = float(delta.abs().max().item()) if delta.numel() else 0.
    if not math.isfinite(relative) or not math.isfinite(maximum):
        raise RuntimeError('nonfinite native calibration numerical metric')
    return dict(finite=True, relative_l2=relative, max_abs=maximum, numerical_policy='record_only')


class _StageEvents:
    """External events survive graph replay; each pair stays on its native stream."""
    def __init__(self, torch, required=('gate_up', 'down')):
        self.torch = torch
        self.required = set(required)
        self.events = {name: (torch.cuda.Event(enable_timing=True, external=True),
                              torch.cuda.Event(enable_timing=True, external=True))
                       for name in ('gate_up', 'down', 'shared', 'all_reduce')}
        self.seen = set()

    @contextmanager
    def span(self, role):
        if role not in self.events or role in self.seen:
            raise RuntimeError('unexpected or repeated calibration event span: ' + str(role))
        self.seen.add(role)
        start, end = self.events[role]
        start.record(self.torch.cuda.current_stream())
        try:
            yield
        finally:
            end.record(self.torch.cuda.current_stream())

    def read_us(self):
        if not self.required.issubset(self.seen):
            raise RuntimeError('calibration missed native path spans: ' + ', '.join(sorted(self.required - self.seen)))
        return {name: float(a.elapsed_time(b)) * 1000.
                for name, (a, b) in self.events.items() if name in self.seen}


def _measure(graph, torch, repeats, divisor=1):
    # The timing events are outside the graph, so objective graph has no events.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    graph.replay()
    torch.cuda.synchronize()
    for _ in range(repeats):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000. / divisor)
    return timing_summary(values)


def _event_overhead(torch, repeats):
    samples = []
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        a.record()
        b.record()
        b.synchronize()
        samples.append(float(a.elapsed_time(b)) * 1000.)
    if any(not math.isfinite(x) or x < 0 for x in samples):
        raise RuntimeError('invalid CUDA event measurement overhead')
    return dict(samples_us=samples, median_us=statistics.median(samples))


def _pattern(bound, index, batch, vocab_size, torch):
    """Eight independent deterministic activations; training includes repeated rows."""
    hidden = bound.arguments['hidden_states']
    repeated = index in (4, 5, 7)
    seed = 41983 + index * 104729 + batch * 1009
    generator = torch.Generator(device=hidden.device).manual_seed(seed)
    values = torch.randn((1 if repeated else batch, hidden.shape[1]),
                         generator=generator, device=hidden.device, dtype=torch.float32)
    values = (values * .125).to(hidden.dtype)
    hidden.copy_(values.expand_as(hidden) if repeated else values)
    ids = (torch.arange(batch, device=hidden.device, dtype=torch.int64) * 1543 + seed) % vocab_size
    if repeated:
        ids.fill_(seed % vocab_size)
    for name in ('input_ids', 'input_ids_global'):
        value = bound.arguments.get(name)
        if value is not None:
            if value.numel() != batch:
                raise RuntimeError('native calibration input IDs differ from exact decode batch')
            value.copy_(ids.reshape_as(value))
    forward_batch = bound.arguments.get('forward_batch')
    if forward_batch is not None and getattr(forward_batch, 'input_ids', None) is not None:
        if forward_batch.input_ids.numel() != batch:
            raise RuntimeError('native calibration forward batch input IDs differ from decode batch')
        forward_batch.input_ids.copy_(ids.reshape_as(forward_batch.input_ids))
    return dict(index=index, seed=seed, activation_pattern='repeated_row' if repeated else 'spread',
                input_sha256=_tensor_digest(hidden, torch), input_ids_sha256=_tensor_digest(ids, torch))


def _route_record(binding, plan, torch):
    args, kwargs = binding
    padded = int(args[12].item())
    block = int(kwargs['moe_block_size'])
    digest = hashlib.sha256()
    for tensor in (args[10][:padded], args[11][:padded // block]):
        digest.update(bytes.fromhex(_tensor_digest(tensor, torch)))
    cells = plan.cpu().tolist()
    return dict(features=[cells[4], cells[6], cells[3], cells[7], cells[8]],
                total_rows=cells[5], global_blocks=cells[2],
                native_routing_sha256=digest.hexdigest())


def calibrate_layer(runtime, layer, batch, original, args, kwargs):
    """Return measured 9×6×8 surface and independent full-MoE validation.

    Caller must already be in native get_is_capture_mode(), before active CUDA
    capture. All ranks execute these fixed, ordered loops, including collectives.
    No data-dependent candidate count or rank-local early acceptance is used.
    """
    from .native_cost import CANDIDATES, predict_cost, workload_difference
    from sglang.srt.model_executor.runner import get_is_capture_mode
    torch = runtime.torch
    if (torch.cuda.is_current_stream_capturing() or not get_is_capture_mode()
            or batch not in range(1, 33)):
        raise RuntimeError('native calibration requires noncapturing decode warmup for batch 1..32')
    repeats = int(runtime.repeats)
    if repeats < 2:
        raise ValueError('native calibration needs at least two repeated measurements')
    bound = private_arguments(original, args, kwargs)
    hidden = bound.arguments['hidden_states']
    if hidden.ndim != 2 or hidden.shape[0] != batch:
        raise RuntimeError('native calibration hidden shape differs from exact batch')
    mlp = runtime.model.model.layers[layer].mlp
    config = getattr(mlp, 'config', getattr(runtime.model, 'config', None))
    vocab_size = int(getattr(config, 'vocab_size', 0))
    if vocab_size <= 0:
        raise RuntimeError('native calibration needs loaded checkpoint vocab_size')
    for name in ('input_ids', 'input_ids_global'):
        if name in bound.signature.parameters and bound.arguments.get(name) is None:
            bound.arguments[name] = torch.zeros(batch, dtype=torch.int64, device=hidden.device)
    group = runtime.layer_to_group[layer]
    table = runtime.tables[group, batch]
    control, plan = runtime.controls[layer], runtime.plans[layer]
    previous_control = control.clone()
    graphs, outputs = [], []
    stream = torch.cuda.Stream(device=hidden.device)
    stream.wait_stream(torch.cuda.current_stream())
    required_spans = ['gate_up', 'down', 'all_reduce']
    if getattr(mlp, 'shared_experts', None) is not None:
        required_spans.append('shared')
    stage_events = _StageEvents(torch, required_spans)
    rows = [[[0.] * 8 for _ in range(6)] for _ in CANDIDATES]
    anchors = []
    captured_paths = set()

    def invoke(observer=None, original_native=False):
        result = runtime.invoke(layer, original, bound.args, bound.kwargs,
                                observer=observer, original_native=original_native)
        path = runtime.path_tags.get(layer)
        if path not in ('forward_normal', 'forward_normal_dual_stream'):
            raise RuntimeError('native calibration path is unsupported or unobserved: ' + str(path))
        captured_paths.add(path)
        if len(captured_paths) != 1:
            raise RuntimeError('native calibration changed forward path across captured graphs')
        return result

    def set_control(mode, pair=(8, 8)):
        control.copy_(torch.tensor([mode, *pair], device=control.device, dtype=torch.int32))

    def capture(call):
        # Allocate graph pools only after native JITs and communicators are warm.
        # Inputs/control can have been copied since this side stream was made.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                warm = call(False)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)  # Include partially captured graphs in teardown.
        with torch.cuda.graph(graph, stream=stream):
            output = call(True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        outputs.append(output)
        return graph, output

    try:
        set_control(1)
        _pattern(bound, 0, batch, vocab_size, torch)
        full_graph, full_output = capture(lambda _: invoke())
        full_bindings = list(runtime.bindings[layer])
        if len(full_bindings) != 2:
            raise RuntimeError('full native MLP did not bind both native Marlin stages')
        full_output = _output_tensor(full_output, torch)
        binding_args, binding_kwargs = full_bindings[0]
        total_rows = int(binding_kwargs['size_m']) * int(binding_kwargs['top_k'])
        planner_calls = 32

        def planner(_):
            for _ in range(planner_calls):
                runtime.candidate.weave_plan(binding_args[10], binding_args[11], binding_args[12],
                    binding_kwargs['moe_block_size'], total_rows, plan, table, control)

        planner_graph, _ = capture(planner)
        stage_graph, _ = capture(lambda capturing: invoke(observer=stage_events if capturing else None))
        native_graph, native_output = capture(lambda _: invoke(original_native=True))
        native_output = _output_tensor(native_output, torch)
        overhead = _event_overhead(torch, repeats)
        geometry = []
        for role_index, (role, (_, kw)) in enumerate(zip(('gate_up', 'down'), full_bindings)):
            state = runtime.states[layer, role_index].cpu().tolist()
            geometry.append(dict(role=role, **{name: kw.get(name) for name in
                ('size_m', 'size_n', 'size_k', 'moe_block_size', 'top_k', 'is_ep',
                 'mul_topk_weights', 'use_fp32_reduce', 'use_atomic_add')},
                 native_grid=state[2], n_tiles=state[6], k_tiles=state[7]))
        for anchor_index in range(6):
            anchor = _pattern(bound, anchor_index, batch, vocab_size, torch)
            anchor['candidates'] = []
            for candidate_index, pair in enumerate(CANDIDATES):
                set_control(1, pair)
                full = _measure(full_graph, torch, repeats)
                if not bool(torch.isfinite(full_output).all()):
                    raise RuntimeError('nonfinite full native MLP calibration output')
                route = _route_record(full_bindings[0], plan, torch)
                if candidate_index == 0:
                    anchor.update(route)
                fixed = _measure(planner_graph, torch, repeats, planner_calls)
                stages = {name: [] for name in stage_events.seen}
                for _ in range(repeats):
                    stage_graph.replay()
                    torch.cuda.synchronize()
                    for name, value in stage_events.read_us().items():
                        stages.setdefault(name, []).append(value)
                anchor['candidates'].append(dict(pair=list(pair), full_moe=full, fixed_planner=fixed,
                                                workload=route, workload_difference=workload_difference(anchor,route),
                                                stages={k: timing_summary(v) for k, v in stages.items()}))
                rows[candidate_index][anchor_index] = list(route['features']) + [
                    full['median_us'], fixed['median_us'], fixed['median_us']]
            anchors.append(anchor)
        # The real search kernel now has a fully measured surface. Its decision
        # may change after this overhead replacement, so publish the final table.
        table.copy_(torch.tensor(rows, device=table.device, dtype=torch.float32))
        for anchor_index, anchor in enumerate(anchors):
            _pattern(bound, anchor_index, batch, vocab_size, torch)
            set_control(1)
            full_graph.replay()
            torch.cuda.synchronize()
            anchor['search_workload'] = _route_record(full_bindings[0], plan, torch)
            anchor['search_workload_difference'] = workload_difference(anchor,anchor['search_workload'])
            set_control(0)
            search = _measure(planner_graph, torch, repeats, planner_calls)
            for candidate_index in range(len(CANDIDATES)):
                rows[candidate_index][anchor_index][7] = search['median_us']
                anchor['candidates'][candidate_index]['search_planner'] = search
                if rows[candidate_index][anchor_index][5] - rows[candidate_index][anchor_index][6] + rows[candidate_index][anchor_index][7] <= 0:
                    raise RuntimeError('nonpositive noise-corrected MoE cost; rerun measured calibration')
        table.copy_(torch.tensor(rows, device=table.device, dtype=torch.float32))
        # Bind validation and the persisted profile to the exact device values,
        # including float32 quantization of the CUDA-event measurement medians.
        rows = table.cpu().tolist()
        heldout = []
        for index in (6, 7):
            item = _pattern(bound, index, batch, vocab_size, torch)
            fixed_curves, adjusted = [], []
            for pair in CANDIDATES:
                set_control(1, pair)
                full = _measure(full_graph, torch, repeats)
                if not bool(torch.isfinite(full_output).all()):
                    raise RuntimeError('nonfinite heldout full native MLP output')
                route = _route_record(full_bindings[0], plan, torch)
                if not fixed_curves:
                    reference_route = route
                fixed = _measure(planner_graph, torch, repeats, planner_calls)
                set_control(0)
                search = _measure(planner_graph, torch, repeats, planner_calls)
                adjusted.append(full['median_us'] - fixed['median_us'] + search['median_us'])
                fixed_curves.append(dict(pair=list(pair), full_moe=full, fixed_planner=fixed, search_planner=search,
                                         workload=route, workload_difference=workload_difference(reference_route,route)))
            set_control(0)
            adaptive = _measure(full_graph, torch, repeats)
            # Check the planner against this cost replay's actual workload,
            # not the last fixed candidate's potentially different routing.
            route = _route_record(full_bindings[0], plan, torch)
            prediction = predict_cost(rows, route['features'], total_rows)
            actual_selection = int(plan[21].item())
            require_planner_selection(prediction, actual_selection, int(plan[20].item()))
            cost_output = full_output.clone()
            native_graph.replay()
            torch.cuda.synchronize()
            item.update(route, fixed_candidates=fixed_curves, cost_full_moe=adaptive,
                        cost_workload_difference=workload_difference(reference_route,route),
                        prediction=prediction_record(prediction['predictions_us'], adjusted, actual_selection),
                        native_numerical=_numerical(cost_output, native_output, torch),
                        selected_pair=list(CANDIDATES[actual_selection]),
                        cpu_selected_index=prediction['candidate_index'],
                        cpu_gpu_selection_agree=actual_selection == prediction['candidate_index'],
                        shares_local_route_with_training=any(route['native_routing_sha256'] == a['native_routing_sha256'] for a in anchors))
            heldout.append(item)
        return dict(rows=rows, measurements=dict(complete=True, units='microseconds', repeats=repeats,
            objective='unmodified_native_MLP_routing_shared_join_collective_post_add',
            objective_graph_component_events=False, geometry=geometry,
            captured_path=next(iter(captured_paths)), observed_stage_spans=sorted(stage_events.seen),
            routing_comparison_policy='record_only',
            event_overhead_us=overhead, planner_calls_per_timed_graph=planner_calls,
            pattern_protocol='six_native_routed_synthetic_v1_four_spread_two_repeated', anchors=anchors),
            validation=dict(complete=True, numerical_policy='record_only', routing_comparison_policy='record_only', heldout=heldout,
                candidate_comparison_scope='same_nominal_input_observed_routing_may_differ',
                scope='two_independent_native_routed_full_MLP_graph_patterns',
                collective_prediction_approximation='fixed-pair calibration versus rank-specific selected pairs',
                numerical_reference='unmodified_loaded_native_MLP_full_graph'))
    finally:
        # Never destroy the model's communicator or graph pool. Release only our
        # owned graphs while their tensor bindings/output references are alive.
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        torch.cuda.synchronize()
        control.copy_(previous_control)
        runtime.bindings.pop(layer, None)
        outputs.clear()
        graphs.clear()
