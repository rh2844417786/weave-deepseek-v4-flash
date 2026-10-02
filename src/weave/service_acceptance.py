"""Evidence checks for an idle, batch-one SGLang graph acceptance request."""
import math


def validate_checkpoint_gate(runtime, ranks, *, source_sha256, index_sha256, checkpoint_sha256):
    if (runtime.get('runtime_status') != 'passed' or runtime.get('checkpoint_validation_status') != 'passed'
            or runtime.get('child_exit_code') != 0 or not runtime.get('resolved_image_id')):
        raise ValueError('current runtime/checkpoint gate must have passed')
    if runtime.get('kernel_source_sha256') != source_sha256:
        raise ValueError('runtime source evidence is stale')
    if len(ranks) != 4 or {r['rank'] for r in ranks} != set(range(4)):
        raise ValueError('four checkpoint rank reports required')
    for r in ranks:
        if (r.get('status') != 'checkpoint_checks_passed' or r.get('layers') != list(range(43))
                or r.get('kernels') != ['scalar', 'wmma'] or r.get('architecture') != [4096, 2048, 256, 6, 43]):
            raise ValueError('all 43 real layers and both kernels must have passed')
        if (r.get('source_sha256') != source_sha256 or r.get('checkpoint_index_sha256') != index_sha256
                or r.get('checkpoint_content_sha256') != checkpoint_sha256 or not checkpoint_sha256):
            raise ValueError('checkpoint evidence is stale for this source/checkpoint')
        cases = {(v['layer'], v['kernel'], v['case']): v for v in r['cases']}
        expected = {(layer, kernel, case) for layer in [-1]+list(range(43))
                    for kernel in ['scalar', 'wmma']
                    for case in ['remote_K2', 'steal_K2', 'graph_replay0', 'graph_replay1']}
        if set(cases) != expected or len(r['cases']) != len(expected):
            raise ValueError('checkpoint cases are incomplete or duplicated')
        for v in cases.values():
            error = v['metrics']['relative_l2']
            if not math.isfinite(error) or error > .025:
                raise ValueError('checkpoint numerical gate failed')
    for layer in [-1]+list(range(43)):
        for kernel in ['scalar', 'wmma']:
            stolen = sum(v['completed_stolen'] for r in ranks for v in r['cases']
                         if v['layer'] == layer and v['kernel'] == kernel and v['case'] == 'steal_K2')
            if stolen < 1:
                raise ValueError('completed nonzero stealing is required')
    return {'status': 'passed', 'image_id': runtime['resolved_image_id'],
            'source_sha256': source_sha256, 'checkpoint_index_sha256': index_sha256,
            'checkpoint_content_sha256': checkpoint_sha256}


def validate_coverage_delta(before, after, *, require_graph):
    def ranks(items):
        if len(items) != 4 or {v['rank'] for v in items} != set(range(4)):
            raise ValueError('four distinct EP ranks required')
        return {v['rank']: v for v in items}
    before, after = ranks(before), ranks(after)
    evidence = {}
    identity = (before[0].get('kernel'), before[0].get('source_sha256'))
    if identity[0] not in ('scalar', 'wmma') or not identity[1]:
        raise ValueError('kernel/source provenance required')
    for rank in range(4):
        a, b = before[rank], after[rank]
        if (a.get('kernel'), a.get('source_sha256')) != identity:
            raise ValueError('EP ranks have different kernel/source provenance')
        expected = {str(i) for i in range(43)}
        if set(a['layers']) != expected or set(b['layers']) != expected:
            raise ValueError('all 43 layers are required on every rank')
        if (a.get('kernel'), a.get('graph_enabled'), a.get('source_sha256')) != (
                b.get('kernel'), b.get('graph_enabled'), b.get('source_sha256')):
            raise ValueError('backend provenance changed during request')
        if require_graph and not b.get('graph_enabled'):
            raise ValueError('graph-enabled backend required')
        evidence[str(rank)] = {}
        for layer in sorted(expected, key=int):
            x, y = a['layers'][layer], b['layers'][layer]
            if len(x['device_calls']) != 3 or len(y['device_calls']) != 3:
                raise ValueError('three device phase counters are required')
            delta = [end-start for start, end in zip(x['device_calls'], y['device_calls'])]
            if len(delta) != 3 or min(delta) < 0 or delta[0] < 1 or delta[1] < 1:
                raise ValueError(f'rank{rank}/layer{layer}: missing prefill/decode device execution')
            py_decode = y['python_decode_calls']-x['python_decode_calls']
            if require_graph and (x['capture_calls'] < 1 or y['capture_calls'] != x['capture_calls']
                                  or delta[1] <= py_decode or py_decode < 0):
                raise ValueError(f'rank{rank}/layer{layer}: graph replay not demonstrated')
            evidence[str(rank)][layer] = {'device_delta': delta, 'python_decode_delta': py_decode,
                                         'graph_replay_lower_bound': max(0, delta[1]-py_decode)}
    return {'status': 'passed', 'scope': '43_layer_request_execution',
            'graph_required': require_graph, 'ranks': evidence}


class NumericalComparisonError(ValueError):
    """A failed numerical gate with complete, serializable comparison evidence."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def compare_first_token(native, weave, *, logprob_atol=.1):
    if native.get('input_sha256') != weave.get('input_sha256') or not native.get('input_sha256'):
        raise ValueError('native and Weave must use the same prompt token IDs')
    a, b = [v['response']['meta_info'] for v in (native, weave)]
    av, aid, *_ = a['output_token_logprobs'][0]
    bv, bid, *_ = b['output_token_logprobs'][0]
    atop = {v[1]: v[0] for v in a['output_top_logprobs'][0]}
    btop = {v[1]: v[0] for v in b['output_top_logprobs'][0]}
    if not math.isfinite(logprob_atol) or logprob_atol < 0:
        raise ValueError('logprob tolerance must be finite and nonnegative')
    if any(not math.isfinite(v) or v > 0 for v in [av, bv, *atop.values(), *btop.values()]):
        raise ValueError('first token logprobs must be finite and nonpositive')
    common = sorted(set(atop) & set(btop))
    deltas = [{'token_id': k, 'native_logprob': atop[k], 'weave_logprob': btop[k],
               'abs_delta': abs(atop[k]-btop[k])} for k in common]
    maximum = max((v['abs_delta'] for v in deltas), default=0.)
    selected_delta = abs(av-bv)
    failures = []
    if aid != bid:
        failures.append(f'first token differs: native={aid} weave={bid}')
    if set(atop) != set(btop) or not atop:
        failures.append('first token top-logprob support differs or is empty')
    if selected_delta > logprob_atol or maximum > logprob_atol:
        failures.append(f'first token logprob difference exceeds {logprob_atol}')
    diagnostics = {
        'status': 'failed' if failures else 'passed', 'scope': 'first_token_and_top_logprobs_only',
        'token_id': aid, 'weave_token_id': bid, 'first_token_matches': aid == bid,
        'first_token_logprob_delta': selected_delta,
        'native_only_token_ids': sorted(set(atop)-set(btop)),
        'weave_only_token_ids': sorted(set(btop)-set(atop)),
        'common_logprob_deltas': deltas,
        'max_abs_common_logprob_delta': maximum,
        'max_abs_logprob_delta': max(selected_delta, maximum),
        'common_tokens_exceeding_atol': sum(v['abs_delta'] > logprob_atol for v in deltas),
        'atol': logprob_atol, 'failures': failures,
    }
    # Both criteria stay strict. Compute all diagnostics before rejecting so a
    # changed support cannot hide numerical drift on the common tokens.
    if failures:
        raise NumericalComparisonError('; '.join(failures), diagnostics)
    return diagnostics
