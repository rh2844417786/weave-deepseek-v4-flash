"""Strict, Torch-free evidence gates for the experimental Hopper compute probe.

Public evidence contains only identities, counts, scalar errors, compile resource
metadata and fresh event timings. A complete report is not a serving result.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

from .replay_trace import validate_capture_id, validate_identity as validate_capture_identity

VARIANTS = ('bf16_eager', 'fp8_eager', 'bf16_graph', 'fp8_graph')
SCOPE = 'online_group_to_routed_local_fp32_slots'
NUMERICAL_POLICY = 'record_only'
IDENTITY_FIELDS = {'candidate_source_sha256', 'bf16_source_sha256', 'replay_source_sha256',
    'checkpoint_content_sha256', 'image_id', 'capture_manifest_sha256', 'capture_identity',
    'warmups', 'repetitions', 'variants', 'layers', 'samples', 'invocation_id','numerical_policy'}
SYNTHETIC_CASES = (
    ('eager_b1', 1, 'eager'), ('eager_b7', 7, 'eager'), ('eager_b17', 17, 'eager'),
    ('eager_b32', 32, 'eager'), ('eager_shrink_b7', 7, 'eager'),
    ('eager_shrink_b1', 1, 'eager'), ('zero_b32', 32, 'eager'),
    ('graph_update_0', 32, 'graph'), ('graph_update_1', 32, 'graph'))
RESOURCE_SCALARS = {
    'device', 'multiprocessors', 'group_capacity', 'scratch_group_capacity',
    'gate_tiles_per_group', 'down_tiles_per_group', 'reduce_grid_blocks',
    'online_group_capacity', 'online_gate_grid_bound', 'online_down_grid_bound', 'online_max_batch',
    'numRegs', 'localSizeBytes', 'sharedSizeBytes', 'maxDynamicSharedSizeBytes',
    'maxThreadsPerBlock', 'dynamic_shared_bytes', 'threads_per_block', 'active_blocks_per_sm',
    'binaryVersion', 'ptxVersion', 'threads', 'registers_per_thread', 'static_shared_bytes',
    'local_bytes', 'binary_version', 'ptx_version', 'max_batch', 'local_experts',
    'active_expert_capacity', 'gate_grid_capacity', 'down_grid_capacity', 'tma_transaction_bytes'}
RESOURCE_ARRAYS = {'grid_controls', 'a_map_shape', 'middle_map_shape'}


def _fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f'{label} fields differ from public schema')


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _number(value, low=0, high=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= low
            and (high is None or value <= high))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def replay_source_sha256():
    """Bind all local Python used to load, check, time and validate this probe.

    CUDA sources and compile flags are separately bound by the BF16/candidate
    identities. They do not cover Python loaders or the independent oracle.
    Include replay_trace's eager-import dependency chain as well. Unused
    PeerRuntime service/profile methods and the outer controller have separate
    scopes; this gate never calls them or loads runtime.json.
    """
    root = Path(__file__).resolve().parents[2]
    names = ('src/weave/hopper_evidence.py', 'tests/gpu/check_hopper_candidate.py',
             'src/weave/hopper_reference.py', 'src/weave/hopper_candidate.py', 'src/weave/ops.py',
             'src/weave/replay_trace.py', 'src/weave/checkpoint_identity.py',
             'src/weave/weights.py', 'src/weave/staged_plan.py', 'tests/gpu/check_checkpoint.py',
             'src/weave/calibration.py', 'src/weave/gpu_allocation.py')
    return hashlib.sha256(b''.join(name.encode() + b'\0' + (root / name).read_bytes() for name in names)).hexdigest()


def validate_identity(value):
    _fields(value, IDENTITY_FIELDS, 'Hopper identity')
    if value['numerical_policy'] != NUMERICAL_POLICY:
        raise ValueError('Hopper numerical errors are record-only; threshold policies are unsupported')
    for key in IDENTITY_FIELDS:
        if key.endswith('_sha256') and not _sha(value[key]):
            raise ValueError('Hopper identity requires complete SHA256 digests')
    validate_capture_identity(value['capture_identity'])
    validate_capture_id(value['invocation_id'])
    capture = value['capture_identity']
    if (value['image_id'] != capture['image_id']
            or value['checkpoint_content_sha256'] != capture['checkpoint_content_sha256']):
        raise ValueError('current image/checkpoint must match captured inputs')
    # The frozen input archive retains its original kernel/capture provenance.
    # Current BF16/FP8/replay code has its own freshly checked source identity;
    # changing compute implementations must not require recapturing inputs.
    if (value['layers'] != list(range(43)) or value['samples'] != [64, 128]
            or value['variants'] != list(VARIANTS)):
        raise ValueError('all 43 layers, two samples and four paired variants required')
    for key in ('warmups', 'repetitions'):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError('positive integer warmups/repetitions required')
    return value


def atomic_report(path, value):
    """Write an atomic 0664 JSON checkpoint; invalid JSON cannot replace evidence."""
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            os.fchmod(stream.fileno(), 0o664)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def public_resources(value):
    """Copy an explicit compile-resource allowlist; never export arbitrary arrays."""
    result = {}
    for key, child in value.items():
        if key in RESOURCE_SCALARS:
            result[key] = child
        elif key in RESOURCE_ARRAYS:
            result[key] = list(child)
        elif key == 'stages':
            result[key] = {name: public_resources(attrs) for name, attrs in child.items()
                           if name in ('gate_up', 'down', 'reduce')}
    _validate_resources(result)
    return result


def _validate_resources(value):
    if not isinstance(value, dict) or set(value) - RESOURCE_SCALARS - RESOURCE_ARRAYS - {'stages'}:
        raise ValueError('unknown public resource fields')
    for key, child in value.items():
        if key in RESOURCE_SCALARS:
            if type(child) is not int or child < 0:
                raise ValueError('resource counts must be nonnegative integers')
        elif key in RESOURCE_ARRAYS:
            if (not isinstance(child, list) or not 1 <= len(child) <= 4
                    or any(type(item) is not int or item < 0 for item in child)):
                raise ValueError('invalid resource shape/grid counts')
        else:
            if not isinstance(child, dict) or set(child) - {'gate_up', 'down', 'reduce'}:
                raise ValueError('unknown resource stage')
            for attrs in child.values():
                _validate_resources(attrs)


def _metric(value):
    _fields(value, ('finite', 'relative_l2', 'max_abs'), 'numeric metric')
    if type(value['finite']) is not bool:
        raise ValueError('finite status requires a boolean')
    if value['finite']:
        if not _number(value['relative_l2']) or not _number(value['max_abs']):
            raise ValueError('finite scalar numerical errors required')
    elif value['relative_l2'] is not None or value['max_abs'] is not None:
        raise ValueError('nonfinite errors must be explicitly null')
    # No tolerance or finite-error magnitude controls acceptance or timing.
    return value['finite']


def _checks(value):
    _fields(value, ('baseline', 'implementation', 'precision'), 'numeric comparisons')
    return {key: _metric(metric) for key, metric in value.items()}


def point_key(point):
    if (type(point.get('layer')) is not int or point['layer'] not in range(43)
            or type(point.get('sample')) is not int or point['sample'] not in (64, 128)
            or point.get('variant') not in VARIANTS):
        raise ValueError('invalid layer/sample/variant point')
    return point['layer'], point['sample'], point['variant']


def _point(value, repetitions):
    _fields(value, ('layer', 'sample', 'variant', 'gpu_ms', 'timing_valid', 'checks',
                   'workload', 'stable_addresses', 'graph_input_update_verified'), 'point')
    point_key(value)
    finite = _checks(value['checks'])
    _fields(value['workload'], ('batch', 'local_slots', 'active_experts', 'max_rows_per_expert', 'bf16_groups'), 'workload')
    work = value['workload']
    if (work['batch'] != 32 or any(type(v) is not int for v in work.values())
            or not 0 <= work['local_slots'] <= 192 or not 0 <= work['active_experts'] <= 64
            or not 0 <= work['max_rows_per_expert'] <= 32 or not 0 <= work['bf16_groups'] <= 192
            or (work['local_slots'] == 0) != (work['active_experts'] == 0)
            or work['active_experts'] > work['local_slots']):
        raise ValueError('invalid local workload counts')
    if type(value['timing_valid']) is not bool or type(value['stable_addresses']) is not bool or type(value['graph_input_update_verified']) is not bool:
        raise ValueError('point gate statuses require booleans')
    samples = value['gpu_ms']
    if value['timing_valid']:
        if (not all(finite.values()) or not value['stable_addresses']
                or not value['graph_input_update_verified']):
            raise ValueError('finite execution, addresses and graph update checks required before valid timing')
        if (not isinstance(samples, list) or len(samples) != repetitions
                or any(not _number(v) or v <= 0 for v in samples)):
            raise ValueError('every fresh event repetition needs a finite positive duration')
    elif samples != []:
        raise ValueError('invalid timing must have no samples')


def _synthetic(value, complete):
    _fields(value, ('status', 'cases', 'stable_addresses', 'graph_input_updates', 'fp4_patterns', 'empty_rank'), 'synthetic gate')
    if value['status'] not in ('pending', 'passed', 'failed') or not isinstance(value['cases'], list):
        raise ValueError('invalid synthetic gate status')
    if (type(value['graph_input_updates']) is not int or not 0 <= value['graph_input_updates'] <= 2
            or any(type(value[key]) is not bool for key in ('stable_addresses', 'fp4_patterns', 'empty_rank'))):
        raise ValueError('invalid synthetic gate metadata')
    observed = []
    for case in value['cases']:
        _fields(case, ('case', 'batch', 'mode', 'checks', 'metadata_exact', 'nonlocal_slots_zero'), 'synthetic case')
        key = (case['case'], case['batch'], case['mode'])
        if key not in SYNTHETIC_CASES or key in observed:
            raise ValueError('unknown or duplicate synthetic case')
        observed.append(key)
        gates = _checks(case['checks'])
        if type(case['metadata_exact']) is not bool or type(case['nonlocal_slots_zero']) is not bool:
            raise ValueError('synthetic metadata gate requires booleans')
        if value['status'] == 'passed' and (not all(gates.values())
                or not case['metadata_exact'] or not case['nonlocal_slots_zero']):
            raise ValueError('synthetic status contradicts finite execution or metadata checks')
    if complete or value['status'] == 'passed':
        if (value['status'] != 'passed' or set(observed) != set(SYNTHETIC_CASES)
                or value['graph_input_updates'] != 2 or not value['stable_addresses']
                or not value['fp4_patterns'] or not value['empty_rank']):
            raise ValueError('complete synthetic changing-batch/empty/zero/FP4/graph gate required')


def validate_rank_report(value, expected_identity=None, require_complete=False):
    _fields(value, ('schema_version', 'kind', 'rank', 'status', 'identity', 'measurement_scope',
        'used_for_formal_performance', 'service_promotion_allowed', 'replica_sha256', 'synthetic',
        'points', 'resources', 'runtime', 'stop_reason'), 'rank report')
    if (value['schema_version'] != 2 or value['kind'] != 'hopper_fp8_local_expert_replay'
            or type(value['rank']) is not int or not 0 <= value['rank'] < 4
            or value['status'] not in ('complete', 'partial', 'failed')
            or value['measurement_scope'] != SCOPE or value['used_for_formal_performance'] is not False
            or value['service_promotion_allowed'] is not False):
        raise ValueError('invalid Hopper report schema, rank or measurement scope')
    validate_identity(value['identity'])
    if expected_identity is not None and value['identity'] != expected_identity:
        raise ValueError('Hopper invocation/source/checkpoint/capture identity mismatch')
    if not _sha(value['replica_sha256']):
        raise ValueError('replicated private input digest required')
    if value['stop_reason'] not in ('none', 'budget', 'nonfinite_output',
                                    'runtime_failure', 'interrupted', 'identity_mismatch'):
        raise ValueError('unknown stop reason')
    complete = value['status'] == 'complete'
    if ((complete and value['stop_reason'] != 'none')
            or (value['status'] == 'failed' and value['stop_reason'] in ('none', 'budget'))):
        raise ValueError('status and stop reason disagree')
    if require_complete and not complete:
        raise ValueError('complete rank evidence required')
    _synthetic(value['synthetic'], complete)
    if not isinstance(value['points'], list):
        raise ValueError('points must be a list')
    if any(point.get('timing_valid') for point in value['points']) and value['synthetic']['status'] != 'passed':
        raise ValueError('synthetic gates must pass before any real timing is valid')
    keys = set()
    for point in value['points']:
        _point(point, value['identity']['repetitions'])
        key = point_key(point)
        if key in keys:
            raise ValueError('duplicate replay point')
        keys.add(key)
    expected = {(layer, sample, variant) for layer in range(43) for sample in (64, 128) for variant in VARIANTS}
    if complete and (keys != expected or any(not point['timing_valid'] for point in value['points'])):
        raise ValueError('complete report needs every layer/sample/eager/graph pair and valid timing')
    _fields(value['resources'], ('bf16', 'fp8'), 'resource families')
    for resources in value['resources'].values():
        _validate_resources(resources)
    runtime = value['runtime']
    _fields(runtime, ('device_name', 'capability', 'multiprocessors', 'device_count', 'cuda_version', 'torch_version'), 'runtime')
    if (runtime['capability'] != [9, 0] or runtime['device_count'] != 4
            or type(runtime['multiprocessors']) is not int or runtime['multiprocessors'] < 1
            or any(not isinstance(runtime[key], str) or not re.fullmatch(r'[a-zA-Z0-9 .+_()-]{1,128}', runtime[key])
                   for key in ('device_name', 'cuda_version', 'torch_version'))
            or 'H100' not in runtime['device_name']):
        raise ValueError('four visible H100 devices with SM90 runtime evidence required')
    return value


def aggregate_rank_reports(values, expected_identity=None):
    if len(values) != 4 or {value.get('rank') for value in values} != {0, 1, 2, 3}:
        raise ValueError('four distinct Hopper rank reports required')
    expected_identity = expected_identity or values[0]['identity']
    for value in values:
        validate_rank_report(value, expected_identity)
    if len({value['replica_sha256'] for value in values}) != 1:
        raise ValueError('all four captured input replicas must match')
    complete = all(value['status'] == 'complete' for value in values)
    status = 'complete' if complete else ('failed' if any(value['status'] == 'failed' for value in values) else 'partial')
    points = [point for value in values for point in value['points']]
    all_checks = [point['checks'] for point in points] + [case['checks'] for value in values for case in value['synthetic']['cases']]
    errors = {}
    for kind in ('baseline','implementation','precision'):
        metrics=[row[kind] for row in all_checks]
        finite=[metric for metric in metrics if metric['finite']]
        errors[kind]={'finite_records':len(finite),'nonfinite_records':len(metrics)-len(finite),
            'relative_l2_max':max((m['relative_l2'] for m in finite),default=None),
            'max_abs':max((m['max_abs'] for m in finite),default=None)}
    pairs = {'eager': [], 'graph': []}
    for value in values:
        index = {point_key(point): point for point in value['points']}
        for layer in range(43):
            for sample in (64, 128):
                for mode in pairs:
                    left = index.get((layer, sample, 'bf16_' + mode))
                    right = index.get((layer, sample, 'fp8_' + mode))
                    if left and right and left['timing_valid'] and right['timing_valid']:
                        if left['workload'] != right['workload']:
                            raise ValueError('paired variants must use identical local workload counts')
                        pairs[mode].append(statistics.median(left['gpu_ms']) / statistics.median(right['gpu_ms']))
    summaries = {mode: {'pairs': len(ratios),
        'geomean': math.exp(sum(math.log(value) for value in ratios) / len(ratios)) if ratios else None,
        'median': statistics.median(ratios) if ratios else None,
        'min': min(ratios) if ratios else None, 'max': max(ratios) if ratios else None}
        for mode, ratios in pairs.items()}
    return {'schema_version': 2, 'status': status, 'identity': expected_identity,
        'measurement_scope': SCOPE, 'used_for_formal_performance': False, 'service_promotion_allowed': False,
        'point_count': len(points), 'expected_point_count': 1376, 'case_count_per_rank': 86 if complete else None,
        'numerical_policy':NUMERICAL_POLICY,'numerical_acceptance_evaluated':False,'error_summary':errors,
        'error_summary_scope':'recorded synthetic cases and per-variant real checks; repeated comparisons retained',
        'timings_diagnostic_only': not complete,
        'paired_speedups': summaries,
        'timing_interpretation': 'paired per-rank local compute event medians; excludes communication; not serving throughput'}
