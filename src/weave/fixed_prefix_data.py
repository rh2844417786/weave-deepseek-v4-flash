"""Private fixed-prefix inputs and one atomic B32 native generation request.

No Torch, dill, model or network imports happen at module load. Numerical
differences in the original logits are record-only; the exact forced output
continuation is a separate experimental input-integrity control.
"""
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

from .calibration import atomic_json
from .c32_serving import ClientError, _abort_owned, _code, validate_base_url
from .native_p0_pair_client import selection_identity

CONCURRENCY = 32
DEFAULT_STEPS = 8
TIMEOUT_SECONDS = 360.0
SELECTION_SHA256 = 'a5748d54c4605d57042784ad3e2de569ae017796411239a360f90aaabc9b3018'
SUMMARY_FILE = 'fixed-prefix-client-summary.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _steps(value):
    if type(value) is not int or value < 1:
        raise ClientError('positive_integer_steps_required')
    return value


def prepare_config(workload, steps=DEFAULT_STEPS, vocab_size=None):
    """Public ordered commitments, retaining the existing first32 selection ID."""
    _steps(steps)
    selection = selection_identity(workload)
    if type(vocab_size) is not int or vocab_size < 2:
        raise ClientError('valid_vocab_size_required')
    requests = []
    for slot, record in enumerate(workload.records[:CONCURRENCY]):
        if any(token >= vocab_size for token in record['input_token_ids']):
            raise ClientError('input_token_outside_vocab')
        requests.append({'row_slot': slot, 'original_row_index': record['row_index'],
                         'input_tokens': record['input_tokens'],
                         'prompt_sha256': record['prompt_sha256'],
                         'input_ids_sha256': digest(record['input_token_ids'])})
    return {'schema_version': 1, 'steps': steps,
            'selection_sha256': selection['selection_sha256'],
            'requests': requests, 'vocab_size': vocab_size}


def validate_pinned_selection(workload):
    selection = selection_identity(workload)
    lengths = [record['input_tokens'] for record in workload.records[:CONCURRENCY]]
    if (selection['selection_sha256'] != SELECTION_SHA256
            or sum(lengths) != 6059 or max(lengths) != 343):
        raise ClientError('pinned_first32_selection_required')
    return selection


def validate_teacher(value, *, selection_sha256=None, steps=None, vocab_size=None):
    """Validate private continuation before any measured or trace request."""
    if (not isinstance(value, dict)
            or set(value) != {'schema_version', 'selection_sha256', 'steps', 'output_ids', 'sha256'}
            or value.get('schema_version') != 1
            or type(value.get('steps')) is not int or value['steps'] < 1
            or not isinstance(value.get('selection_sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', value['selection_sha256'])):
        raise ClientError('invalid_teacher_schema')
    ids = value['output_ids']
    if (not isinstance(ids, list) or len(ids) != CONCURRENCY
            or any(not isinstance(row, list) or len(row) != value['steps']
                   or any(type(token) is not int or token < 0
                          or (vocab_size is not None and token >= vocab_size) for token in row)
                   for row in ids)):
        raise ClientError('invalid_teacher_token_shape')
    if value['sha256'] != digest(ids):
        raise ClientError('teacher_digest_mismatch')
    if selection_sha256 is not None and value['selection_sha256'] != selection_sha256:
        raise ClientError('teacher_selection_mismatch')
    if steps is not None and (type(steps) is not int or not 1 <= steps <= value['steps']):
        raise ClientError('teacher_has_insufficient_steps')
    return value


def load_teacher(path, **kwargs):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ClientError('private_teacher_file_required')
    try:
        value = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        raise ClientError('invalid_teacher_json') from None
    return validate_teacher(value, **kwargs)


def _write_teacher(path, value):
    """Private JSON with the host/container readable ACL mask, replaced atomically."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ClientError('fresh_teacher_path_required')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.teacher-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False)+'\n').encode())
            stream.flush()
            os.fchmod(stream.fileno(), 0o664)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ClientError('response_nonfinite_value')
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


def build_payload(workload, selection, phase, steps, rids, processor):
    return {'input_ids': [record['input_token_ids'] for record in workload.records[:CONCURRENCY]],
            'rid': rids, 'sampling_params': [
                {'max_new_tokens': steps, 'temperature': 0, 'ignore_eos': True,
                 'custom_params': {'weave_fixed_prefix': {
                     'phase': phase, 'row_slot': slot,
                     'selection_sha256': selection['selection_sha256'], 'steps': steps}}}
                for slot in range(CONCURRENCY)],
            'custom_logit_processor': processor, 'stream': False,
            'return_logprob': False, 'return_text_in_logprobs': False}


def parse_batch(value, rids, steps):
    if not isinstance(value, list) or len(value) != CONCURRENCY:
        raise ClientError('exact_32_response_list_required')
    expected = set(rids)
    by_id = {}
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get('meta_info'), dict):
            raise ClientError('invalid_response_schema')
        meta = row['meta_info']
        _finite(meta)
        rid = meta.get('id')
        if not isinstance(rid, str) or rid not in expected:
            raise ClientError('response_request_id_mismatch')
        if rid in by_id:
            raise ClientError('duplicate_response_request_id')
        ids = row.get('output_ids')
        if (not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids)):
            raise ClientError('actual_output_ids_required')
        if (len(ids) != steps or type(meta.get('completion_tokens')) is not int
                or meta['completion_tokens'] != steps):
            raise ClientError('output_token_count_mismatch')
        finish = meta.get('finish_reason')
        if (not isinstance(finish, dict) or finish.get('type') != 'length'
                or type(finish.get('length')) is not int or finish['length'] != steps):
            raise ClientError('length_finish_required')
        by_id[rid] = ids
    return [by_id[rid] for rid in rids]


async def run_phase(session, base_url, workload, output_dir, teacher_path, *, phase,
                    steps=DEFAULT_STEPS, timeout_seconds=TIMEOUT_SECONDS, processor=None):
    """No warmup/flush: the owning controller performs those before each phase."""
    output_dir, teacher_path = Path(output_dir), Path(teacher_path)
    summary = {'schema_version': 1, 'mode': 'fixed_prefix_native_p0', 'phase': phase,
               'status': 'running', 'request_count': CONCURRENCY, 'requests_completed': 0,
               'steps': steps, 'output_tokens': 0, 'numerical_policy': 'record_only',
               'continuation_policy': 'exact_forced_control_inputs_not_numerical_agreement',
               'raw_content_exported': False, 'accuracy_evaluated': False,
               'timeout_seconds': timeout_seconds, 'cleanup_complete': False,
               'abort_requests': 0, 'abort_failures': 0, 'error_codes': [],
               'response_metadata_finite': None, 'teacher_sha256': None,
               'started_monotonic': None, 'deadline_monotonic': None,
               'elapsed_seconds': None, 'rows': []}
    owned = set()
    cancelled = None
    try:
        base_url = validate_base_url(base_url)
        _steps(steps)
        if phase not in ('seed', 'measure', 'trace'):
            raise ClientError('invalid_fixed_prefix_phase')
        if (type(timeout_seconds) not in (float, int) or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0):
            raise ClientError('invalid_timeout_seconds')
        if output_dir.resolve() in teacher_path.resolve().parents:
            raise ClientError('teacher_must_be_outside_public_client_directory')
        selection = selection_identity(workload)
        summary.update(selection=selection, source_workload_identity=workload.identity,
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       input_tokens=sum(record['input_tokens'] for record in workload.records[:CONCURRENCY]),
                       input_ids_sha256=digest([record['input_token_ids']
                                                for record in workload.records[:CONCURRENCY]]))
        teacher = None
        if phase == 'seed':
            if teacher_path.exists() or teacher_path.is_symlink():
                raise ClientError('fresh_teacher_path_required')
        else:
            teacher = load_teacher(teacher_path, selection_sha256=selection['selection_sha256'], steps=steps)
            summary['teacher_sha256'] = teacher['sha256']
        if processor is None:
            # Keep preview/data-preparation independent of Torch/dill. The import
            # path must match sitecustomize so dill resolves the same class.
            from src.weave.fixed_prefix_runtime import FixedPrefixProcessor
            processor = FixedPrefixProcessor.to_str()
        if not isinstance(processor, str) or not processor:
            raise ClientError('serialized_processor_required')
        rids = [uuid.uuid4().hex for _ in range(CONCURRENCY)]
        payload = build_payload(workload, selection, phase, steps, rids, processor)
        owned.update(rids)
        start = time.monotonic()
        summary.update(started_monotonic=start, deadline_monotonic=start+timeout_seconds)

        async def request():
            async with session.post(base_url+'/generate', json=payload, allow_redirects=False) as response:
                if response.status != 200:
                    raise ClientError('generate_http_error')
                try:
                    value = await response.json()
                except (ValueError, UnicodeError):
                    raise ClientError('generate_invalid_json') from None
                return parse_batch(value, rids, steps)

        try:
            output = await asyncio.wait_for(request(), timeout_seconds)
        except asyncio.TimeoutError:
            raise ClientError('fixed_prefix_timeout') from None
        summary['response_metadata_finite'] = True
        if teacher is not None and output != [row[:steps] for row in teacher['output_ids']]:
            raise ClientError('forced_continuation_integrity_mismatch')
        if phase == 'seed':
            teacher = {'schema_version': 1, 'selection_sha256': selection['selection_sha256'],
                       'steps': steps, 'output_ids': output, 'sha256': digest(output)}
            _write_teacher(teacher_path, teacher)
            summary['teacher_sha256'] = teacher['sha256']
        summary.update(status='passed', requests_completed=CONCURRENCY,
                       output_tokens=CONCURRENCY*steps, output_ids_sha256=digest(output),
                       rows=[{'row_slot': index, 'original_row_index': record['row_index'],
                              'input_tokens': record['input_tokens'],
                              'input_ids_sha256': digest(record['input_token_ids']),
                              'output_tokens': steps, 'output_ids_sha256': digest(output[index]),
                              'finish_reason': 'length'}
                             for index, record in enumerate(workload.records[:CONCURRENCY])])
        owned.clear()
    except asyncio.CancelledError as error:
        cancelled = error
        summary.update(status='failed')
        summary['error_codes'].append('client_cancelled')
    except Exception as error:
        summary.update(status='failed')
        summary['error_codes'].append(_code(error))
    finally:
        if summary['started_monotonic'] is not None:
            summary['elapsed_seconds'] = time.monotonic()-summary['started_monotonic']
        summary['abort_requests'] = len(owned)
        failures = await _abort_owned(session, base_url, owned)
        summary.update(abort_failures=failures, cleanup_complete=failures == 0)
        if failures:
            summary['status'] = 'failed'
            summary['error_codes'].append('abort_failed')
        summary['error_codes'] = sorted(set(summary['error_codes']))
        atomic_json(output_dir/SUMMARY_FILE, summary)
    if cancelled:
        raise cancelled
    return summary
