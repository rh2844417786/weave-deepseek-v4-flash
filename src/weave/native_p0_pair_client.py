"""Bounded same-request diagnostic client; public artifacts contain commitments only.

The server must use --incremental-streaming-output --stream-interval 1.
Text hashes cover concatenated UTF-8 chunks, including chunks with no token delta.
Receipt times are client observations, not GPU timings or formal throughput.
"""
import asyncio
import hashlib
import json
import math
from pathlib import Path
import re
import time
import uuid

from .calibration import atomic_json
from .c32_serving import (ClientError, SSEParser, _abort_owned, _code,
                          _count_and_terminal, _validate_response_identity,
                          validate_base_url)
from .gpqa_c32_client import (_flush, _payload, _warmup, generation_settings,
                              HANDSHAKE_TIMEOUT, OUTPUT_TOKENS)

CONCURRENCY = 32
TIMEOUT_SECONDS = 3600.0
PREFIX_COUNTS = (1, 8, 32, 128, 512, 2048, 8192)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def selection_identity(workload):
    """Public ordered input commitments; input IDs and dynamic rids never escape."""
    if len(workload.records) != 448:
        raise ClientError('gpqa_main_requires_448_records')
    requests = []
    for record in workload.records[:CONCURRENCY]:
        if (type(record.get('row_index')) is not int or not 0 <= record['row_index'] < 448
                or not isinstance(record.get('prompt_sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', record['prompt_sha256'])
                or type(record.get('input_tokens')) is not int
                or not isinstance(record.get('input_token_ids'), list)
                or not record['input_token_ids']
                or record['input_tokens'] != len(record['input_token_ids'])
                or any(type(token) is not int or token < 0 for token in record['input_token_ids'])):
            raise ClientError('invalid_prepared_request')
        payload = _payload(record, 'excluded')
        del payload['rid']
        requests.append({'original_row_index': record['row_index'],
                         'prompt_sha256': record['prompt_sha256'],
                         'input_tokens': record['input_tokens'],
                         'request_payload_sha256': _digest(payload)})
    if len({row['original_row_index'] for row in requests}) != CONCURRENCY:
        raise ClientError('duplicate_prepared_row')
    return {'selection_seed': 42, 'selection_policy': 'first_32_file_order',
            'request_count': CONCURRENCY, 'requests': requests,
            'selection_sha256': _digest(requests)}


def _check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ClientError('stream_nonfinite_value')
    if isinstance(value, dict):
        for child in value.values():
            _check_finite(child)
    elif isinstance(value, list):
        for child in value:
            _check_finite(child)


async def _stream(session, base_url, payload, row):
    parser = SSEParser()
    digest = hashlib.sha256()
    prefixes = {}
    async with session.post(base_url + '/generate', json=payload,
                            allow_redirects=False) as response:
        if response.status != 200:
            raise ClientError('generate_http_error')
        async for block in response.content.iter_any():
            received = time.monotonic()
            for value in parser.feed(block):
                if value is None:
                    raise ClientError('stream_done_without_terminal')
                _check_finite(value)
                _validate_response_identity(value, payload['rid'])
                count, terminal = _count_and_terminal(value, OUTPUT_TOKENS, allow_normal_eos=True)
                if count < row['output_tokens']:
                    raise ClientError('stream_completion_tokens_decreased')
                if not isinstance(value.get('text'), str):
                    raise ClientError('stream_text_required')
                try:
                    encoded = value['text'].encode('utf-8')
                except UnicodeError:
                    raise ClientError('stream_invalid_text_encoding') from None
                digest.update(encoded)
                row['output_text_bytes'] += len(encoded)
                row['output_text_sha256'] = digest.hexdigest()
                row['output_tokens'] = count
                if row['first_receipt_monotonic'] is None:
                    row['first_receipt_monotonic'] = received
                row['last_receipt_monotonic'] = received
                row['elapsed_seconds'] = received - row['started_monotonic']
                if count in PREFIX_COUNTS:
                    # Only observed counts are committed. Repeated counts include
                    # all text emitted at that count; skipped counts stay absent.
                    prefixes[count] = digest.hexdigest()
                    row['prefix_digests'] = [
                        {'completion_tokens': n, 'sha256': sha}
                        for n, sha in sorted(prefixes.items())]
                if terminal:
                    row['finish_reason'] = value['meta_info']['finish_reason']['type']
                    row['status'] = 'completed'
                    return
        parser.finish()
        raise ClientError('stream_eof_without_terminal')


async def run_measurement(session, base_url, workload, output_dir, *,
                          timeout_seconds=TIMEOUT_SECONDS, controller_handshake=False):
    """Warm up and flush, then await one cohort of 32 requests without refilling."""
    base_url = validate_base_url(base_url)
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ClientError('invalid_timeout_seconds')
    selection = selection_identity(workload)
    output_dir = Path(output_dir)
    rows = [{**request, 'status': 'pending', 'output_tokens': 0, 'finish_reason': None,
             'output_text_sha256': hashlib.sha256(b'').hexdigest(), 'output_text_bytes': 0,
             'prefix_digests': [], 'started_monotonic': None, 'first_receipt_monotonic': None,
             'last_receipt_monotonic': None, 'elapsed_seconds': None}
            for request in selection['requests']]
    summary = {'schema_version': 1, 'mode': 'native_p0_pair', 'status': 'running',
               'selection': selection, 'source_workload_identity': workload.identity,
               'generation_settings': generation_settings(), 'request_count': CONCURRENCY,
               'concurrency': CONCURRENCY, 'requests_completed': 0,
               'timeout_seconds': timeout_seconds, 'started_monotonic': None,
               'deadline_monotonic': None, 'elapsed_seconds': None,
               'warmup': None, 'post_warmup_flush': None,
               'cleanup_complete': False, 'abort_requests': 0, 'abort_failures': 0,
               'error_codes': [], 'rows': rows, 'numerical_policy': 'record_only',
               'accuracy_evaluated': False, 'raw_content_exported': False,
               'stream_text_semantics': 'incremental_utf8',
               'prefix_digest_policy': 'last_emission_at_exact_observed_completion_count',
               'timing_scope': 'client_receipt_diagnostic_no_formal_tps',
               'completion_policy': 'all_32_responses_finish_by_natural_EOS_or_32768_tokens',
               'source_workload_metadata_note': 'Source identity retains the historical seconds=1200 field; it is not this paired run duration. timeout_seconds is an operational deadline.'}
    owned, tasks = set(), []
    cancelled = None
    try:
        summary['warmup'] = await _warmup(session, base_url, workload, owned)
        summary['post_warmup_flush'] = await _flush(session, base_url)
        atomic_json(output_dir / 'warmup-ready.json',
                    {'schema_version': 1, 'status': 'warmup_ready',
                     'ready_monotonic': time.monotonic()})
        if controller_handshake:
            async def wait_go():
                while not (output_dir / 'measurement-go.json').is_file():
                    await asyncio.sleep(.05)
            try:
                await asyncio.wait_for(wait_go(), HANDSHAKE_TIMEOUT)
            except asyncio.TimeoutError:
                raise ClientError('controller_handshake_timeout') from None
        start = time.monotonic()
        summary.update(started_monotonic=start, deadline_monotonic=start + timeout_seconds)
        atomic_json(output_dir / 'paired-load-start.json',
                    {'schema_version': 1, 'status': 'load_started',
                     'started_monotonic': start, 'deadline_monotonic': start + timeout_seconds,
                     'selection_sha256': selection['selection_sha256']})

        async def one(index):
            row = rows[index]
            rid = uuid.uuid4().hex
            owned.add(rid)
            row.update(status='running', started_monotonic=time.monotonic())
            try:
                await _stream(session, base_url, _payload(workload.records[index], rid), row)
                owned.remove(rid)
            except asyncio.CancelledError:
                row['status'] = 'cancelled'
                raise
            except Exception as error:
                row['status'] = 'failed'
                summary['error_codes'].append(_code(error))
                raise
            finally:
                if row['last_receipt_monotonic'] is None:
                    row['elapsed_seconds'] = time.monotonic() - row['started_monotonic']

        tasks = [asyncio.create_task(one(index)) for index in range(CONCURRENCY)]
        done, pending = await asyncio.wait(
            tasks, timeout=max(0, start + timeout_seconds - time.monotonic()),
            return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            if task.exception() is not None:
                raise task.exception()
        if pending:
            raise ClientError('measurement_timeout')
        summary['status'] = 'passed'
    except asyncio.CancelledError as error:
        cancelled = error
        summary.update(status='failed')
        summary['error_codes'].append('client_cancelled')
    except Exception as error:
        summary.update(status='failed')
        summary['error_codes'].append(_code(error))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if summary['started_monotonic'] is not None:
            summary['elapsed_seconds'] = time.monotonic() - summary['started_monotonic']
        summary['abort_requests'] = len(owned)
        failures = await _abort_owned(session, base_url, owned)
        summary.update(abort_failures=failures, cleanup_complete=failures == 0,
                       requests_completed=sum(row['status'] == 'completed' for row in rows))
        if failures:
            summary['status'] = 'failed'
            summary['error_codes'].append('abort_failed')
        if summary['status'] == 'passed' and summary['requests_completed'] != CONCURRENCY:
            summary['status'] = 'failed'
            summary['error_codes'].append('incomplete_cohort')
        summary['error_codes'] = sorted(set(summary['error_codes']))
        atomic_json(output_dir / 'paired-client-summary.json', summary)
    if cancelled:
        raise cancelled
    return summary
