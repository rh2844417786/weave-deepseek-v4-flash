"""Natural GPQA C32 client; all published artifacts contain counts and times only."""
import asyncio
from pathlib import Path
import time
import uuid

from .calibration import atomic_json
from .c32_serving import (ClientError, SSEParser, WindowMetrics, _abort_owned,
                          _code, _count_and_terminal, _validate_response_identity,
                          validate_base_url)

CONCURRENCY = 32
WINDOW_SECONDS = 1200.0
OUTPUT_TOKENS = 32768
WARMUP_TIMEOUT = 900.0
FLUSH_TIMEOUT = 30.0
HANDSHAKE_TIMEOUT = 30.0


def generation_settings(tokens=OUTPUT_TOKENS):
    return {'max_new_tokens': tokens, 'temperature': 0, 'ignore_eos': False}


def _payload(record, rid, *, tokens=OUTPUT_TOKENS, stream=True):
    # Think High is already encoded by the pinned offline encoder.
    return {'input_ids': record['input_token_ids'], 'rid': rid,
            'sampling_params': generation_settings(tokens), 'stream': stream,
            'return_logprob': False, 'return_text_in_logprobs': False}


async def _warmup(session, base_url, workload, owned):
    async def one(record):
        rid = uuid.uuid4().hex
        owned.add(rid)
        async with session.post(base_url+'/generate',
                                json=_payload(record, rid, tokens=8, stream=False),
                                allow_redirects=False) as response:
            if response.status != 200:
                raise ClientError('warmup_http_error')
            try:
                value = await response.json()
            except (ValueError, UnicodeError):
                raise ClientError('warmup_invalid_json') from None
            _validate_response_identity(value, rid)
            count, terminal = _count_and_terminal(value, 8, allow_normal_eos=True)
            if not terminal:
                raise ClientError('warmup_incomplete')
            owned.remove(rid)
            return count
    tasks = [asyncio.create_task(one(record)) for record in workload.records[:32]]
    try:
        counts = await asyncio.wait_for(asyncio.gather(*tasks), WARMUP_TIMEOUT)
        return {'requests_completed':32, 'concurrency':32, 'output_tokens':sum(counts),
                'max_new_tokens':8, 'included_in_measurement':False}
    finally:
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _flush(session, base_url):
    """Immediate idle flush, never a deferred server mutation during load."""
    deadline = time.monotonic()+FLUSH_TIMEOUT
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        async with session.get(base_url+'/flush_cache', params={'timeout':0},
                               allow_redirects=False,
                               timeout=min(2.0, max(.001, deadline-time.monotonic()))) as response:
            if response.status == 200:
                try:
                    ack = await response.text()
                except (ValueError, UnicodeError):
                    raise ClientError('post_warmup_flush_invalid_ack') from None
                if isinstance(ack,str) and ack.startswith('Cache flushed.'):
                    return {'status':'flushed', 'attempts':attempts,
                            'included_in_measurement':False}
                raise ClientError('post_warmup_flush_invalid_ack')
            if response.status != 400:
                raise ClientError('post_warmup_flush_http_error')
        await asyncio.sleep(min(.1, max(0,deadline-time.monotonic())))
    raise ClientError('post_warmup_flush_timeout')


async def _stream(session, base_url, payload, index, meter, rows):
    parser = SSEParser()
    async with session.post(base_url+'/generate', json=payload, allow_redirects=False) as response:
        if response.status != 200:
            raise ClientError('generate_http_error')
        async for block in response.content.iter_any():
            received = time.monotonic()
            if received >= meter.deadline:
                return False
            for value in parser.feed(block):
                if value is None:
                    raise ClientError('stream_done_without_terminal')
                _validate_response_identity(value, payload['rid'])
                terminal = meter.observe(index, value, received)
                if terminal:
                    # Publish only the type: matched stop strings can be text.
                    rows[index]['finish_reason'] = value['meta_info']['finish_reason']['type']
                    rows[index]['completed_monotonic'] = received
                    return True
        parser.finish()
        raise ClientError('stream_eof_without_terminal')


async def run_measurement(session, base_url, workload, output_dir, *, controller_handshake=False):
    """One fixed formal window, continuously refill each of 32 workers once per row."""
    base_url = validate_base_url(base_url)
    if len(workload.records) != 448:
        raise ClientError('gpqa_main_requires_448_records')
    output_dir = Path(output_dir)
    summary = {'schema_version':1, **workload.identity, 'workload':workload.identity, 'status':'running',
               'mode':'gpqa_measurement', 'concurrency':32, 'requested_seconds':WINDOW_SECONDS,
               'generation_settings':generation_settings(), 'numerical_policy':'record_only',
               'accuracy_evaluated':False, 'raw_content_exported':False, 'error_codes':[],
               'dataset_exhausted':False, 'dataset_exhausted_monotonic':None,
               'refill_exhausted':False, 'refill_exhausted_monotonic':None,
               'idle_seconds':0.0, 'last_completed_monotonic':None, 'reached_deadline':False, 'cleanup_complete':False}
    owned, tasks, rows = set(), [], {}
    meter = progress = None
    cancelled = None
    next_index = 0
    try:
        summary['warmup'] = await _warmup(session, base_url, workload, owned)
        summary['post_warmup_flush'] = await _flush(session, base_url)
        if controller_handshake:
            atomic_json(output_dir/'warmup-ready.json', {'schema_version':1,'status':'warmup_ready',
                        'ready_monotonic':time.monotonic()})
            async def wait_go():
                while not (output_dir/'measurement-go.json').is_file():
                    await asyncio.sleep(.05)
            try:
                await asyncio.wait_for(wait_go(),HANDSHAKE_TIMEOUT)
            except asyncio.TimeoutError:
                raise ClientError('controller_handshake_timeout') from None
        start = time.monotonic()
        meter = WindowMetrics(start,start+WINDOW_SECONDS,output_tokens=OUTPUT_TOKENS,allow_normal_eos=True)
        atomic_json(output_dir/'load-window.json', {'schema_version':1,'status':'load_started',
                    'started_monotonic':start,'deadline_monotonic':meter.deadline,
                    **workload.identity})

        async def worker():
            nonlocal next_index
            while True:
                dispatched = time.monotonic()
                if dispatched >= meter.deadline: return
                # Only an actual predeadline refill attempt proves loss of C32.
                # Dispatching the final row can still leave all 32 workers active.
                if next_index == len(workload.records):
                    if not summary['refill_exhausted']:
                        summary.update(refill_exhausted=True,refill_exhausted_monotonic=dispatched)
                    return
                index = next_index
                next_index += 1
                rid = uuid.uuid4().hex
                meter.start(index,dispatched)
                owned.add(rid)
                rows[index] = {'original_row_index':workload.records[index]['row_index'],
                               'finish_reason':None,'completed_monotonic':None}
                try:
                    complete = await _stream(session,base_url,_payload(workload.records[index],rid),index,meter,rows)
                    if not complete: return
                    owned.remove(rid)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    meter.fail(index,_code(error))
                    raise
                # Give the other workers a dispatch opportunity even with an
                # immediately available buffered response (also useful in tests).
                await asyncio.sleep(0)

        async def publish():
            while True:
                await asyncio.sleep(1)
                if time.monotonic() >= meter.deadline: return
                atomic_json(output_dir/'load-progress.json',{'schema_version':1,'status':'running',**meter.summary()})

        tasks = [asyncio.create_task(worker()) for _ in range(CONCURRENCY)]
        progress = asyncio.create_task(publish())
        done,pending = await asyncio.wait(tasks,timeout=max(0,meter.deadline-time.monotonic()),
                                          return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            if task.exception() is not None: raise task.exception()
        completed_times = [row['completed_monotonic'] for row in rows.values()
                           if row['completed_monotonic'] is not None]
        last_completed = max(completed_times) if completed_times else None
        summary['last_completed_monotonic'] = last_completed
        if len(completed_times) == len(workload.records) and last_completed < meter.deadline:
            summary.update(dataset_exhausted=True,dataset_exhausted_monotonic=last_completed,
                           idle_seconds=meter.deadline-last_completed)
            await asyncio.sleep(max(0,meter.deadline-time.monotonic()))
        summary['reached_deadline'] = time.monotonic() >= meter.deadline
        if not summary['reached_deadline']: raise ClientError('load_ended_before_deadline')
        if not meter.decode_tokens: raise ClientError('no_decode_tokens_received')
        if meter.peak_active != 32: raise ClientError('client_never_reached_C32')
        summary['status'] = 'passed'
    except asyncio.CancelledError as error:
        cancelled = error
        summary.update(status='failed',error_codes=['client_cancelled'])
    except Exception as error:
        summary.update(status='failed',error_codes=[_code(error)])
    finally:
        all_tasks = tasks + ([progress] if progress is not None else [])
        for task in all_tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*all_tasks,return_exceptions=True)
        failures = await _abort_owned(session,base_url,owned)
        summary.update(abort_requests=len(owned),abort_failures=failures,cleanup_complete=failures==0,
                       requests_unattempted=len(workload.records)-next_index,attempted_rows=next_index)
        if failures:
            summary['status']='failed'
            summary['error_codes'].append('abort_failed')
        if meter is not None:
            metrics=meter.summary()
            summary['error_codes']=sorted(set(summary['error_codes']+metrics.pop('error_codes')))
            summary.update(metrics)
            for index,row in rows.items():
                state=meter.states[index]
                row.update(status=state['status'],started_monotonic=state['start'],
                           last_token_monotonic=state['last'],output_tokens=state['count'],
                           decode_tokens=max(0,state['count']-1))
        completed_times = [row['completed_monotonic'] for row in rows.values()
                           if row['completed_monotonic'] is not None]
        summary['last_completed_monotonic'] = max(completed_times) if completed_times else None
        summary['completed_row_indices']=sorted(row['original_row_index'] for row in rows.values() if row['status']=='completed')
        atomic_json(output_dir/'gpqa-rows.json',{'schema_version':1,'rows':[rows[i] for i in sorted(rows)]})
        atomic_json(output_dir/'load-progress.json',summary)
        atomic_json(output_dir/'measurement-summary.json',summary)
    if cancelled: raise cancelled
    return summary
