"""Formal GPQA streaming and sanitized server-boundary performance metrics."""
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import queue
import socket
import tempfile
import threading
import time
import uuid
from urllib.parse import urlsplit

from .request_timing import BOUNDARY, SOURCE, NATIVE_SHA256, source_sha256
from .gpqa import load_prepared_records as _load_records

GROUP_SECONDS = 600
WARMUP_SECONDS = 120
MAX_WARMUPS = 2


def _sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def summarize_response(row_index, rid, input_ids, response, *, state=None):
    meta = response.get('meta_info') or {}
    ids = response.get('output_ids') or []
    reason = meta.get('finish_reason')
    reason_type = reason.get('type') if isinstance(reason, dict) else reason
    result = {'row_index': row_index, 'request_id': rid, 'input_tokens': len(input_ids),
              'input_sha256': _sha(input_ids), 'output_tokens': len(ids),
              'output_sha256': _sha(ids), 'finish_reason': reason_type,
              'status': state or 'partial', 'ttft_s': None, 'decode_s': None,
              'decode_tokens': None, 'decode_tps': None, 'tpot_s': None}
    if state is not None:
        return result
    if reason_type not in ('stop', 'length'):
        result['status'] = 'aborted' if reason_type == 'abort' else 'partial'
        return result
    timing = meta.get('weave_timing') or {}
    try:
        n = len(ids)
        if n < 1 or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError('missing actual token IDs')
        if type(meta.get('completion_tokens')) is not int or meta['completion_tokens'] != n or type(timing.get('output_tokens')) is not int or timing['output_tokens'] != n:
            raise ValueError('actual IDs/count mismatch')
        if meta.get('id') != rid:
            raise ValueError('response request identity mismatch')
        if (timing.get('boundary') != BOUNDARY or timing.get('clock') != 'perf_counter'
                or timing.get('source') != SOURCE or not _digest(timing.get('source_sha256'))
                or timing.get('native_source_sha256') != NATIVE_SHA256
                or timing.get('source_sha256') != source_sha256()):
            raise ValueError('server timing source identity missing')
        received, first, last = (timing.get(key) for key in
            ('request_received_s', 'first_token_ready_s', 'last_token_ready_s'))
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (received, first, last)):
            raise ValueError('missing positive finite server timestamps')
        if received > first or last < first or (n == 1 and last != first) or (n > 1 and last == first):
            raise ValueError('inconsistent server token boundaries')
        span = last-first
        result.update(status='completed', ttft_s=first-received, decode_s=span,
                      decode_tokens=n-1, decode_tps=(n-1)/span if n > 1 else None,
                      tpot_s=span/(n-1) if n > 1 else None, server_timing=timing)
    except (TypeError, ValueError) as error:
        result.update(status='invalid', invalid_reason=str(error))
    return result


def aggregate(rows):
    completed = [r for r in rows if r['status'] == 'completed']
    decoded = [r for r in completed if r['output_tokens'] > 1]
    tokens = sum(r['decode_tokens'] for r in decoded)
    seconds = sum(r['decode_s'] for r in decoded)
    reasons = {}
    for row in rows:
        reason = row.get('finish_reason') or row['status']
        reasons[reason] = reasons.get(reason, 0)+1
    return {'completed_count': len(completed), 'decode_tokens': tokens, 'decode_s': seconds,
            'decode_tps': tokens/seconds if seconds > 0 else None,
            'tpot_s': seconds/tokens if tokens > 0 else None,
            'mean_ttft_s': sum(r['ttft_s'] for r in completed)/len(completed) if completed else None,
            'output_tokens': sum(r['output_tokens'] for r in completed),
            'input_tokens': sum(r['input_tokens'] for r in completed), 'reason_counts': reasons}


def compare_groups(native, weave):
    def completed(group):
        rows = group['records']
        if len({r['row_index'] for r in rows}) != len(rows):
            raise ValueError('duplicate original row identities')
        return {r['row_index']: r for r in rows if r['status'] == 'completed'}
    n, w = completed(native), completed(weave)
    common = sorted(n.keys() & w.keys())
    mismatch = [i for i in common if n[i]['input_sha256'] != w[i]['input_sha256']
                or n[i]['server_timing']['source_sha256'] != w[i]['server_timing']['source_sha256']
                or n[i]['server_timing']['native_source_sha256'] != w[i]['server_timing']['native_source_sha256']]
    valid = [] if mismatch else common
    a, b = aggregate([n[i] for i in valid]), aggregate([w[i] for i in valid])
    speed = b['decode_tps']/a['decode_tps'] if a['decode_tps'] and b['decode_tps'] is not None else None
    return {'status': 'identity_mismatch' if mismatch else ('compared' if common else 'no_common_completed'),
            'common_completed_row_ids': valid, 'identity_mismatch_row_ids': mismatch,
            'native': a, 'weave': b, 'relative_decode_speed': speed,
            'native_completed_count': len(n), 'weave_completed_count': len(w),
            'quality_diagnostics': {'different_output_row_ids':
                [i for i in valid if n[i]['output_sha256'] != w[i]['output_sha256']]}}


class StreamAccumulator:
    """The pinned server emits incremental output IDs and cumulative counts."""
    def __init__(self, rid):
        self.rid, self.ids, self.meta = rid, [], {}
        self.first_token_logprobs = None

    def add(self, event):
        ids = event.get('output_ids', [])
        if not isinstance(ids, list) or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError('stream output IDs must be integer deltas')
        meta = event.get('meta_info') or {}
        if meta.get('id') != self.rid:
            raise ValueError('foreign stream request identity')
        if self.first_token_logprobs is None and ids:
            self.first_token_logprobs = {key: value[0] for key, value in meta.items()
                if key.startswith('output_') and 'logprob' in key and isinstance(value, list) and value}
        self.ids.extend(ids)
        self.meta.update({key: value for key, value in meta.items() if 'logprob' not in key})

    def response(self):
        return {'output_ids': self.ids, 'meta_info': self.meta}


def _connection(url, timeout):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise ValueError('HTTP server URL required')
    cls = http.client.HTTPSConnection if parts.scheme == 'https' else http.client.HTTPConnection
    return cls(parts.hostname, parts.port, timeout=timeout), parts.path.rstrip('/')


def _close_socket(transport):
    if transport is not None:
        try: transport.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        transport.close()


class _ConnectionLifetime:
    """Prevent late socket publication and stdlib reconnect after cleanup.

    The lock covers only stop checks and transport ownership. DNS, connect,
    TLS negotiation, send and close run outside it, so cancellation never waits
    for a blocking network operation. A socket returned after stop is closed
    before HTTPConnection can assign or send through it.
    """
    def __init__(self, conn, deadline, clock):
        self.conn, self.deadline, self.clock = conn, deadline, clock
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.transports = []
        create = getattr(conn, '_create_connection', None)
        connect = getattr(conn, 'connect', None)
        send = getattr(conn, 'send', None)
        request = conn.request

        if create is not None:
            def guarded_create(*args, **kwargs):
                self.check()
                transport = create(*args, **kwargs)
                self.publish(transport)
                return transport
            conn._create_connection = guarded_create
        if connect is not None:
            def guarded_connect(*args, **kwargs):
                self.check()
                connect(*args, **kwargs)
                # Covers the interval between socket creation/publication and
                # HTTPConnection.sock assignment, including a new TLS wrapper.
                self.publish(getattr(conn, 'sock', None))
                self.check()
            conn.connect = guarded_connect
        if send is not None:
            def guarded_send(*args, **kwargs):
                self.check()
                value = send(*args, **kwargs)
                self.check()
                return value
            conn.send = guarded_send

        def guarded_request(*args, **kwargs):
            self.check()
            return request(*args, **kwargs)
        conn.request = guarded_request

    def _live(self):
        return not self.stopped.is_set() and self.clock() < self.deadline

    def check(self):
        with self.lock:
            if not self._live():
                raise TimeoutError('HTTP operation stopped before network action')

    def publish(self, transport):
        with self.lock:
            live = self._live()
            if live and transport is not None and not any(s is transport for s in self.transports):
                self.transports.append(transport)
        if not live:
            _close_socket(transport)
            raise TimeoutError('HTTP socket arrived after stop/deadline')

    def close(self):
        with self.lock:
            self.stopped.set()
            transports, self.transports = self.transports, []
            current = getattr(self.conn, 'sock', None)
            if current is not None and not any(s is current for s in transports):
                transports.append(current)
        for transport in transports:
            _close_socket(transport)
        self.conn.close()


def _json_request(url, endpoint, payload=None, *, timeout=2, deadline=None, clock=time.monotonic):
    deadline = clock()+timeout if deadline is None else min(deadline, clock()+timeout)
    conn, base = _connection(url, max(.001, deadline-clock()))
    events = queue.Queue()
    lifetime = _ConnectionLifetime(conn, deadline, clock)

    def reader():
        try:
            conn.request('POST' if payload is not None else 'GET', base+endpoint,
                         body=json.dumps(payload) if payload is not None else None,
                         headers={'Content-Type': 'application/json'})
            lifetime.check()
            response = conn.getresponse()
            data = response.read()
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status} at {endpoint}')
            if not data:
                value = None
            else:
                try: value = json.loads(data)
                except json.JSONDecodeError: value = data.decode('utf-8')
            events.put(('result', value))
        except Exception as error:
            events.put(('error', error))
        finally:
            lifetime.close()

    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            remaining = deadline-clock()
            if remaining <= 0: raise TimeoutError(f'{endpoint} absolute deadline')
            try: kind, value = events.get(timeout=min(remaining, .1))
            except queue.Empty: continue
            if clock() >= deadline: raise TimeoutError(f'{endpoint} absolute deadline')
            if kind == 'error': raise value
            return value
    finally:
        lifetime.close()


def _abort_request(url, rid):
    try:
        _json_request(url, '/abort_request', {'rid': rid}, timeout=2)
    except Exception:
        pass


def _stream_request(url, payload, deadline, clock):
    accumulator = StreamAccumulator(payload['rid'])
    events = queue.Queue()
    conn, base = _connection(url, max(.001, deadline-clock()))
    lifetime = _ConnectionLifetime(conn, deadline, clock)
    stop = lifetime.stopped

    def reader():
        try:
            conn.request('POST', base+'/generate', body=json.dumps(payload), headers={'Content-Type': 'application/json'})
            lifetime.check()
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f'generate HTTP {response.status}')
            data_lines = []
            while not stop.is_set():
                line = response.readline()
                if not line:
                    events.put(('end', None)); return
                line = line.decode('utf-8').strip()
                if not line:
                    if data_lines:
                        data = '\n'.join(data_lines); data_lines = []
                        if data == '[DONE]': events.put(('end', None)); return
                        events.put(('event', json.loads(data)))
                elif line.startswith('data:'):
                    data_lines.append(line[5:].lstrip())
        except Exception as error:
            events.put(('error', type(error).__name__))
        finally:
            lifetime.close()

    worker = threading.Thread(target=reader, daemon=True)
    worker.start()
    state = None
    try:
        while True:
            remaining = deadline-clock()
            if remaining <= 0:
                state = 'deadline'; break
            try:
                kind, data = events.get(timeout=min(remaining, .1))
            except queue.Empty:
                continue
            if clock() >= deadline:
                if kind == 'event':
                    accumulator.add(data)
                state = 'deadline'; break
            if kind == 'end':
                state = 'partial'
                break
            if kind == 'error':
                state = 'http_error'; break
            accumulator.add(data)
            if accumulator.meta.get('finish_reason') is not None:
                break
    except BaseException as error:
        # Caller receives partial data and can checkpoint before re-raising cancellation.
        state = 'cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'invalid_stream'
        accumulator.interruption = error
    finally:
        lifetime.close()
        # Retain already parsed deltas on a timeout; these remain partial and
        # never enter throughput even if their terminal event arrived late.
        if state == 'deadline':
            while True:
                try:
                    kind, data = events.get_nowait()
                except queue.Empty:
                    break
                if kind == 'event':
                    try:
                        accumulator.add(data)
                    except (TypeError, ValueError):
                        break
        if state is not None:
            _abort_request(url, payload['rid'])
    return accumulator, state


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def benchmark_group(url, records_path, output_dir, *, ready_monotonic, clock=time.monotonic, raw_root=None):
    """Run identical requests; ready_monotonic is the observed health-success time.

    Sanitized summary lives in output_dir. Decodable token/logprob artifacts always
    live under the configured working disk gpqa/<unique group>/, never public reports.
    """
    records, manifest = _load_records(records_path)
    if not math.isfinite(ready_monotonic) or ready_monotonic > clock():
        raise ValueError('observed ready time must precede benchmark start')
    deadline = ready_monotonic+GROUP_SECONDS
    output_dir = Path(output_dir)
    repo=Path(__file__).resolve().parents[2]
    raw_base=Path(raw_root if raw_root is not None else os.environ.get('WORK_ROOT',repo/'work'))
    if not raw_base.is_absolute(): raise ValueError('raw work root must be absolute')
    raw_dir=raw_base.resolve()/'gpqa'/uuid.uuid4().hex
    summary = {'status': 'running', 'ready_monotonic': ready_monotonic,
               'deadline_monotonic': deadline, 'record_sha256': manifest['record_sha256'],
               'order_seed': 42, 'records': [], 'warmups': [], 'warmup_count': 0,
               'timeout_count': 0, 'warmup_timeout_count': 0,
               'raw_artifact_relative_path': str(raw_dir.relative_to(repo)) if raw_dir.is_relative_to(repo) else None,
               'raw_artifact_directory': str(raw_dir),
               'request_settings': {'sampling_params': {'temperature': 0, 'max_new_tokens': 32768, 'ignore_eos': False},
                                    'return_logprob': True, 'top_logprobs_num': 10,
                                    'return_text_in_logprobs': False, 'stream': True}}

    def checkpoint():
        summary.update(aggregate(summary['records']))
        summary['attempted_count'] = len(summary['records'])
        summary['unattempted_count'] = len(records)-len(summary['records'])
        _atomic_json(output_dir/'summary.json', summary)

    checkpoint()
    for warmup, selected, phase_deadline in (
        (True, records[:MAX_WARMUPS], min(deadline, ready_monotonic+WARMUP_SECONDS)),
        (False, records, deadline),
    ):
        for record in selected:
            if clock() >= phase_deadline: break
            rid = 'weave-gpqa-'+uuid.uuid4().hex
            payload = dict(summary['request_settings'], rid=rid, input_ids=record['input_token_ids'])
            # A standalone batch1 service is required; flush makes each natural input uncached.
            try:
                _json_request(url, '/flush_cache', timeout=max(.001, min(2, phase_deadline-clock())),
                              deadline=phase_deadline, clock=clock)
            except BaseException as error:
                interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                state = 'cancelled' if interrupted else ('deadline' if clock() >= phase_deadline else 'cache_flush_error')
                row = summarize_response(record['row_index'], rid, payload['input_ids'], {}, state=state)
                accumulator = StreamAccumulator(rid)
                if interrupted:
                    accumulator.interruption = error
            else:
                if clock() >= phase_deadline: break
                accumulator, state = _stream_request(url, payload, phase_deadline, clock)
                row = summarize_response(record['row_index'], rid, payload['input_ids'], accumulator.response(), state=state)
                if row['status'] == 'invalid':
                    _abort_request(url, rid)
            row['prompt_sha256'] = record.get('prompt_sha256')
            (summary['warmups'] if warmup else summary['records']).append(row)
            summary['warmup_count'] += int(warmup)
            summary['warmup_timeout_count' if warmup else 'timeout_count'] += int(row['status'] == 'deadline')
            _atomic_json(raw_dir/(rid+'.json'), {'row_index': record['row_index'], 'warmup': warmup,
                'output_ids': accumulator.ids, 'first_token_logprobs': accumulator.first_token_logprobs})
            checkpoint()
            if hasattr(accumulator, 'interruption'):
                summary['status'] = row['status']; checkpoint()
                raise accumulator.interruption
    summary['status'] = ('complete' if len(summary['records']) == len(records) and summary['completed_count'] == len(records)
                         else 'deadline' if clock() >= deadline else 'incomplete')
    checkpoint()
    return summary
