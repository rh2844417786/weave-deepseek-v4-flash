"""Bounded anonymous observations of client SSE delivery (diagnostic only)."""
import math
import time

TIMING_KEYS = ('created_time', 'tokenize_finish_time', 'api_server_dispatch_finish_time',
               'first_token_time', 'wait_queue_entry_time', 'forward_entry_time', 'prefill_finished_time')


class StallClient:
    def __init__(self, max_requests=16000):
        self.max_requests = max_requests
        self.requests = {}
        self.gaps = []

    def start(self, index, timestamp):
        if len(self.requests) >= self.max_requests:
            raise ValueError('stall_client_request_overflow')
        self.requests[index] = dict(index=index, start_ns=round(timestamp * 1e9), first_ns=None,
                                    last_ns=None, count=0, finished=False, server_timing={})

    def observe(self, index, value, timestamp):
        row = self.requests[index]
        now = round(timestamp * 1e9)
        meta = value.get('meta_info', {})
        count = meta.get('completion_tokens', 0)
        if count > row['count']:
            if row['last_ns'] is not None and now > row['last_ns']:
                self.gaps.append(dict(index=index, start_ns=row['last_ns'], end_ns=now,
                                      delta_ns=now-row['last_ns'], after_token_position=row['count'],
                                      before_token_position=count))
                self.gaps.sort(key=lambda x: x['delta_ns'], reverse=True)
                del self.gaps[64:]
            if row['first_ns'] is None:
                row['first_ns'] = now
            row['last_ns'] = now
        row['count'] = count
        row['finished'] = meta.get('finish_reason') is not None
        timing = meta.get('weave_stall_timing', {})
        if isinstance(timing, dict):
            for key in TIMING_KEYS:
                v = timing.get(key)
                if isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and v > 0:
                    row['server_timing'][key] = v

    def report(self, status, errors):
        return dict(schema_version=1, status=status, error_codes=list(errors),
                    clock='CLOCK_MONOTONIC', clock_contract='Linux client monotonic and server perf_counter share CLOCK_MONOTONIC on the same host',
                    instrumented_only=True, instrumented_diagnostic=True, used_for_formal_performance=False, numerical_policy='record_only',
                    raw_content_exported=False, requests=list(self.requests.values()), top_gaps=self.gaps)


async def arm_ranks(session, port_base):
    if isinstance(port_base, bool) or not isinstance(port_base, int) or not 1024 <= port_base <= 65532:
        raise ValueError('stall_invalid_port_base')
    for rank in range(4):
        async with session.post(f'http://127.0.0.1:{port_base + rank}/arm', json={}, allow_redirects=False) as response:
            if response.status != 200:
                raise ValueError('stall_arm_http_error')
            payload = await response.json()
            if payload.get('status') not in ('armed', 'ok'):
                raise ValueError('stall_arm_failed')
