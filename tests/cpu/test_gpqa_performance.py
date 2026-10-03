import copy
import hashlib
import http.client
import importlib.util
import json
import os
import random
import queue
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

try:
    from src.weave import performance as perf
    from src.weave import request_timing as timing
except ImportError:
    perf = timing = None


def response(ids=(7, 8, 9), reason='stop'):
    return {'output_ids': list(ids), 'meta_info': {'id': 'owned', 'completion_tokens': len(ids),
        'finish_reason': {'type': reason}, 'weave_timing': {
            'request_received_s': 10., 'first_token_ready_s': 12., 'last_token_ready_s': 16.,
            'output_tokens': len(ids), 'boundary': 'tokenizer_manager_output_handling',
            'clock': 'perf_counter', 'source': 'sglang_api_request_time_stats',
            'source_sha256': timing.source_sha256() if timing else 'a'*64,
            'native_source_sha256': timing.NATIVE_SHA256 if timing else 'b'*64}}}


def fixed_order():
    order = list(range(448)); random.Random(42).shuffle(order)
    return order


def make_records(root):
    records = root/'main.jsonl'
    records.write_text(''.join(json.dumps({'row_index': i, 'input_token_ids': [1, 2]})+'\n' for i in fixed_order()))
    pinned=json.loads((Path(__file__).resolve().parents[2]/'data/gpqa/deepseek_model_manifest.json').read_text())
    sources=json.loads((Path(__file__).resolve().parents[2]/'data/gpqa/source_manifest.json').read_text())
    main=next(row for row in sources['records'] if Path(row['path']).name=='gpqa_main.csv')
    (root/'manifest.json').write_text(json.dumps({'model_id':pinned['model_id'], 'model_revision':pinned['revision'],
        'pinned_model_metadata_sha256':pinned['sha256'], 'count': 448, 'order_seed': 42,
        'observed_model_metadata_sha256':pinned['sha256'], 'config_matches_pinned':True,
        'gpqa_main_sha256':main['sha256'], 'thinking_mode':'thinking', 'reasoning_effort':'high',
        'reasoning_mode': 'Think High', 'max_new_tokens': 32768, 'normal_eos_allowed': True,
        'record_sha256': hashlib.sha256(records.read_bytes()).hexdigest()}))
    return records


def delayed_connection_fixture():
    started, release, closed, finished = (threading.Event() for _ in range(4))
    log = []
    class Socket:
        def setsockopt(self, *args): pass
        def sendall(self, data): log.append(('wire_send', bytes(data)))
        def makefile(self, *args): raise OSError('fake response end')
        def close(self): log.append(('socket_close', None)); closed.set()
        def shutdown(self, *args): log.append(('socket_shutdown', None))
    class DelayedConnection(http.client.HTTPConnection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            def create(*args, **kwargs):
                started.set()
                release.wait(1.)
                log.append(('socket_created', None))
                return Socket()
            self._create_connection = create
        def request(self, *args, **kwargs):
            try: return super().request(*args, **kwargs)
            finally: finished.set()
    return DelayedConnection, started, release, closed, finished, log


class PerformanceTest(unittest.TestCase):
    def test_formal_reuse_rejects_contradictory_or_missing_saved_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=make_records(root)
            original=json.loads((root/'manifest.json').read_text())
            for changes in ({'gpqa_main_sha256':'0'*64},
                            {'observed_model_metadata_sha256':{}},
                            {'thinking_mode':'disabled'}, {'reasoning_effort':'low'},
                            {'observed_model_metadata_sha256':dict(original['observed_model_metadata_sha256'],
                                                                  **{'tokenizer.json':'0'*64})},
                            {'observed_model_metadata_sha256':dict(original['observed_model_metadata_sha256'],
                                                                  **{'encoding/encoding_dsv4.py':'0'*64})}):
                with self.subTest(changes=changes):
                    (root/'manifest.json').write_text(json.dumps(dict(original,**changes)))
                    with self.assertRaises(ValueError): perf._load_records(records)

    def test_observed_config_difference_preserves_preparer_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=make_records(root)
            manifest=json.loads((root/'manifest.json').read_text())
            manifest['observed_model_metadata_sha256']['config.json']='0'*64
            manifest['config_matches_pinned']=False
            (root/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(len(perf._load_records(records)[0]),448)

    def test_raw_artifacts_use_the_configured_work_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();records=make_records(root);now=[121.]
            def control(*args,**kwargs):
                now[0]=1201.
                raise TimeoutError('deadline')
            with patch.object(perf,'_json_request',side_effect=control):
                result=perf.benchmark_group('http://localhost:1',records,root/'public',
                    ready_monotonic=0.,clock=lambda:now[0],raw_root=root/'working-disk')
            raw=Path(result['raw_artifact_directory'])
            self.assertTrue(raw.is_relative_to(root/'working-disk'))
            self.assertEqual(len(list(raw.glob('*.json'))),1)
            self.assertEqual(result['records'][0]['status'],'deadline')

    def setUp(self):
        self.assertIsNotNone(perf, 'formal GPQA performance module required')

    def test_metrics_use_server_token_boundaries(self):
        row = perf.summarize_response(3, 'owned', [1, 2], response())
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(row['ttft_s'], 2.)
        self.assertEqual(row['decode_s'], 4.)
        self.assertEqual(row['decode_tokens'], 2)
        self.assertEqual(row['decode_tps'], .5)
        self.assertEqual(row['tpot_s'], 2.)
        self.assertNotIn('output_ids', row)
        self.assertNotIn('input_ids', row)

    def test_counts_identity_and_client_timestamps_cannot_complete(self):
        for mutate in (
            lambda x: x['meta_info'].update(completion_tokens=100),
            lambda x: x['meta_info'].update(id='other'),
            lambda x: x['meta_info']['weave_timing'].update(output_tokens=2),
            lambda x: x['meta_info']['weave_timing'].update(clock='client'),
            lambda x: x['meta_info']['weave_timing'].update(native_source_sha256='0'*64),
            lambda x: x['meta_info']['weave_timing'].update(source_sha256='0'*64),
            lambda x: x['meta_info']['weave_timing'].update(last_token_ready_s=11),
            lambda x: x['meta_info'].pop('weave_timing'),
        ):
            value = response(); mutate(value)
            self.assertEqual(perf.summarize_response(3, 'owned', [1], value)['status'], 'invalid')

    def test_partial_abort_empty_and_one_token_have_honest_metrics(self):
        for reason in (None, 'abort'):
            self.assertNotEqual(perf.summarize_response(3, 'owned', [1], response(reason=reason))['status'], 'completed')
        self.assertEqual(perf.summarize_response(3, 'owned', [1], response(ids=()))['status'], 'invalid')
        value = response(ids=(9,)); value['meta_info']['weave_timing']['last_token_ready_s'] = 12.
        row = perf.summarize_response(3, 'owned', [1], value)
        self.assertEqual(row['status'], 'completed')
        self.assertIsNone(row['decode_tps']); self.assertIsNone(row['tpot_s'])

    def test_common_ids_use_weighted_tps_and_record_quality_difference(self):
        n0 = perf.summarize_response(0, 'owned', [1], response())
        n1 = perf.summarize_response(1, 'owned', [1], response())
        w0 = perf.summarize_response(0, 'owned', [1], response(ids=(4, 5, 6)))
        w0['decode_s'] = 2.
        result = perf.compare_groups({'records': [n0, n1]}, {'records': [w0]})
        self.assertEqual(result['common_completed_row_ids'], [0])
        self.assertEqual(result['native']['decode_tps'], .5)
        self.assertEqual(result['weave']['decode_tps'], 1.)
        self.assertEqual(result['relative_decode_speed'], 2.)
        self.assertEqual(result['quality_diagnostics']['different_output_row_ids'], [0])

    def test_empty_and_identity_mismatch_have_no_comparison(self):
        self.assertIsNone(perf.compare_groups({'records': []}, {'records': []})['relative_decode_speed'])
        a = perf.summarize_response(0, 'owned', [1], response())
        b = perf.summarize_response(0, 'owned', [2], response())
        self.assertEqual(perf.compare_groups({'records': [a]}, {'records': [b]})['status'], 'identity_mismatch')

    def test_stream_counts_actual_incremental_ids_and_keeps_only_first_logprob(self):
        stream = perf.StreamAccumulator('owned')
        first = response(ids=(7,)); first['meta_info']['finish_reason'] = None
        first['meta_info']['output_top_logprobs'] = [[[0., 7]]]
        stream.add(first)
        final = response(ids=(8, 9)); final['meta_info']['completion_tokens'] = 3
        final['meta_info']['weave_timing']['output_tokens'] = 3
        final['meta_info']['output_top_logprobs'] = [[[1., 8]], [[2., 9]]]
        stream.add(final)
        self.assertEqual(stream.response()['output_ids'], [7, 8, 9])
        self.assertEqual(stream.first_token_logprobs['output_top_logprobs'], [[0., 7]])
        self.assertEqual(perf.summarize_response(0, 'owned', [1], stream.response())['output_tokens'], 3)

    def test_pinned_manifest_order_and_deadline_starts_at_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);records=make_records(root)
            result = perf.benchmark_group('http://localhost:1', records, root/'out', ready_monotonic=10., clock=lambda: 1210.)
            self.assertEqual(result['status'], 'deadline')
            self.assertEqual(result['completed_count'], 0)
            self.assertEqual(result['warmup_count'], 0)
            self.assertEqual(len(result['records']), 0)
            self.assertEqual(json.loads((root/'out'/'summary.json').read_text())['status'], 'deadline')

    def test_warmup_budget_reduces_measurement_and_requests_start_at_fixed_row_zero(self):
        clock = [1.]
        observed = []
        class FakeQueue(queue.Queue):
            def get(self, *args, **kwargs):
                item = super().get(*args, **kwargs)
                if item[0] == 'event': clock[0] += 50.
                return item
        class FakeConnection:
            def __init__(self, *args, **kwargs): self.sock = None
            def request(self, method, path, body=None, headers=None):
                self.path = path; self.payload = json.loads(body) if body else None
            def getresponse(self):
                conn = self
                class Reply:
                    status = 200
                    def read(self): return b'{}'
                    def readline(self):
                        if conn.path.endswith('/generate'):
                            observed.append(conn.payload)
                            event = response(); event['meta_info']['id'] = conn.payload['rid']
                            conn.path = '/done'
                            self.line = b'\n'
                            return ('data: '+json.dumps(event)+'\n').encode()
                        line = getattr(self, 'line', b'')
                        self.line = b''
                        return line
                return Reply()
            def close(self): pass
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); records = make_records(root)
            with patch.object(perf.http.client, 'HTTPConnection', FakeConnection), patch.object(perf.queue, 'Queue', FakeQueue):
                result = perf.benchmark_group('http://localhost:1', records, root/'out', ready_monotonic=0., clock=lambda: clock[0])
            self.assertEqual(result['warmup_count'], 2)
            self.assertEqual(result['completed_count'], 21)
            self.assertEqual(result['timeout_count'], 1)
            self.assertEqual(result['records'][-1]['output_tokens'], 3)
            self.assertEqual(result['records'][0]['row_index'], 37)
            self.assertEqual([r['row_index'] for r in result['records']], fixed_order()[:22])
            self.assertEqual(observed[0]['input_ids'], observed[2]['input_ids'])
            self.assertTrue(all(x['sampling_params'] == {'temperature': 0, 'max_new_tokens': 32768, 'ignore_eos': False}
                                and x['return_logprob'] and x['top_logprobs_num'] == 10
                                and x['return_text_in_logprobs'] is False for x in observed))
            checkpoint = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(checkpoint['status'], 'deadline')
            self.assertNotIn('output_ids', json.dumps(checkpoint))
            self.assertNotIn('input_token_ids', json.dumps(checkpoint))

    def test_wrong_order_and_modified_prepared_data_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); records = make_records(root)
            records.write_text(records.read_text().replace('[1, 2]', '[1, 3]', 1))
            with self.assertRaises(ValueError): perf._load_records(records)
            manifest = json.loads((root/'manifest.json').read_text())
            lines = records.read_text().splitlines(); lines[0], lines[1] = lines[1], lines[0]
            records.write_text('\n'.join(lines)+'\n')
            manifest['record_sha256'] = hashlib.sha256(records.read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): perf._load_records(records)

    def test_weighted_metrics_do_not_average_per_row_rates(self):
        a = perf.summarize_response(0, 'owned', [1], response())
        value = response(ids=(1, 2, 3, 4, 5)); value['meta_info']['weave_timing']['last_token_ready_s'] = 13.
        b = perf.summarize_response(1, 'owned', [2], value)
        native = {'records': [a, b]}
        weave = copy.deepcopy(native)
        weave['records'][0]['decode_s'] = 2.
        weave['records'][1]['decode_s'] = 2.
        result = perf.compare_groups(native, weave)
        self.assertAlmostEqual(result['native']['decode_tps'], 1.2)
        self.assertAlmostEqual(result['weave']['decode_tps'], 1.5)
        self.assertAlmostEqual(result['relative_decode_speed'], 1.25)
        self.assertAlmostEqual(result['native']['tpot_s'], 5/6)

    def test_cancellation_checkpoints_partial_row_before_raising(self):
        class InterruptQueue(queue.Queue):
            count = 0
            def get(self, *args, **kwargs):
                self.count += 1
                if self.count == 2: raise KeyboardInterrupt('cancelled')
                return super().get(*args, **kwargs)
        class PartialConnection:
            def __init__(self, *args, **kwargs): self.sock = None
            def request(self, method, path, body=None, headers=None):
                self.path = path; self.payload = json.loads(body) if body else None
            def getresponse(self):
                conn = self
                class Reply:
                    status = 200
                    def read(self): return b'{}'
                    def readline(self):
                        if not hasattr(self, 'lines'):
                            event = response(ids=(7,)); event['meta_info']['finish_reason'] = None; event['meta_info']['id'] = conn.payload['rid']
                            self.lines = iter([('data: '+json.dumps(event)+'\n').encode(), b'\n'])
                        try: return next(self.lines)
                        except StopIteration: time.sleep(.3); return b''
                return Reply()
            def close(self): pass
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); records = make_records(root)
            with patch.object(perf.http.client, 'HTTPConnection', PartialConnection), patch.object(perf.queue, 'Queue', InterruptQueue):
                with self.assertRaises(KeyboardInterrupt):
                    perf.benchmark_group('http://localhost:1', records, root/'out', ready_monotonic=time.monotonic()-121.)
            result = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(result['status'], 'cancelled')
            self.assertEqual(result['completed_count'], 0)
            self.assertEqual(result['records'][0]['output_tokens'], 1)
            self.assertEqual(result['records'][0]['status'], 'cancelled')
            raw = Path(__file__).resolve().parents[2]/result['raw_artifact_relative_path']
            self.assertEqual(json.loads(next(raw.glob('*.json')).read_text())['output_ids'], [7])

    def test_stream_eof_without_terminal_reason_is_aborted_as_partial(self):
        class TruncatedConnection:
            def __init__(self, *args, **kwargs): self.sock = None
            def request(self, *args, **kwargs): pass
            def getresponse(self):
                class Reply:
                    status = 200
                    lines = iter([b'data: {"output_ids": [7], "meta_info": {"id": "only-mine", "finish_reason": null}}\n', b'\n', b''])
                    def readline(self): return next(self.lines)
                return Reply()
            def close(self): pass
        with patch.object(perf.http.client, 'HTTPConnection', TruncatedConnection), patch.object(perf, '_abort_request') as abort:
            stream, state = perf._stream_request('http://localhost:1', {'rid': 'only-mine'}, time.monotonic()+1., time.monotonic)
            self.assertEqual(state, 'partial')
            self.assertEqual(stream.response()['output_ids'], [7])
            abort.assert_called_once_with('http://localhost:1', 'only-mine')

    def _assert_delayed_connect_cleanup(self, *, control, cancelled):
        connection, started, release, closed, finished, log = delayed_connection_fixture()
        class CancelQueue(queue.Queue):
            def get(self, *args, **kwargs):
                if not started.wait(.5): raise AssertionError('worker did not enter connect')
                raise KeyboardInterrupt('cancel while connect pending')
        boundary = patch.object(perf.queue, 'Queue', CancelQueue) if cancelled else patch.object(perf.queue, 'Queue', queue.Queue)
        try:
            with patch.object(perf.http.client, 'HTTPConnection', connection), patch.object(perf, '_abort_request'), boundary:
                begin = time.monotonic()
                if control:
                    with self.assertRaises(KeyboardInterrupt if cancelled else TimeoutError):
                        perf._json_request('http://localhost:1', '/flush_cache', timeout=.025)
                else:
                    stream, state = perf._stream_request('http://localhost:1', {'rid': 'owned'}, begin+.025, time.monotonic)
                    self.assertEqual(state, 'cancelled' if cancelled else 'deadline')
                    self.assertEqual(stream.ids, [])
                self.assertTrue(started.is_set())
                self.assertLess(time.monotonic()-begin, .2)
                self.assertFalse(any(kind == 'wire_send' for kind, _ in log))
                release.set()
                self.assertTrue(finished.wait(.5))
                self.assertFalse(any(kind == 'wire_send' for kind, _ in log), log)
                self.assertTrue(closed.wait(.5), log)
        finally:
            release.set()

    def test_delayed_connect_deadline_never_submits_late_generate(self):
        self._assert_delayed_connect_cleanup(control=False, cancelled=False)

    def test_delayed_connect_cancellation_never_submits_late_generate(self):
        self._assert_delayed_connect_cleanup(control=False, cancelled=True)

    def test_delayed_connect_deadline_never_submits_late_flush(self):
        self._assert_delayed_connect_cleanup(control=True, cancelled=False)

    def test_delayed_connect_cancellation_never_submits_late_flush(self):
        self._assert_delayed_connect_cleanup(control=True, cancelled=True)

    def test_stop_blocks_automatic_reconnect_when_send_resumes_late(self):
        started, release, finished = (threading.Event() for _ in range(3))
        log = []
        class PausedSend(http.client.HTTPConnection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                def create(*args, **kwargs):
                    log.append('socket_created')
                    raise AssertionError('automatic reconnect after stop')
                self._create_connection = create
            def send(self, data):
                started.set(); release.wait(1.)
                return super().send(data)
            def request(self, *args, **kwargs):
                try: return super().request(*args, **kwargs)
                finally: finished.set()
        try:
            with patch.object(perf.http.client, 'HTTPConnection', PausedSend), patch.object(perf, '_abort_request'):
                _, state = perf._stream_request('http://localhost:1', {'rid': 'owned'}, time.monotonic()+.025, time.monotonic)
                self.assertEqual(state, 'deadline')
                self.assertTrue(started.is_set())
                release.set()
                self.assertTrue(finished.wait(.5))
                self.assertEqual(log, [])
        finally: release.set()

    def test_socket_assignment_after_controller_stop_remains_closed(self):
        started, release, closed, finished = (threading.Event() for _ in range(4))
        log = []
        class Transport:
            def shutdown(self, *args): pass
            def close(self): closed.set()
            def sendall(self, data): log.append(bytes(data))
            def makefile(self, *args): raise OSError('no response')
        class PausedPublication(http.client.HTTPConnection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._create_connection = lambda *args, **kwargs: Transport()
            def connect(self):
                transport = self._create_connection((self.host, self.port), self.timeout)
                started.set(); release.wait(1.)
                self.sock = transport
            def request(self, *args, **kwargs):
                try: return super().request(*args, **kwargs)
                finally: finished.set()
        try:
            with patch.object(perf.http.client, 'HTTPConnection', PausedPublication), patch.object(perf, '_abort_request'):
                _, state = perf._stream_request('http://localhost:1', {'rid': 'owned'}, time.monotonic()+.025, time.monotonic)
                self.assertEqual(state, 'deadline')
                self.assertTrue(started.is_set())
                self.assertTrue(closed.is_set())
                release.set()
                self.assertTrue(finished.wait(.5))
                self.assertEqual(log, [])
        finally: release.set()

    def test_blocked_cache_flush_read_respects_absolute_timeout(self):
        class SlowControl:
            def __init__(self, *args, **kwargs): self.sock = None
            def request(self, *args, **kwargs): pass
            def getresponse(self):
                class Reply:
                    status = 200
                    def read(self): time.sleep(.3); return b'Cache flushed.'
                return Reply()
            def close(self): pass
        with patch.object(perf.http.client, 'HTTPConnection', SlowControl):
            start = time.monotonic()
            with self.assertRaises(TimeoutError): perf._json_request('http://localhost:1', '/flush_cache', timeout=.04)
            self.assertLess(time.monotonic()-start, .2)

    def test_group_deadline_during_cache_flush_counts_timeout(self):
        class SlowControl:
            def __init__(self, *args, **kwargs): self.sock = None
            def request(self, *args, **kwargs): pass
            def getresponse(self):
                class Reply:
                    status = 200
                    def read(self): time.sleep(.3); return b'Cache flushed.'
                return Reply()
            def close(self): pass
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); records = make_records(root)
            with patch.object(perf.http.client, 'HTTPConnection', SlowControl):
                result = perf.benchmark_group('http://localhost:1', records, root/'out', ready_monotonic=time.monotonic()-1199.94)
            self.assertEqual(result['timeout_count'], 1)
            self.assertEqual(result['records'][0]['status'], 'deadline')
            self.assertEqual(result['completed_count'], 0)

    def test_native_flush_cache_plain_text_response_succeeds(self):
        class FlushConnection:
            def __init__(self, *args, **kwargs): pass
            def request(self, *args, **kwargs): pass
            def getresponse(self):
                class Reply:
                    status = 200
                    def read(self): return b'Cache flushed.\nPlease check backend logs for more details. '
                return Reply()
            def close(self): pass
        with patch.object(perf.http.client, 'HTTPConnection', FlushConnection):
            value = perf._json_request('http://localhost:1', '/flush_cache')
            self.assertTrue(value.startswith('Cache flushed.'))

    def test_slow_http_read_aborts_only_owned_request_and_preserves_partial(self):
        class BlockingConnection:
            def __init__(self, *args, **kwargs): self.sock = None
            def request(self, method, path, body=None, headers=None): pass
            def getresponse(self): time.sleep(.3); raise OSError('closed')
            def close(self): pass
        with patch.object(perf.http.client, 'HTTPConnection', BlockingConnection), patch.object(perf, '_abort_request') as abort:
            start = time.monotonic()
            stream, state = perf._stream_request('http://localhost:1', {'rid': 'only-mine'}, start+.04, time.monotonic)
            self.assertEqual(state, 'deadline')
            self.assertLess(time.monotonic()-start, .2)
            abort.assert_called_once_with('http://localhost:1', 'only-mine')
            self.assertEqual(stream.response()['output_ids'], [])


class TimingTest(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(timing, 'request timing module required')

    def test_native_one_token_last_is_first_and_invalid_stats_rejected(self):
        class Stats: created_time=10.; first_token_time=12.; finished_time=12.5
        value = timing.timing_metadata(Stats(), 1, native_source_sha256='b'*64)
        self.assertEqual(value['last_token_ready_s'], 12.)
        with self.assertRaises(ValueError): timing.timing_metadata(Stats(), 0, native_source_sha256='b'*64)
        Stats.first_token_time = 9.
        with self.assertRaises(ValueError): timing.timing_metadata(Stats(), 2, native_source_sha256='b'*64)

    def test_timing_wrapper_preserves_native_metadata_and_reports_invalid_empty(self):
        class Stats:
            created_time=10.; first_token_time=12.; finished_time=16.
            def convert_to_output_meta_info(self, scheduler_time_stats=None, completion_tokens=0):
                return {'native': scheduler_time_stats, 'count': completion_tokens}
        timing.patch_timing_class(Stats)
        result = Stats().convert_to_output_meta_info('scheduler', 3)
        self.assertEqual(result['native'], 'scheduler')
        self.assertEqual(result['weave_timing']['last_token_ready_s'], 16.)
        self.assertIs(result['weave_timing'].get('valid', True), True)
        empty = Stats().convert_to_output_meta_info(None, 0)
        self.assertIs(empty['weave_timing']['valid'], False)

    def test_modified_native_source_rejected_before_execution(self):
        class Loader:
            executed = False
            def exec_module(self, module): self.executed = True
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'req_time_stats.py'; path.write_text('changed')
            loader = Loader()
            with self.assertRaises(ImportError): timing.TimingLoader(loader, path).exec_module(object())
            self.assertFalse(loader.executed)

    def test_hook_import_does_not_import_torch_or_sglang(self):
        root = Path(__file__).resolve().parents[2]
        env = dict(os.environ, WEAVE_REQUEST_TIMING='1', PYTHONPATH=str(root/'timing_hooks'))
        run = subprocess.run([sys.executable, '-c',
            "import sys; assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules; "
            "assert any(type(x).__name__ == 'TimingFinder' for x in sys.meta_path)"], env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == '__main__': unittest.main()
