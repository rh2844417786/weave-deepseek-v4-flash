import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from src.weave import performance as perf
from src.weave.experiment_export import ensure_public
from tests.cpu.test_gpqa_performance import response


class ServingFixture:
    """Real localhost HTTP, with controllable scheduler-idle acknowledgements."""
    def __init__(self,*,busy_after_generate=0,failure=None,delay=0.):
        self.flushes=0;self.generations=[];self.pending=0
        self.busy_after_generate=busy_after_generate;self.failure=failure;self.delay=delay

    def __enter__(self):
        state=self
        class Handler(BaseHTTPRequestHandler):
            def reply(self,status,body,kind='text/plain'):
                self.send_response(status);self.send_header('Content-Type',kind)
                self.send_header('Content-Length',str(len(body)));self.end_headers()
                try: self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError): pass
            def do_GET(self):
                if self.path!='/flush_cache': self.send_error(404);return
                state.flushes+=1
                if state.delay: time.sleep(state.delay)
                if state.failure: self.reply(*state.failure)
                elif state.pending:
                    state.pending-=1;self.reply(400,b'Flush cache failed.\n')
                else: self.reply(200,b'Cache flushed.\nPlease check backend logs for more details. ')
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path=='/abort_request': self.reply(200,b'');return
                if self.path!='/generate': self.send_error(404);return
                state.generations.append(payload['input_ids'])
                state.pending=state.busy_after_generate
                item=response();item['meta_info']['id']=payload['rid']
                self.reply(200,('data: '+json.dumps(item)+'\n\ndata: [DONE]\n\n').encode(),'text/event-stream')
            def log_message(self,*args): pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'
        return self

    def __exit__(self,*args):
        self.server.shutdown();self.server.server_close();self.thread.join(1)


def records_file(root):
    path=root/'records.jsonl'
    path.write_text(''.join(json.dumps({'row_index':i,'input_token_ids':ids})+'\n'
                            for i,ids in enumerate(([1,2],[3,4]))))
    (root/'manifest.json').write_text(json.dumps({'record_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    return path


class CacheFlushRecoveryTest(unittest.TestCase):
    def test_transient_busy_does_not_skip_the_next_record(self):
        with tempfile.TemporaryDirectory() as tmp,ServingFixture(busy_after_generate=2) as server:
            root=Path(tmp);records=records_file(root)
            with patch.object(perf,'MAX_WARMUPS',0):
                value=perf.benchmark_group(server.url,records,root/'out',ready_monotonic=time.monotonic(),raw_root=root)
            self.assertEqual(value['completed_count'],2)
            self.assertEqual([r['row_index'] for r in value['records']],[0,1])
            self.assertEqual(server.generations,[[1,2],[3,4]])
            self.assertEqual(value['records'][1]['cache_flush']['attempts'],3)
            self.assertEqual(value['records'][1]['cache_flush']['retryable_failures'],2)
            self.assertEqual(value['records'][1]['cache_flush']['status'],'flushed')
            rid=value['records'][1]['request_id']
            raw=json.loads((Path(value['raw_artifact_directory'])/(rid+'.json')).read_text())
            self.assertEqual(len(raw['cache_flush_private_errors']),2)
            self.assertEqual(raw['cache_flush_private_errors'][0]['http_status'],400)

    def test_persistent_busy_is_bounded_and_records_http_reason(self):
        self.assertTrue(hasattr(perf,'flush_cache_when_idle'))
        with ServingFixture(failure=(400,b'Flush cache failed.\n')) as server:
            started=time.monotonic()
            with self.assertRaises(perf.CacheFlushError) as caught:
                perf.flush_cache_when_idle(server.url,deadline=started+5,max_wait=.13)
            report=caught.exception.diagnostic
            self.assertEqual(report['status'],'timeout')
            self.assertGreaterEqual(report['attempts'],2)
            self.assertEqual(report['last_error']['http_status'],400)
            self.assertEqual(report['last_error']['reason'],'scheduler_busy')
            self.assertLess(time.monotonic()-started,.5)

    def test_phase_deadline_clips_retry_window(self):
        self.assertTrue(hasattr(perf,'flush_cache_when_idle'))
        with ServingFixture(failure=(400,b'Flush cache failed.\n')) as server:
            started=time.monotonic()
            with self.assertRaises(perf.CacheFlushError) as caught:
                perf.flush_cache_when_idle(server.url,deadline=started+.13,max_wait=30)
            self.assertEqual(caught.exception.diagnostic['status'],'deadline')
            self.assertLess(time.monotonic()-started,.5)
            attempts=server.flushes;time.sleep(.1)
            self.assertEqual(server.flushes,attempts)

    def test_slow_control_response_cannot_outlive_absolute_deadline(self):
        self.assertTrue(hasattr(perf,'flush_cache_when_idle'))
        with ServingFixture(delay=.3) as server:
            started=time.monotonic()
            with self.assertRaises(perf.CacheFlushError) as caught:
                perf.flush_cache_when_idle(server.url,deadline=started+.08,max_wait=30)
            self.assertEqual(caught.exception.diagnostic['status'],'deadline')
            self.assertLess(time.monotonic()-started,.25)

    def test_permanent_error_stops_group_and_keeps_raw_error_private(self):
        secret=b'allocator failed: PRIVATE_PROMPT_SENTINEL'
        with tempfile.TemporaryDirectory() as tmp,ServingFixture(failure=(500,secret)) as server:
            root=Path(tmp);records=records_file(root)
            with patch.object(perf,'MAX_WARMUPS',0):
                value=perf.benchmark_group(server.url,records,root/'out',ready_monotonic=time.monotonic(),raw_root=root)
            self.assertEqual(len(value['records']),1)
            self.assertEqual(value['status'],'incomplete')
            self.assertEqual(value['stop_reason'],'cache_flush_failed')
            self.assertEqual(server.generations,[])
            self.assertEqual(server.flushes,1)
            self.assertEqual(value['records'][0]['cache_flush']['last_error']['http_status'],500)
            ensure_public(value)
            self.assertNotIn('PRIVATE_PROMPT_SENTINEL',json.dumps(value))
            private=''.join(p.read_text() for p in Path(value['raw_artifact_directory']).glob('*.json'))
            self.assertIn('PRIVATE_PROMPT_SENTINEL',private)

    def test_http200_without_positive_ack_does_not_start_generation(self):
        with tempfile.TemporaryDirectory() as tmp,ServingFixture(failure=(200,b'false')) as server:
            root=Path(tmp);records=records_file(root)
            with patch.object(perf,'MAX_WARMUPS',0):
                value=perf.benchmark_group(server.url,records,root/'out',ready_monotonic=time.monotonic(),raw_root=root)
            self.assertEqual(server.generations,[])
            self.assertEqual(value['records'][0]['status'],'cache_flush_error')
            self.assertEqual(value['records'][0]['cache_flush']['last_error']['reason'],'invalid_acknowledgement')

    def test_unrecognized400_is_not_retried_as_busy(self):
        with ServingFixture(failure=(400,b'Bad parameter: PRIVATE_DETAIL')) as server:
            with self.assertRaises(perf.CacheFlushError) as caught:
                perf.flush_cache_when_idle(server.url,deadline=time.monotonic()+5)
            self.assertEqual(caught.exception.diagnostic['status'],'failed')
            self.assertEqual(caught.exception.diagnostic['attempts'],1)
            self.assertEqual(server.flushes,1)
            self.assertNotIn('PRIVATE_DETAIL',json.dumps(caught.exception.diagnostic))

    def test_warmup_flush_failure_stops_before_formal_requests(self):
        with tempfile.TemporaryDirectory() as tmp,ServingFixture(failure=(500,b'allocator error')) as server:
            root=Path(tmp);records=records_file(root)
            value=perf.benchmark_group(server.url,records,root/'out',ready_monotonic=time.monotonic(),raw_root=root)
            self.assertEqual(value['status'],'incomplete')
            self.assertEqual(value['stop_reason'],'cache_flush_failed')
            self.assertEqual(len(value['warmups']),1)
            self.assertEqual(value['records'],[])
            self.assertEqual(server.generations,[])
