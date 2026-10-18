import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.weave.c32_serving import ClientError, WindowMetrics


def event(count, reason=None, rid='owned'):
    return {'meta_info': {'id': rid, 'completion_tokens': count, 'finish_reason': reason}}


class EosPolicyTests(unittest.TestCase):
    def test_normal_eos_opt_in_and_legacy_strict_default(self):
        value = event(2, {'type': 'stop', 'matched': 151645})
        strict = WindowMetrics(1, 10, output_tokens=8)
        strict.start(0, 1)
        with self.assertRaises(ClientError):
            strict.observe(0, value, 2)
        natural = WindowMetrics(1, 10, output_tokens=8, allow_normal_eos=True)
        natural.start(0, 1)
        self.assertTrue(natural.observe(0, value, 2))
        self.assertEqual(natural.decode_tokens, 1)

    def test_eos_still_rejects_abort_and_short_length(self):
        for reason in ({'type': 'abort'}, {'type': 'length'}):
            meter = WindowMetrics(1, 10, output_tokens=8, allow_normal_eos=True)
            meter.start(0, 1)
            with self.assertRaises(ClientError):
                meter.observe(0, event(2, reason), 2)


class FakeContent:
    def __init__(self, payload, stall=False, foreign=False):
        self.payload, self.stall, self.foreign = payload, stall, foreign
    async def iter_any(self):
        await asyncio.sleep(0)
        rid = 'foreign' if self.foreign else self.payload['rid']
        yield ('data: '+json.dumps(event(2, rid=rid))+'\n\n').encode()
        if self.stall:
            await asyncio.sleep(10)
        yield ('data: '+json.dumps(event(3, {'type':'stop'}, rid))+'\n\n').encode()


class FakeResponse:
    status = 200
    def __init__(self, payload=None, stall=False, foreign=False):
        self.payload = payload
        self.content = FakeContent(payload, stall, foreign)
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def text(self): return 'Cache flushed.\n'
    async def json(self):
        return event(2, {'type':'stop'}, self.payload['rid']) if self.payload else 'Cache flushed.'


class FakeSession:
    def __init__(self, stall=False, foreign=False, abort_fail=False):
        self.payloads=[]; self.aborted=[]; self.stall=stall; self.foreign=foreign; self.abort_fail=abort_fail
    def post(self, url, json, **kwargs):
        if url.endswith('/abort_request'):
            self.aborted.append(json['rid'])
            result=FakeResponse()
            result.status=500 if self.abort_fail else 200
            return result
        self.payloads.append(json)
        return FakeResponse(json, self.stall and json['stream'], self.foreign and json['stream'])
    def get(self, url, **kwargs): return FakeResponse()


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def run_client(self, session):
        from src.weave import gpqa_c32_client as client
        from types import SimpleNamespace
        workload=SimpleNamespace(records=tuple({'row_index':i, 'input_token_ids':[1,2]} for i in range(448)),
                                 identity={'record_sha256':'a'*64})
        with tempfile.TemporaryDirectory() as tmp, patch.object(client, 'WINDOW_SECONDS', .2):
            result=await client.run_measurement(session, 'http://127.0.0.1:30000', workload, tmp)
            rows=json.loads((Path(tmp)/'gpqa-rows.json').read_text())
            self.assertEqual((Path(tmp)/'measurement-summary.json').stat().st_mode & 0o777, 0o664)
            return result,rows

    async def test_exhaustion_once_full_window_and_payload_privacy(self):
        session=FakeSession()
        result,rows=await self.run_client(session)
        self.assertEqual(result['status'],'passed')
        self.assertTrue(result['dataset_exhausted'])
        self.assertEqual(result['dataset_exhausted_monotonic'],result['last_completed_monotonic'])
        self.assertAlmostEqual(result['idle_seconds'],result['deadline_monotonic']-result['last_completed_monotonic'])
        self.assertEqual(result['requests_completed'],448)
        self.assertEqual(result['decode_tokens_in_window'],896)
        self.assertEqual(len(rows['rows']),448)
        self.assertEqual(len(session.payloads),480)
        self.assertTrue(all(p['sampling_params']['ignore_eos'] is False and not p['return_logprob'] for p in session.payloads))
        self.assertNotIn('input_token_ids',json.dumps(rows))
        self.assertFalse(result['accuracy_evaluated'])

    async def test_deadline_counts_unfinished_and_aborts_only_owned(self):
        session=FakeSession(stall=True)
        result,rows=await self.run_client(session)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['requests_unfinished'],32)
        self.assertEqual(result['decode_tokens_in_window'],32)
        self.assertEqual(len(session.aborted),32)
        self.assertFalse(result['dataset_exhausted'])
        self.assertTrue(result['cleanup_complete'])

    async def test_foreign_identity_fails(self):
        result,_=await self.run_client(FakeSession(foreign=True))
        self.assertEqual(result['status'],'failed')
        self.assertIn('response_request_id_mismatch',result['error_codes'])

    async def test_abort_failure_never_passes(self):
        result,_=await self.run_client(FakeSession(stall=True,abort_fail=True))
        self.assertEqual(result['status'],'failed')
        self.assertFalse(result['cleanup_complete'])
        self.assertIn('abort_failed',result['error_codes'])

class ExtendedClientTests(ClientTests):
    async def test_summary_keeps_nested_identity_for_controller(self):
        result,_=await self.run_client(FakeSession(stall=True))
        self.assertEqual(result['workload'],{'record_sha256':'a'*64})

    async def test_cancellation_persists_failure_and_cleanup(self):
        from src.weave import gpqa_c32_client as client
        from types import SimpleNamespace
        workload=SimpleNamespace(records=tuple({'row_index':i,'input_token_ids':[1]} for i in range(448)),identity={})
        session=FakeSession(stall=True)
        with tempfile.TemporaryDirectory() as tmp:
            task=asyncio.create_task(client.run_measurement(session,'http://127.0.0.1:30000',workload,tmp))
            for _ in range(100):
                if (Path(tmp)/'load-window.json').exists(): break
                await asyncio.sleep(.001)
            await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            result=json.loads((Path(tmp)/'measurement-summary.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertIn('client_cancelled',result['error_codes'])
            self.assertTrue(result['cleanup_complete'])
            self.assertEqual(len(session.aborted),32)

    def test_window_excludes_deadline_event_and_validates_native_length(self):
        meter=WindowMetrics(1,10,output_tokens=8,allow_normal_eos=True)
        meter.start(0,1)
        meter.observe(0,event(2),2)
        self.assertFalse(meter.observe(0,event(8,{'type':'length'}),10))
        self.assertEqual(meter.output_tokens,2)
        self.assertEqual(meter.states[0]['status'],'unfinished')
        self.assertTrue(meter.observe(0,event(8,{'type':'length'}),9))

    async def test_handshake_waits_for_parent_before_marker(self):
        from src.weave import gpqa_c32_client as client
        from types import SimpleNamespace
        workload=SimpleNamespace(records=tuple({'row_index':i,'input_token_ids':[1]} for i in range(448)),identity={})
        with tempfile.TemporaryDirectory() as tmp,patch.object(client,'WINDOW_SECONDS',.1):
            path=Path(tmp)
            task=asyncio.create_task(client.run_measurement(FakeSession(stall=True),'http://127.0.0.1:30000',workload,path,controller_handshake=True))
            for _ in range(100):
                if (path/'warmup-ready.json').exists(): break
                await asyncio.sleep(.001)
            self.assertTrue((path/'warmup-ready.json').exists())
            self.assertFalse((path/'load-window.json').exists())
            (path/'measurement-go.json').write_text('{}')
            result=await task
            self.assertEqual(result['status'],'passed')

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    aiohttp = web = None


@unittest.skipIf(aiohttp is None, 'aiohttp available in existing serving image or connected development environment')
class RealHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_loopback_sse_and_owned_deadline_cleanup(self):
        from src.weave import gpqa_c32_client as client
        from types import SimpleNamespace
        payloads=[];aborted=[];flushes=[]
        release=asyncio.Event()
        async def generate(request):
            payload=await request.json();payloads.append(payload)
            if not payload['stream']:
                return web.json_response(event(2,{'type':'stop'},payload['rid']))
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            data=('data: '+json.dumps(event(3,rid=payload['rid']))+'\n\n').encode()
            try:
                await response.write(data[:9]);await response.write(data[9:])
                await release.wait()
                await response.write_eof()
            except (ConnectionResetError,asyncio.CancelledError):
                pass
            return response
        async def abort(request):
            aborted.append((await request.json())['rid'])
            return web.json_response({'success':True})
        async def flush(request):
            flushes.append(len(payloads))
            self.assertEqual(request.query.get('timeout'),'0')
            return web.Response(text='Cache flushed.\n')
        app=web.Application();app.router.add_post('/generate',generate)
        app.router.add_post('/abort_request',abort);app.router.add_get('/flush_cache',flush)
        runner=web.AppRunner(app,shutdown_timeout=.1);await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        workload=SimpleNamespace(records=tuple({'row_index':i,'input_token_ids':[123,456]} for i in range(448)),identity={})
        try:
            with tempfile.TemporaryDirectory() as tmp,patch.object(client,'WINDOW_SECONDS',.25):
                async with aiohttp.ClientSession(trust_env=False) as session:
                    result=await client.run_measurement(session,f'http://127.0.0.1:{port}',workload,tmp)
                self.assertEqual(result['status'],'passed')
                self.assertEqual(result['requests_unfinished'],32)
                self.assertEqual(result['decode_tokens_in_window'],64)
                self.assertEqual(flushes,[32])
                self.assertEqual(len(aborted),32)
                self.assertEqual(set(aborted),{p['rid'] for p in payloads if p['stream']})
        finally:
            release.set();await runner.cleanup()

class CLITests(unittest.TestCase):
    def test_missing_runtime_dependency_has_actionable_safe_error(self):
        import contextlib
        import io
        from scripts import bench_gpqa_c32 as cli
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        with tempfile.TemporaryDirectory() as tmp,patch('src.weave.gpqa_c32.load_gpqa_workload',return_value=SimpleNamespace(identity={})),patch.object(cli,'execute',new=AsyncMock(side_effect=ClientError('aiohttp_missing_use_existing_SGLang_image_no_online_install'))):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                code=cli.main(['--records',str(Path(tmp)/'records.jsonl'),'--output-dir',tmp,'--base-url','http://127.0.0.1:30000','--execute'])
            self.assertEqual(code,1)
            self.assertIn('aiohttp_missing_use_existing_SGLang_image_no_online_install',output.getvalue())

class ExhaustionSchedulingTests(ClientTests):
    async def test_completion_before_deadline_remains_exhausted_after_late_wait_resume(self):
        original_wait=asyncio.wait
        async def late_resume(*args,**kwargs):
            result=await original_wait(*args,**kwargs)
            await asyncio.sleep(.21)
            return result
        with patch('src.weave.gpqa_c32_client.asyncio.wait',side_effect=late_resume):
            result,rows=await self.run_client(FakeSession())
        self.assertEqual(result['requests_completed'],448)
        self.assertTrue(result['dataset_exhausted'])
        self.assertEqual(result['last_completed_monotonic'],max(r['completed_monotonic'] for r in rows['rows']))
        self.assertEqual(result['dataset_exhausted_monotonic'],result['last_completed_monotonic'])
        self.assertGreater(result['idle_seconds'],0)

class RefillSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_refill_failure_vs_last_dispatch(self):
        from src.weave import gpqa_c32_client as client
        from types import SimpleNamespace
        workload=SimpleNamespace(records=tuple({'row_index':i,'input_token_ids':[1]} for i in range(448)),identity={})
        for completed in (417,416):
            async def controlled_stream(session,base_url,payload,index,meter,rows):
                await asyncio.sleep(0)
                now=client.time.monotonic()
                meter.observe(index,event(2,rid=payload['rid']),now)
                if index>=completed:
                    await asyncio.Event().wait()
                now=client.time.monotonic()
                meter.observe(index,event(3,{'type':'stop'},payload['rid']),now)
                rows[index]['completed_monotonic']=now
                rows[index]['finish_reason']={'type':'stop'}
                return True
            with tempfile.TemporaryDirectory() as tmp,patch.object(client,'WINDOW_SECONDS',.2),patch.object(client,'_stream',side_effect=controlled_stream):
                result=await client.run_measurement(FakeSession(),'http://127.0.0.1:30000',workload,tmp)
            with self.subTest(completed=completed):
                self.assertEqual(result['status'],'passed')
                self.assertEqual(result['requests_started'],448)
                self.assertEqual(result['requests_completed'],completed)
                self.assertEqual(result['requests_unfinished'],448-completed)
                self.assertFalse(result['dataset_exhausted'])
                self.assertEqual(result['refill_exhausted'],completed==417)
                if completed==417:
                    self.assertLess(result['refill_exhausted_monotonic'],result['deadline_monotonic'])
                else:self.assertIsNone(result['refill_exhausted_monotonic'])
