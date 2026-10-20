import asyncio
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def workload():
    return SimpleNamespace(
        records=tuple({'row_index':447-i, 'prompt_sha256':hashlib.sha256(str(i).encode()).hexdigest(),
                       'input_tokens':2, 'input_token_ids':[i,2]} for i in range(448)),
        identity={'record_sha256':'a'*64,'order_seed':42})


def event(count, text, rid, reason=None):
    return {'text':text,'meta_info':{'id':rid,'completion_tokens':count,'finish_reason':reason}}


class Content:
    def __init__(self, payload, kind):
        self.payload, self.kind = payload, kind

    async def iter_any(self):
        await asyncio.sleep(0)
        rid=self.payload['rid']
        values=[event(0,'前',rid),event(1,'a',rid),event(8,'bc',rid),
                event(9,'尾',rid,{'type':'stop','matched':'PRIVATE_STOP'})]
        if self.kind=='different': values[-1]['text']='异'
        if self.kind=='foreign': values[0]['meta_info']['id']='foreign'
        if self.kind=='decreasing': values[-1]['meta_info']['completion_tokens']=7
        if self.kind=='nonfinite': values[0]['meta_info']['latency']=float('nan')
        if self.kind=='missing_text': values[0].pop('text')
        if self.kind=='empty_final': values[-1]['text']=''
        if self.kind=='transport_error': raise RuntimeError('PRIVATE_TRANSPORT_CONTENT')
        if self.kind=='empty':
            for value in values: value['text']=''
        for value in values:
            raw=('data: '+json.dumps(value,ensure_ascii=False)+'\n\n').encode()
            for start in range(0,len(raw),7):
                yield raw[start:start+7]
            if self.kind in ('stall','abort_fail'):
                await asyncio.sleep(10)


class Response:
    status=200
    def __init__(self,payload=None,kind='ok'):
        self.payload=payload
        self.content=Content(payload,kind)
    async def __aenter__(self): return self
    async def __aexit__(self,*args): pass
    async def json(self): return event(2,'warmup',self.payload['rid'],{'type':'stop'})
    async def text(self): return 'Cache flushed.\n'


class Session:
    def __init__(self,kind='ok'):
        self.kind=kind
        self.payloads=[]
        self.aborted=[]
    def post(self,url,json,**kwargs):
        if url.endswith('/abort_request'):
            self.aborted.append(json['rid'])
            response=Response()
            response.status=500 if self.kind=='abort_fail' else 200
            return response
        self.payloads.append(json)
        return Response(json,self.kind if json['stream'] else 'ok')
    def get(self,url,**kwargs): return Response()


class PairClientTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        self.assertIsNotNone(importlib.util.find_spec('src.weave.native_p0_pair_client'),
                             'bounded paired client is not implemented')
        return importlib.import_module('src.weave.native_p0_pair_client')

    async def run_client(self,kind='ok',timeout=.5):
        client=self.client()
        session=Session(kind)
        with tempfile.TemporaryDirectory() as tmp:
            result=await client.run_measurement(session,'http://127.0.0.1:30000',workload(),tmp,
                                                timeout_seconds=timeout)
            self.assertEqual(result,json.loads((Path(tmp)/'paired-client-summary.json').read_text()))
            return result,session

    async def test_incremental_fragmented_utf8_hash_short_eos_and_exact_cohort(self):
        result,session=await self.run_client()
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['requests_completed'],32)
        self.assertEqual(len(session.payloads),64)
        self.assertEqual(len([p for p in session.payloads if p['stream']]),32)
        row=result['rows'][0]
        self.assertEqual(row['output_tokens'],9)
        self.assertEqual(row['output_text_sha256'],hashlib.sha256('前abc尾'.encode()).hexdigest())
        self.assertEqual(row['output_text_bytes'],9)
        self.assertEqual(row['prefix_digests'],[
            {'completion_tokens':1,'sha256':hashlib.sha256('前a'.encode()).hexdigest()},
            {'completion_tokens':8,'sha256':hashlib.sha256('前abc'.encode()).hexdigest()}])
        self.assertEqual(row['finish_reason'],'stop')
        self.assertGreaterEqual(row['last_receipt_monotonic'],row['first_receipt_monotonic'])
        self.assertEqual([r['original_row_index'] for r in result['rows']],list(range(447,415,-1)))
        public=json.dumps(result,ensure_ascii=False)
        for secret in ('PRIVATE_STOP','input_token_ids','input_ids','"rid"','前','尾'):
            self.assertNotIn(secret,public)
        for payload in session.payloads:
            self.assertNotIn(payload['rid'],public)

    def test_selection_payload_identity_ignores_rid_and_tracks_input_ids(self):
        client=self.client()
        source=workload()
        selection=client.selection_identity(source)
        self.assertEqual(selection,client.selection_identity(source))
        source.records[0]['input_token_ids'][0]=999
        changed=client.selection_identity(source)
        self.assertNotEqual(selection['selection_sha256'],changed['selection_sha256'])
        self.assertNotEqual(selection['requests'][0]['request_payload_sha256'],
                            changed['requests'][0]['request_payload_sha256'])
        source.records[40]['input_token_ids'][0]=888
        self.assertEqual(changed,client.selection_identity(source))

    async def test_different_output_content_remains_valid(self):
        a,_=await self.run_client()
        b,_=await self.run_client('different')
        self.assertEqual(b['status'],'passed')
        self.assertEqual(a['selection'],b['selection'])
        self.assertNotEqual(a['rows'][0]['output_text_sha256'],b['rows'][0]['output_text_sha256'])

    async def test_empty_terminal_chunk_keeps_preceding_text(self):
        result,_=await self.run_client('empty_final')
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['rows'][0]['output_text_sha256'],
                         hashlib.sha256('前abc'.encode()).hexdigest())

    async def test_empty_decoded_text_with_valid_tokens_remains_valid(self):
        result,_=await self.run_client('empty')
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['requests_completed'],32)
        self.assertEqual(result['rows'][0]['output_tokens'],9)
        self.assertEqual(result['rows'][0]['output_text_bytes'],0)
        self.assertEqual(result['rows'][0]['output_text_sha256'],
                         hashlib.sha256(b'').hexdigest())

    async def test_transport_error_never_exports_exception_text(self):
        result,_=await self.run_client('transport_error')
        self.assertEqual(result['error_codes'],['client_transport_error'])
        self.assertNotIn('PRIVATE_TRANSPORT_CONTENT',json.dumps(result))

    async def test_timeout_preserves_partial_rows_and_aborts_owned(self):
        result,session=await self.run_client('stall',.02)
        self.assertEqual(result['status'],'failed')
        self.assertIn('measurement_timeout',result['error_codes'])
        self.assertEqual(result['requests_completed'],0)
        self.assertEqual(len(session.aborted),32)
        self.assertTrue(result['cleanup_complete'])
        self.assertEqual(len(result['rows']),32)
        self.assertEqual(result['rows'][0]['output_text_sha256'],hashlib.sha256('前'.encode()).hexdigest())

    async def test_cleanup_failure_cannot_pass(self):
        result,_=await self.run_client('abort_fail',.02)
        self.assertFalse(result['cleanup_complete'])
        self.assertIn('abort_failed',result['error_codes'])

    async def test_invalid_responses_fail_with_safe_codes(self):
        for kind,code in [('foreign','response_request_id_mismatch'),
                          ('decreasing','stream_completion_tokens_decreased'),
                          ('nonfinite','stream_nonfinite_value'),
                          ('missing_text','stream_text_required')]:
            with self.subTest(kind=kind):
                result,_=await self.run_client(kind)
                self.assertEqual(result['status'],'failed')
                self.assertIn(code,result['error_codes'])

    async def test_controller_handshake_and_cancellation_cleanup(self):
        client=self.client()
        session=Session('stall')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            task=asyncio.create_task(client.run_measurement(
                session,'http://127.0.0.1:30000',workload(),tmp,controller_handshake=True))
            for _ in range(100):
                if (path/'warmup-ready.json').exists(): break
                await asyncio.sleep(.001)
            self.assertTrue((path/'warmup-ready.json').exists())
            self.assertFalse((path/'paired-load-start.json').exists())
            (path/'measurement-go.json').write_text('{}')
            for _ in range(100):
                if (path/'paired-load-start.json').exists(): break
                await asyncio.sleep(.001)
            await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            result=json.loads((path/'paired-client-summary.json').read_text())
            self.assertIn('client_cancelled',result['error_codes'])
            self.assertTrue(result['cleanup_complete'])
            self.assertEqual(len(session.aborted),32)

    async def test_handshake_timeout_never_dispatches_measurement(self):
        client=self.client()
        session=Session()
        with tempfile.TemporaryDirectory() as tmp, patch.object(client,'HANDSHAKE_TIMEOUT',.01):
            result=await client.run_measurement(session,'http://127.0.0.1:30000',
                                                workload(),tmp,controller_handshake=True)
            self.assertEqual(result['error_codes'],['controller_handshake_timeout'])
            self.assertTrue(result['cleanup_complete'])
            self.assertTrue(all(not p['stream'] for p in session.payloads))
            self.assertFalse((Path(tmp)/'paired-load-start.json').exists())

    def test_preview_validates_real_prepared_source_without_http_or_files(self):
        from scripts import bench_native_p0_pair as cli
        records=Path(__file__).resolve().parents[2]/'data/gpqa/prepared/deepseek-v4-flash/main.jsonl'
        if not records.exists():
            self.skipTest('pinned prepared local corpus unavailable')
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as output:
            destination=Path(tmp)/'preview'
            with patch.object(cli,'execute',side_effect=AssertionError('preview invoked HTTP')):
                code=cli.main(['--records',str(records),'--output-dir',str(destination),
                               '--base-url','http://127.0.0.1:30000'])
            value=json.loads(output.getvalue())
            self.assertEqual(code,0)
            self.assertEqual(value['status'],'preview_not_executed')
            self.assertFalse(destination.exists())
            self.assertEqual(value['selection']['requests'][0]['original_row_index'],37)
            self.assertEqual(value['selection']['selection_sha256'],
                             'a5748d54c4605d57042784ad3e2de569ae017796411239a360f90aaabc9b3018')
            self.assertEqual(value['timeout_seconds'],3600)


if __name__=='__main__':
    unittest.main()
