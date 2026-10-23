"""Service IPC release must follow all-rank HTTP preparation, never TP collectives."""
import json
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from src.weave.graph_observer import GraphObserver
from src.weave.native_tail_runtime import NativeTailRuntime
from src.weave.native_tail_transport import NativeTailTransport
from scripts import run_native_tail as controller


class LocalTransportCloseTests(unittest.TestCase):
    def transport(self):
        transport = NativeTailTransport.__new__(NativeTailTransport)
        transport.closed = transport.peers_disconnected = False
        transport.ws = Mock()
        transport.dist = NS(all_gather_object=Mock(side_effect=AssertionError('scheduler collective collision')))
        transport.group = object()
        return transport

    def test_service_disconnect_and_release_are_local_and_ordered(self):
        transport = self.transport()
        with self.assertRaisesRegex(RuntimeError, 'disconnect'):
            transport.release_local()
        transport.disconnect_peers()
        transport.disconnect_peers()
        transport.release_local()
        transport.release_local()
        transport.ws.disconnect.assert_called_once()
        transport.ws.close.assert_called_once()
        transport.dist.all_gather_object.assert_not_called()
        self.assertTrue(transport.closed)

    def test_standalone_close_keeps_its_explicit_collective_barrier(self):
        transport = self.transport()
        events = []
        transport.ws.disconnect.side_effect = lambda: events.append('disconnect')
        transport.ws.close.side_effect = lambda: events.append('release')
        def barrier(output, error, group):
            self.assertIsNone(error)
            events.append('barrier')
            output[:] = [None] * 4
        transport.dist.all_gather_object.side_effect = barrier
        transport.close()
        self.assertEqual(events, ['disconnect', 'barrier', 'release'])


class RuntimeCloseTests(unittest.TestCase):
    def runtime(self, directory, reset=None):
        runtime = NativeTailRuntime.__new__(NativeTailRuntime)
        runtime.rank = 3
        active = [0]
        @contextmanager
        def device(rank):
            previous, active[0] = active[0], rank
            try:
                yield
            finally:
                active[0] = previous
        def owning_call():
            self.assertEqual(active[0], 3)
        graph = NS(reset=Mock(side_effect=reset or owning_call))
        runtime.torch = NS(cuda=NS(device=device, synchronize=Mock(side_effect=lambda rank: owning_call())))
        runtime.graph_backend = NS(_graphs={0: graph}, cleanup=Mock(side_effect=owning_call))
        runtime.executor = NS(disconnect_peers=Mock(side_effect=owning_call),
                              release_local=Mock(side_effect=owning_call),
                              close=Mock(side_effect=AssertionError('service must not call collective close')))
        runtime._init_cleanup(Path(directory))
        return runtime, graph

    def test_prepare_preserves_local_allocation_until_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, graph = self.runtime(directory)
            with self.assertRaisesRegex(RuntimeError, 'prepare'):
                runtime.close()
            prepared = runtime.prepare_close()
            self.assertEqual(prepared['status'], 'prepared')
            self.assertEqual(prepared['graphs_reset'], 1)
            self.assertTrue(prepared['peers_disconnected'])
            self.assertFalse(prepared['ipc_closed'])
            runtime.executor.release_local.assert_not_called()
            self.assertEqual(runtime.prepare_close(), prepared)
            complete = runtime.close()
            self.assertEqual(complete['status'], 'complete')
            self.assertTrue(complete['ipc_closed'])
            self.assertEqual(runtime.close(), complete)
            graph.reset.assert_called_once()
            runtime.executor.release_local.assert_called_once()
            runtime.executor.close.assert_not_called()
            self.assertEqual(json.loads((Path(directory)/'cleanup-rank3.json').read_text()), complete)

    def test_status_and_journal_remain_readable_while_graph_reset_blocks(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_reset():
            entered.set()
            self.assertTrue(release.wait(3))
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self.runtime(directory, blocked_reset)
            errors = []
            def prepare():
                try: runtime.prepare_close()
                except BaseException as error: errors.append(error)
            thread = threading.Thread(target=prepare)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                state = runtime.cleanup_status()
                self.assertEqual(state['phase'], 'reset_graphs')
                self.assertEqual(state['graphs_reset'], 0)
                self.assertEqual(json.loads((Path(directory)/'cleanup-rank3.json').read_text())['phase'], 'reset_graphs')
            finally:
                release.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])

    def test_graph_reset_failure_is_terminal_and_does_not_free_ipc(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self.runtime(directory, Mock(side_effect=RuntimeError('reset failed')))
            with self.assertRaises(RuntimeError): runtime.prepare_close()
            state = runtime.cleanup_status()
            self.assertEqual((state['status'], state['phase'], state['error_type']), ('failed', 'reset_graphs', 'RuntimeError'))
            runtime.executor.disconnect_peers.assert_not_called()
            runtime.executor.release_local.assert_not_called()
            with self.assertRaises(RuntimeError): runtime.close()
            with self.assertRaises(RuntimeError): runtime.prepare_close()


class CleanupAdmissionTests(unittest.TestCase):
    def test_terminal_close_excludes_new_model_operations(self):
        observer = GraphObserver(0, {}, 'source')
        with observer.operation():
            with self.assertRaises(ValueError): observer.begin_cleanup()
            self.assertFalse(observer.closing)
        observer.begin_cleanup()
        observer.begin_cleanup()
        with self.assertRaisesRegex(RuntimeError, 'closing'):
            with observer.operation(): pass
        self.assertEqual(observer.active, 0)


class CoordinatorCloseTests(unittest.TestCase):
    args = NS(graph_evidence_port_base=30310, cleanup_timeout=3)

    def test_every_rank_prepares_before_any_local_allocation_is_released(self):
        barrier = threading.Barrier(4)
        prepared, calls = set(), []
        lock = threading.Lock()
        def http(url, timeout):
            rank = int(url.split(':')[2].split('/')[0]) - self.args.graph_evidence_port_base
            path = '/' + url.rsplit('/', 1)[1]
            with lock: calls.append(path)
            if path == '/tail-close-prepare':
                with lock: prepared.add(rank)
                barrier.wait(2)
                return dict(rank=rank, status='prepared', graphs_owned=32, graphs_reset=32,
                            peers_disconnected=True, ipc_closed=False)
            self.assertEqual(prepared, set(range(4)))
            self.assertEqual(path, '/tail-close')
            return dict(rank=rank, status='complete', graphs_owned=32, graphs_reset=32,
                        peers_disconnected=True, ipc_closed=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(controller, 'http_json', side_effect=http):
            result = controller.close_tail_service(self.args, Path(directory))
        self.assertEqual(calls, ['/tail-close-prepare']*4 + ['/tail-close']*4)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['graphs_reset'], 128)

    def test_failed_prepare_never_commits_and_uses_durable_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = root/'server'
            (server/'native-tail').mkdir(parents=True)
            for rank in range(4):
                (server/'native-tail'/f'cleanup-rank{rank}.json').write_text(json.dumps(dict(
                    rank=rank, status='preparing', phase='reset_graphs', graphs_owned=32,
                    graphs_reset=5, peers_disconnected=False, ipc_closed=False)))
            with patch.object(controller, 'http_json', side_effect=TimeoutError) as http:
                with self.assertRaises(RuntimeError):
                    controller.close_tail_service(self.args, root, server_dir=server)
            self.assertFalse(any(call.args[0].endswith('/tail-close') for call in http.call_args_list))
            report = json.loads((root/'tail-cleanup.json').read_text())
            self.assertEqual(report['status'], 'unverified')
            self.assertEqual(report['ranks']['0']['phase'], 'reset_graphs')
            self.assertEqual(report['graphs_reset'], 20)
            self.assertFalse(report['ipc_closed'])

    def test_commit_timeout_preserves_successful_peers_without_retry(self):
        calls = []
        def http(url, timeout):
            rank = int(url.split(':')[2].split('/')[0]) - self.args.graph_evidence_port_base
            prepare = url.endswith('/tail-close-prepare')
            calls.append((rank, prepare))
            if not prepare and rank == 2:
                raise TimeoutError()
            return dict(rank=rank, status='prepared' if prepare else 'complete',
                        graphs_owned=32, graphs_reset=32, peers_disconnected=True,
                        ipc_closed=not prepare)
        with tempfile.TemporaryDirectory() as directory, patch.object(controller, 'http_json', side_effect=http):
            with self.assertRaises(RuntimeError): controller.close_tail_service(self.args, Path(directory))
            report = json.loads((Path(directory)/'tail-cleanup.json').read_text())
        self.assertEqual(len(calls), 8)
        self.assertEqual(report['status'], 'unverified')
        self.assertEqual(report['phase'], 'commit')
        self.assertFalse(report['ipc_closed'])
        self.assertEqual(report['ranks']['2']['status'], 'prepared')
        self.assertTrue(all(report['ranks'][str(rank)]['ipc_closed'] for rank in (0, 1, 3)))
        self.assertEqual(report['errors'], [dict(rank=2, phase='commit', error_type='TimeoutError')])

    def test_missing_progress_is_unknown_not_zero_graph_resets(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(controller, 'http_json', side_effect=TimeoutError):
            with self.assertRaises(RuntimeError): controller.close_tail_service(self.args, Path(directory))
            report = json.loads((Path(directory)/'tail-cleanup.json').read_text())
        self.assertIsNone(report['graphs_reset'])
        self.assertIsNone(report['graphs_reset_observed'])
        self.assertEqual(report['progress_rank_count'], 0)


class HttpCleanupTests(unittest.TestCase):
    def test_blocked_prepare_does_not_hold_observer_lock_or_block_status_handler(self):
        from concurrent.futures import ThreadPoolExecutor
        import os
        from urllib.error import HTTPError
        from urllib.request import urlopen
        from src.weave.graph_hooks import start_endpoint
        entered, release = threading.Event(), threading.Event()
        observer = GraphObserver(0, {}, 'source')
        def prepare():
            entered.set()
            if not release.wait(3): raise TimeoutError()
            return dict(status='prepared')
        runtime = NS(executor=object(), prepare_close=prepare,
                     cleanup_status=lambda:dict(phase='reset_graphs'), close=lambda:dict(status='complete'))
        with patch.dict(os.environ, {'WEAVE_GRAPH_EVIDENCE_PORT_BASE':'0'}):
            server = start_endpoint(observer, lambda:None, runtime)
        base = f'http://127.0.0.1:{server.server_port}'
        def get(path):
            with urlopen(base + path, timeout=2) as response: return json.load(response)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(get, '/tail-close-prepare')
                try:
                    self.assertTrue(entered.wait(2))
                    self.assertTrue(observer.closing)
                    self.assertEqual(get('/tail-close-status'), dict(phase='reset_graphs'))
                    self.assertFalse(get('/graph-coverage')['active'])
                    with self.assertRaises(HTTPError) as caught: get('/tail-coverage')
                    self.assertEqual(caught.exception.code, 409)
                finally:
                    release.set()
                self.assertEqual(future.result(), dict(status='prepared'))
        finally:
            release.set()
            server.shutdown()
            server.server_close()
