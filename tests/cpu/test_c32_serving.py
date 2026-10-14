import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

try:
    from src.weave import c32_serving as client
except ImportError:
    client = None

from src.weave.experiment_export import ensure_public


def event(count, *, finish=False):
    return {"text": "PRIVATE_OUTPUT", "meta_info": {
        "completion_tokens": count,
        "finish_reason": {"type": "length"} if finish else None}}


def wire(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def dataset_file(root):
    path = root / "cases.jsonl"
    rows = [{"conversations": [{"value": f"PRIVATE_PROMPT_{i}"},
                               {"value": "PRIVATE_ANSWER"}]} for i in range(9)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows * 200))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class FakeContent:
    def __init__(self, mode, owner, rid):
        self.mode, self.owner, self.rid = mode, owner, rid

    def event(self, count, *, finish=False):
        return self.owner.identify(event(count, finish=finish), self.rid, "stream")

    async def iter_any(self):
        # All 32 requests enter before the first response, as on a live socket.
        await asyncio.sleep(0)
        payload = wire(self.event(1)) + wire(self.event(4))
        for part in (payload[:7], payload[7:21], payload[21:]):
            yield part
        if self.mode == "exhaust":
            yield wire(self.event(1024, finish=True)) + b"data: [DONE]\n\n"
        elif self.mode == "error":
            yield wire({"error": "PRIVATE_SERVER_ERROR"})
        elif self.mode == "missing_terminal":
            yield b"data: [DONE]\n\n"
        elif self.mode == "complete_then_hold" and self.owner.completed < 32:
            self.owner.completed += 1
            yield wire(self.event(1024, finish=True)) + b"data: [DONE]\n\n"
        else:
            await asyncio.sleep(60)


class FakeResponse:
    def __init__(self, owner, path, payload):
        self.owner, self.path, self.payload = owner, path, payload
        self.status = 200
        self.content = FakeContent(owner.mode, owner, payload["rid"])

    async def __aenter__(self):
        if self.path.endswith("/generate"):
            self.owner.active += 1
            self.owner.peak = max(self.owner.peak, self.owner.active)
            self.owner.payloads.append(self.payload)
        return self

    async def __aexit__(self, *args):
        if self.path.endswith("/generate"):
            self.owner.active -= 1
            self.owner.closed += 1

    async def json(self):
        await asyncio.sleep(0)
        count = self.payload["sampling_params"]["max_new_tokens"]
        answer = event(count, finish=True)
        answer["meta_info"]["output_token_logprobs"] = [[-0.1, 7 + i, None] for i in range(count)]
        if self.owner.direct_ids:
            answer["output_ids"] = [7 + i for i in range(count)]
        phase = "warmup" if self.payload["rid"].endswith("-warmup") else "acceptance"
        return self.owner.identify(answer, self.payload["rid"], phase)

    async def read(self):
        return b""


class FakeSession:
    def __init__(self, mode="hold", direct_ids=True, identity_fault=None):
        self.mode, self.direct_ids = mode, direct_ids
        self.identity_fault = identity_fault
        self.payloads, self.aborted = [], []
        self.active = self.peak = self.closed = self.completed = 0

    def identify(self, value, rid, phase):
        fault = self.identity_fault
        target = phase
        if phase == "stream" and value["meta_info"]["completion_tokens"] > 1:
            if fault and fault[0] == "stream_late":
                target = "stream_late"
        if fault and fault[0] == target:
            if fault[1] == "stale":
                value["meta_info"]["id"] = "PRIVATE_STALE_ID"
        else:
            value["meta_info"]["id"] = rid
        return value

    def post(self, path, *, json, **kwargs):
        if path.endswith("/abort_request"):
            self.aborted.append(json["rid"])
        return FakeResponse(self, path, json)


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(client, "C32 streaming client module must be implemented")

    def test_sse_parser_preserves_split_unicode_and_multiple_events(self):
        parser = client.SSEParser()
        data = b": keepalive\r\n\r\n" + b"data: " + json.dumps(
            {"text": "\u4e2d", "meta_info": {"completion_tokens": 1}}, ensure_ascii=False).encode() + b"\n\n"
        data += b"data: [DONE]\r\n\r\n"
        actual = []
        for byte in data:
            actual.extend(parser.feed(bytes([byte])))
        parser.finish()
        self.assertEqual(actual, [{"text": "\u4e2d", "meta_info": {"completion_tokens": 1}}, None])

    def test_parser_rejects_truncated_and_invalid_json_without_echoing_data(self):
        for data in (b"data: PRIVATE_BAD\n\n", b"data: {\"text\":\"PRIVATE_BAD\""):
            with self.subTest(data=data), self.assertRaises(client.ClientError) as caught:
                parser = client.SSEParser()
                parser.feed(data)
                parser.finish()
            self.assertNotIn("PRIVATE_BAD", str(caught.exception))

    def test_incomplete_request_tokens_and_first_chunk_decode_are_included(self):
        meter = client.WindowMetrics(10.0, 20.0, output_tokens=1024)
        meter.start(0, 10.0)
        meter.observe(0, event(3), 11.0)
        meter.observe(0, event(3), 12.0)  # Final/metadata repeat is not new tokens.
        meter.observe(0, event(8), 19.0)
        meter.observe(0, event(10), 20.0)  # Half-open window excludes the deadline.
        value = meter.summary()
        self.assertEqual(value["output_tokens_in_window"], 8)
        self.assertEqual(value["decode_tokens_in_window"], 7)
        self.assertEqual(value["requests_unfinished"], 1)
        self.assertEqual(value["requests_completed"], 0)
        self.assertEqual(value["decode_tokens_per_second"], 0.7)
        self.assertEqual(value["ttft_ms"]["mean"], 1000.0)
        self.assertEqual(value["chunk_interarrival_ms"]["mean"], 8000.0)
        self.assertEqual(value["single_token_itl_ms"]["count"], 0)
        self.assertFalse(value["all_positive_chunks_single_token"])

    def test_completed_and_incomplete_counts_share_one_window_denominator(self):
        meter = client.WindowMetrics(10.0, 20.0, output_tokens=3)
        meter.start(0, 10.0)
        meter.start(1, 10.0)
        meter.observe(0, event(1), 11.0)
        meter.observe(0, event(2), 12.0)
        meter.observe(0, event(3, finish=True), 13.0)
        meter.observe(1, event(2), 19.0)
        value = meter.summary()
        self.assertEqual(value["output_tokens_in_window"], 5)
        self.assertEqual(value["decode_tokens_in_window"], 3)
        self.assertEqual(value["requests_completed"], 1)
        self.assertEqual(value["requests_unfinished"], 1)
        self.assertEqual(value["single_token_itl_ms"]["count"], 2)
        self.assertEqual(value["single_token_itl_ms"]["mean"], 1000.0)

    def test_missing_regressing_boolean_and_over_limit_counts_fail(self):
        invalid = [{"meta_info": {}}, event(True), event(-1), event(1.5), event(1025),
                   {"error": "PRIVATE_FAILURE"}, event(1, finish=True)]
        for item in invalid:
            with self.subTest(item=item):
                meter = client.WindowMetrics(0.0, 10.0, output_tokens=1024)
                meter.start(0, 0.0)
                with self.assertRaises(client.ClientError):
                    meter.observe(0, item, 1.0)
        meter = client.WindowMetrics(0.0, 10.0, output_tokens=1024)
        meter.start(0, 0.0)
        meter.observe(0, event(5), 1.0)
        with self.assertRaises(client.ClientError):
            meter.observe(0, event(4), 2.0)

    def test_workload_validates_structure_and_hash_preserving_natural_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, digest = dataset_file(Path(tmp))
            with self.assertRaisesRegex(client.ClientError, "dataset_sha256_mismatch"):
                client.load_dataset(path)
            workload = client.load_dataset(path, expected_sha256=digest)
            self.assertEqual(len(workload.prompts), 1800)
            self.assertEqual(workload.prompts[:2], ("PRIVATE_PROMPT_0", "PRIVATE_PROMPT_1"))
            self.assertEqual(workload.identity["dataset_sha256"], digest)
            self.assertEqual(workload.identity["request_order"], "dataset_file_order_once")
            ensure_public(workload.identity)
            self.assertNotIn("PRIVATE", json.dumps(workload.identity))
            path.write_text(json.dumps({"conversations": [None, {}]}) + "\n")
            bad_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(client.ClientError):
                client.load_dataset(path, expected_sha256=bad_digest)

    def test_only_explicit_loopback_http_is_allowed(self):
        for url in ("https://127.0.0.1:3", "http://example.org", "http://localhost:3/path",
                    "http://user:secret@127.0.0.1:3", "http://127.0.0.1:3/?secret=x"):
            with self.subTest(url=url), self.assertRaises(client.ClientError):
                client.validate_base_url(url)
        self.assertEqual(client.validate_base_url("http://localhost:30001/"), "http://localhost:30001")

    def test_acceptance_extracts_ids_and_never_exposes_raw_response_in_public(self):
        answer = event(8, finish=True)
        answer["meta_info"]["output_token_logprobs"] = [[-0.1, 7+i, "PRIVATE_TOKEN"] for i in range(8)]
        proof = client.extract_acceptance(answer, 4)
        self.assertEqual(proof["output_ids"], list(range(7, 15)))
        answer["output_ids"] = [99] * 8
        with self.assertRaises(client.ClientError):
            client.extract_acceptance(answer, 4)
        answer["output_ids"] = list(range(7, 15))
        self.assertEqual(client.extract_acceptance(answer, 4)["output_ids"], list(range(7, 15)))

    def test_acceptance_logprob_values_do_not_gate_exact_token_match(self):
        answer = event(8, finish=True)
        answer["output_ids"] = list(range(7, 15))
        answer["meta_info"]["output_token_logprobs"] = [[float("nan"), 7+i] for i in range(8)]
        proof = client.extract_acceptance(answer, 0)
        self.assertEqual(proof["output_ids"], list(range(7, 15)))
        self.assertEqual(proof["selected_token_logprobs"], [None] * 8)


class AsyncClientTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertIsNotNone(client, "C32 streaming client module must be implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        path, digest = dataset_file(self.root)
        self.workload = client.load_dataset(path, expected_sha256=digest)

    async def test_32_workers_count_unfinished_output_abort_and_close_at_deadline(self):
        session = FakeSession()
        value = await client.run_measurement(session, "http://127.0.0.1:3", self.workload,
                                             self.root, seconds=0.04)
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["requests_started"], 32)
        self.assertEqual(value["requests_unfinished"], 32)
        self.assertEqual(value["output_tokens_in_window"], 128)
        self.assertEqual(value["decode_tokens_in_window"], 96)
        self.assertEqual(session.peak, 32)
        self.assertEqual(session.active, 0)
        self.assertEqual(session.closed, 33)  # One B1 warmup and 32 measured streams.
        self.assertEqual(len(session.aborted), 32)
        self.assertEqual(len(set(session.aborted)), 32)
        marker = json.loads((self.root / "load-window.json").read_text())
        self.assertEqual(marker["status"], "load_finished")
        self.assertAlmostEqual(marker["deadline_monotonic"] - marker["started_monotonic"], 0.04)
        self.assertEqual(session.payloads[0]["sampling_params"]["max_new_tokens"], 8)
        self.assertEqual(session.payloads[1]["text"], "PRIVATE_PROMPT_0")
        self.assertEqual(session.payloads[1]["sampling_params"],
                         {"max_new_tokens": 1024, "temperature": 0, "ignore_eos": True})
        self.assertEqual(value["warmup"]["requests_completed"], 1)
        ensure_public(value)
        self.assertNotIn("PRIVATE", json.dumps(value))

    async def test_continuous_queue_replenishes_requests_in_file_order(self):
        session = FakeSession("complete_then_hold")
        value = await client.run_measurement(session, "http://127.0.0.1:3", self.workload,
                                             self.root, seconds=0.04)
        self.assertEqual(value["requests_completed"], 32)
        self.assertEqual(value["requests_started"], 64)
        self.assertEqual(value["requests_unfinished"], 32)
        self.assertEqual(value["decode_tokens_in_window"], 32 * 1023 + 32 * 3)
        self.assertEqual([row["text"] for row in session.payloads[1:]],
                         list(self.workload.prompts[:64]))

    async def test_stream_error_and_missing_terminal_fail_with_sanitized_evidence(self):
        for mode in ("error", "missing_terminal"):
            with self.subTest(mode=mode):
                value = await client.run_measurement(FakeSession(mode), "http://127.0.0.1:3",
                                                     self.workload, self.root, seconds=0.1)
                self.assertEqual(value["status"], "failed")
                self.assertGreater(value["requests_failed"], 0)
                self.assertIsNone(value["decode_tokens_per_second"])
                self.assertNotIn("PRIVATE", json.dumps(value))

    async def test_early_queue_exhaustion_and_low_concurrency_fail(self):
        for concurrency in (1, 31, 33):
            with self.subTest(concurrency=concurrency), self.assertRaises(client.ClientError):
                await client.run_measurement(FakeSession(), "http://127.0.0.1:3", self.workload,
                                             self.root, seconds=0.1, concurrency=concurrency)
        short = client.Workload(self.workload.prompts[:32], self.workload.identity)
        value = await client.run_measurement(FakeSession("exhaust"), "http://127.0.0.1:3",
                                             short, self.root, seconds=0.2)
        self.assertEqual(value["status"], "failed")
        self.assertIn("load_exhausted_before_deadline", value["error_codes"])

    async def test_cancellation_aborts_owned_requests_and_publishes_failure(self):
        session = FakeSession()
        task = asyncio.create_task(client.run_measurement(session, "http://127.0.0.1:3",
                                                          self.workload, self.root, seconds=30))
        while session.peak < 32:
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(session.active, 0)
        self.assertEqual(len(session.aborted), 32)
        value = json.loads((self.root / "client-summary.json").read_text())
        self.assertEqual(value["status"], "failed")
        self.assertIn("client_cancelled", value["error_codes"])

    async def test_acceptance_runs_32_concurrent_rows_with_private_ordered_ids(self):
        session = FakeSession(direct_ids=False)
        value = await client.run_acceptance(session, "http://127.0.0.1:3", self.workload, self.root)
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["requests_completed"], 32)
        self.assertEqual(value["generated_tokens"], 256)
        self.assertEqual(session.peak, 32)
        private = json.loads((self.root / "acceptance-private.json").read_text())
        self.assertEqual([row["request_index"] for row in private["records"]], list(range(32)))
        self.assertEqual(private["records"][0]["output_ids"], list(range(7, 15)))
        self.assertEqual(value["generation_settings"]["max_new_tokens"], 8)
        ensure_public(value)
        self.assertNotIn("PRIVATE", json.dumps(value))
        self.assertNotIn("output_ids", json.dumps(value))

    async def test_progress_disk_writes_are_at_most_one_per_second(self):
        writes = []
        write = client.atomic_json

        def capture(path, value):
            if path.name == "load-progress.json":
                writes.append(time.monotonic())
            write(path, value)

        with patch.object(client, "atomic_json", side_effect=capture):
            result = await client.run_measurement(FakeSession(), "http://127.0.0.1:3", self.workload,
                                                  self.root, seconds=2.05)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(writes), 2)
        self.assertGreaterEqual(writes[1] - writes[0], 1.0)

    async def test_failed_progress_publication_invalidates_measurement(self):
        write = client.atomic_json

        def fail_progress(path, value):
            if path.name == "load-progress.json":
                raise OSError("PRIVATE_DISK_DETAIL")
            write(path, value)

        with patch.object(client, "atomic_json", side_effect=fail_progress):
            result = await client.run_measurement(FakeSession(), "http://127.0.0.1:3", self.workload,
                                                  self.root, seconds=1.05)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["decode_tokens_per_second"])
        self.assertNotIn("PRIVATE", json.dumps(result))

    async def test_sequential_measurements_never_reuse_wire_ids_or_abort_new_run(self):
        session = FakeSession()
        first = await client.run_measurement(session, "http://127.0.0.1:3", self.workload,
                                              self.root / "first", seconds=0.04)
        first_ids = {payload["rid"] for payload in session.payloads}
        old_aborts = set(session.aborted)
        offset = len(session.payloads)
        second = await client.run_measurement(session, "http://127.0.0.1:3", self.workload,
                                               self.root / "second", seconds=0.04)
        second_ids = {payload["rid"] for payload in session.payloads[offset:]}
        self.assertEqual((first["status"], second["status"]), ("passed", "passed"))
        self.assertEqual(len(first_ids), 33)
        self.assertEqual(len(second_ids), 33)
        self.assertTrue(first_ids.isdisjoint(second_ids))
        self.assertTrue(old_aborts.isdisjoint(second_ids))

    async def test_acceptance_wire_ids_vary_while_logical_comparison_ids_stay_stable(self):
        session = FakeSession()
        first = await client.run_acceptance(session, "http://127.0.0.1:3", self.workload,
                                             self.root / "first")
        first_ids = {payload["rid"] for payload in session.payloads}
        offset = len(session.payloads)
        second = await client.run_acceptance(session, "http://127.0.0.1:3", self.workload,
                                              self.root / "second")
        second_ids = {payload["rid"] for payload in session.payloads[offset:]}
        first_records = json.loads((self.root / "first" / "acceptance-private.json").read_text())["records"]
        second_records = json.loads((self.root / "second" / "acceptance-private.json").read_text())["records"]
        self.assertEqual((first["status"], second["status"]), ("passed", "passed"))
        self.assertTrue(first_ids.isdisjoint(second_ids))
        self.assertEqual(first_records, second_records)
        self.assertEqual(first["generated_ids_sha256"], second["generated_ids_sha256"])
        self.assertEqual(session.aborted, [])

    async def test_warmup_missing_or_stale_identity_stops_both_modes(self):
        for mode in ("acceptance", "measurement"):
            for fault, code in (("missing", "response_request_id_missing"),
                                ("stale", "response_request_id_mismatch")):
                with self.subTest(mode=mode, fault=fault):
                    session = FakeSession(identity_fault=("warmup", fault))
                    if mode == "acceptance":
                        result = await client.run_acceptance(session, "http://127.0.0.1:3", self.workload, self.root)
                    else:
                        result = await client.run_measurement(session, "http://127.0.0.1:3", self.workload,
                                                              self.root, seconds=0.04)
                    self.assertEqual(result["status"], "failed")
                    self.assertIn(code, result["error_codes"])
                    self.assertEqual(len(session.payloads), 1)
                    self.assertNotIn("PRIVATE", json.dumps(result))

    async def test_acceptance_rejects_missing_or_stale_identity_before_recording_ids(self):
        for fault, code in (("missing", "response_request_id_missing"),
                            ("stale", "response_request_id_mismatch")):
            with self.subTest(fault=fault):
                result = await client.run_acceptance(FakeSession(identity_fault=("acceptance", fault)),
                    "http://127.0.0.1:3", self.workload, self.root)
                self.assertEqual(result["status"], "failed")
                self.assertIn(code, result["error_codes"])
                self.assertEqual(result["requests_completed"], 0)
                self.assertEqual(result["generated_tokens"], 0)
                self.assertEqual(json.loads((self.root / "acceptance-private.json").read_text())["records"], [])
                self.assertNotIn("PRIVATE", json.dumps(result))

    async def test_every_sse_event_requires_matching_identity_before_token_accounting(self):
        for phase in ("stream", "stream_late"):
            for fault, code in (("missing", "response_request_id_missing"),
                                ("stale", "response_request_id_mismatch")):
                with self.subTest(phase=phase, fault=fault):
                    result = await client.run_measurement(FakeSession(identity_fault=(phase, fault)),
                        "http://127.0.0.1:3", self.workload, self.root, seconds=0.04)
                    self.assertEqual(result["status"], "failed")
                    self.assertIn(code, result["error_codes"])
                    # Late mismatches may retain earlier verified first tokens,
                    # but must never contribute any unverified decode tokens.
                    self.assertEqual(result["decode_tokens_in_window"], 0)
                    if phase == "stream":
                        self.assertEqual(result["output_tokens_in_window"], 0)
                    self.assertNotIn("PRIVATE", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
