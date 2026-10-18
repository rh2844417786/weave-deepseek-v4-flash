"""Offline C32 client receipt metrics for the pinned, natural custom workload.

No tokenizer, model download, Torch import, server ownership, or GPU probe is
needed here. The caller provides an aiohttp ClientSession from the existing
image; keeping the dependency at that boundary also permits offline CPU tests.
"""
import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

from .calibration import atomic_json


DATASET_SHA256 = "7d7bcdb9996f9293b63d337f21ee977f39013107ef23c2c411de2310d64501c2"
CONCURRENCY = 32
OUTPUT_TOKENS = 1024
ACCEPTANCE_TOKENS = 8
SCHEMA_VERSION = 1


class ClientError(ValueError):
    """A fixed diagnostic code, never a server body or private request value."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Workload:
    prompts: tuple
    identity: dict


def load_dataset(path, *, expected_sha256=DATASET_SHA256):
    """Validate the original nine prompts x200, then retain original file order.

    The optional expected hash is a library-level fixture hook. The executable
    always uses DATASET_SHA256 and provides no identity override argument.
    """
    try:
        payload = Path(path).read_bytes()
    except OSError:
        raise ClientError("dataset_unreadable") from None
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ClientError("dataset_sha256_mismatch")
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeError, ValueError):
        raise ClientError("dataset_invalid_jsonl") from None
    prompts = []
    for row in rows:
        turns = row.get("conversations", row.get("conversation")) if isinstance(row, dict) else None
        if not isinstance(turns, list) or len(turns) < 2 or not all(isinstance(v, dict) for v in turns[:2]):
            raise ClientError("dataset_invalid_turns")
        prompt = turns[0].get("content", turns[0].get("value"))
        answer = turns[1].get("content", turns[1].get("value"))
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(answer, str):
            raise ClientError("dataset_invalid_text_fields")
        prompts.append(prompt)
    hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts]
    counts = Counter(hashes)
    if len(prompts) != 1800 or len(counts) != 9 or set(counts.values()) != {200}:
        raise ClientError("dataset_requires_nine_cases_times_200")
    identity = {"dataset_sha256": digest, "dataset_records": len(prompts),
                "unique_prompts": 9, "repetitions_per_prompt": 200,
                "dataset_format": "sglang_custom_jsonl",
                "request_order": "dataset_file_order_once",
                "request_order_sha256": _digest(list(enumerate(hashes)))}
    return Workload(tuple(prompts), identity)


def validate_dataset(path):
    """Public identity check with the same pinned workload as the executable."""
    return load_dataset(path).identity


def validate_base_url(value):
    try:
        url = urlsplit(value)
        valid = (url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost", "::1"}
                 and url.username is None and url.password is None and not url.query
                 and not url.fragment and url.path in {"", "/"} and url.port is not None
                 and 1 <= url.port <= 65535)
    except ValueError:
        valid = False
    if not valid:
        raise ClientError("explicit_loopback_http_url_required")
    return value.rstrip("/")


def generation_settings(tokens):
    return {"max_new_tokens": tokens, "temperature": 0, "ignore_eos": True}


def _payload(prompt, rid, *, tokens=OUTPUT_TOKENS, acceptance=False):
    return {"text": prompt, "rid": rid, "sampling_params": generation_settings(tokens),
            "stream": not acceptance, "return_logprob": acceptance,
            "return_text_in_logprobs": False}


class SSEParser:
    """Incremental byte parser: one result per SSE event; None denotes DONE."""

    def __init__(self):
        self.buffer = bytearray()
        self.lines = []
        self.done = False

    def feed(self, block):
        self.buffer.extend(block)
        values = []
        while True:
            end = self.buffer.find(b"\n")
            if end < 0:
                break
            line = bytes(self.buffer[:end]).rstrip(b"\r")
            del self.buffer[:end + 1]
            if line.startswith(b"data:"):
                if self.done:
                    raise ClientError("stream_data_after_done")
                self.lines.append(line[5:].lstrip(b" "))
            elif not line and self.lines:
                encoded = b"\n".join(self.lines)
                self.lines.clear()
                if encoded == b"[DONE]":
                    self.done = True
                    values.append(None)
                    continue
                try:
                    value = json.loads(encoded)
                except (UnicodeError, ValueError):
                    raise ClientError("stream_invalid_json") from None
                if not isinstance(value, dict):
                    raise ClientError("stream_event_not_object")
                values.append(value)
        # The pinned server emits incremental text. Bound corrupted streams so
        # an unframed response cannot accumulate indefinitely in client memory.
        if len(self.buffer) + sum(map(len, self.lines)) > 16 * 1024 * 1024:
            raise ClientError("stream_event_too_large")
        return values

    def finish(self):
        if self.buffer.strip() or self.lines:
            raise ClientError("stream_truncated_event")


class _Moments:
    """Bounded-memory aggregate over observations, with units set by the caller."""

    def __init__(self):
        self.count, self.total, self.minimum, self.maximum = 0, 0.0, None, None

    def add(self, value):
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def summary(self):
        return {"count": self.count, "mean": self.total / self.count if self.count else None,
                "min": self.minimum, "max": self.maximum}


def _count_and_terminal(value, maximum, *, allow_normal_eos=False):
    if not isinstance(value, dict) or "error" in value:
        raise ClientError("stream_error_event")
    meta = value.get("meta_info")
    if not isinstance(meta, dict):
        raise ClientError("stream_missing_metadata")
    count = meta.get("completion_tokens")
    if type(count) is not int or not 0 <= count <= maximum:
        raise ClientError("stream_invalid_completion_tokens")
    reason = meta.get("finish_reason")
    if reason is not None:
        kind = reason.get("type") if isinstance(reason, dict) else None
        valid = kind == "length" and count == maximum
        if allow_normal_eos and kind in {"stop", "eos"} and count > 0:
            valid = True
        if not valid:
            raise ClientError("stream_invalid_finish")
    return count, reason is not None


def _validate_response_identity(value, expected_rid):
    meta = value.get("meta_info") if isinstance(value, dict) else None
    rid = meta.get("id") if isinstance(meta, dict) else None
    if not isinstance(rid, str) or not rid:
        raise ClientError("response_request_id_missing")
    if rid != expected_rid:
        raise ClientError("response_request_id_mismatch")


class WindowMetrics:
    """Counts actual cumulative token deltas in [start, deadline) at client receipt.

    These are client-delivery times, including HTTP buffering. They are neither
    GPU decode-step times nor server generation timestamps. Failed runs keep
    their observed counters for diagnosis but have no usable throughput rate.
    """

    def __init__(self, start, deadline, *, output_tokens=OUTPUT_TOKENS, allow_normal_eos=False):
        if not all(math.isfinite(v) for v in (start, deadline)) or deadline <= start:
            raise ClientError("invalid_measurement_window")
        self.started, self.deadline, self.output_limit = start, deadline, output_tokens
        self.allow_normal_eos = allow_normal_eos
        self.states = {}
        self.output_tokens = self.decode_tokens = self.positive_chunks = self.multi_chunks = 0
        self.peak_active = 0
        self.errors = set()
        self.ttft, self.chunk_intervals, self.single_itl = _Moments(), _Moments(), _Moments()

    def start(self, index, timestamp):
        if index in self.states or not self.started <= timestamp < self.deadline:
            raise ClientError("invalid_request_start")
        self.states[index] = {"start": timestamp, "count": 0, "last": None, "last_delta": 0,
                              "status": "unfinished"}
        self.peak_active = max(self.peak_active, sum(v["status"] == "unfinished" for v in self.states.values()))

    def observe(self, index, value, timestamp):
        count, terminal = _count_and_terminal(value, self.output_limit, allow_normal_eos=self.allow_normal_eos)
        row = self.states[index]
        if count < row["count"]:
            raise ClientError("stream_regressing_completion_tokens")
        if timestamp < row["start"] or (row["last"] is not None and timestamp < row["last"]):
            raise ClientError("stream_regressing_timestamp")
        if timestamp >= self.deadline:
            return False
        if row["status"] != "unfinished":
            raise ClientError("stream_after_terminal")
        delta = count - row["count"]
        if delta:
            self.output_tokens += delta
            self.decode_tokens += max(count - 1, 0) - max(row["count"] - 1, 0)
            self.positive_chunks += 1
            self.multi_chunks += delta > 1
            if row["last"] is None:
                self.ttft.add((timestamp - row["start"]) * 1000)
            else:
                interval = (timestamp - row["last"]) * 1000
                self.chunk_intervals.add(interval)
                # No division of a multi-token chunk interval into invented
                # per-token latencies. Only adjacent single-token events count.
                if delta == 1 and row["last_delta"] == 1:
                    self.single_itl.add(interval)
            row["last"], row["last_delta"] = timestamp, delta
        row["count"] = count
        if terminal:
            row["status"] = "completed"
        return terminal

    def fail(self, index, code):
        self.states[index]["status"] = "failed"
        self.errors.add(code)

    def summary(self):
        counts = Counter(row["status"] for row in self.states.values())
        seconds = self.deadline - self.started
        return {"started_monotonic": self.started, "deadline_monotonic": self.deadline,
                "window_seconds": seconds, "requests_started": len(self.states),
                "requests_completed": counts["completed"], "requests_failed": counts["failed"],
                "requests_unfinished": counts["unfinished"], "peak_client_inflight": self.peak_active,
                "output_tokens_in_window": self.output_tokens, "decode_tokens_in_window": self.decode_tokens,
                "output_tokens_per_second": self.output_tokens / seconds,
                "decode_tokens_per_second": self.decode_tokens / seconds,
                "positive_chunks": self.positive_chunks, "multi_token_chunks": self.multi_chunks,
                "all_positive_chunks_single_token": self.multi_chunks == 0,
                "ttft_ms": self.ttft.summary(), "chunk_interarrival_ms": self.chunk_intervals.summary(),
                "single_token_itl_ms": self.single_itl.summary(),
                "timing_boundary": "client_monotonic_complete_sse_event_receipt",
                "window_boundary": "half_open_start_inclusive_deadline_exclusive",
                "numerator_scope": "cumulative_completion_token_deltas_including_unfinished_requests",
                "decode_exclusion": "one_first_output_token_per_request_with_output_in_window",
                "ttft_scope": "dispatch_to_first_positive_chunk_received_in_window; includes_unfinished",
                "chunk_interval_scope": "between_positive_sse_events_received_in_window; includes_unfinished",
                "single_token_itl_scope": "only_adjacent_positive_events_both_with_delta_one; client_receipt_proxy",
                "multi_token_interval_policy": "chunk_intervals_only; never_impute_token_ITL",
                "error_codes": sorted(self.errors)}


def _code(error):
    return str(error) if isinstance(error, ClientError) else "client_transport_error"


async def _abort_owned(session, base_url, rids):
    async def abort(rid):
        try:
            async with session.post(base_url + "/abort_request", json={"rid": rid},
                                    allow_redirects=False, timeout=2.0) as response:
                return response.status != 200
        except Exception:
            return True

    tasks = [asyncio.create_task(abort(rid)) for rid in sorted(rids)]
    if not tasks:
        return 0
    try:
        return sum(await asyncio.wait_for(asyncio.gather(*tasks), timeout=3.0))
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)


async def _json_generate(session, base_url, payload):
    async with session.post(base_url + "/generate", json=payload, allow_redirects=False) as response:
        if response.status != 200:
            raise ClientError("generate_http_error")
        try:
            value = await response.json()
        except (ValueError, UnicodeError):
            raise ClientError("generate_invalid_json") from None
        _validate_response_identity(value, payload["rid"])
        _count_and_terminal(value, payload["sampling_params"]["max_new_tokens"])
        return value


async def _warmup(session, base_url, workload, wire_prefix):
    rid = f"{wire_prefix}-warmup"
    try:
        value = await asyncio.wait_for(_json_generate(session, base_url, _payload(
            workload.prompts[0], rid, tokens=ACCEPTANCE_TOKENS, acceptance=True)), timeout=900.0)
        count, terminal = _count_and_terminal(value, ACCEPTANCE_TOKENS)
        if count != ACCEPTANCE_TOKENS or not terminal:
            raise ClientError("warmup_incomplete")
        return {"requests_completed": 1, "output_tokens": ACCEPTANCE_TOKENS,
                "concurrency": 1, "included_in_measurement": False}
    except BaseException:
        await _abort_owned(session, base_url, {rid})
        raise


def extract_acceptance(value, index):
    count, terminal = _count_and_terminal(value, ACCEPTANCE_TOKENS)
    if count != ACCEPTANCE_TOKENS or not terminal:
        raise ClientError("acceptance_incomplete")
    selected = value["meta_info"].get("output_token_logprobs")
    selected_ids = None
    logprobs = None
    if selected is not None:
        if (not isinstance(selected, list) or len(selected) != count
                or any(not isinstance(row, (list, tuple)) or len(row) < 2
                       or type(row[1]) is not int or row[1] < 0 for row in selected)):
            raise ClientError("acceptance_invalid_selected_token_ids")
        selected_ids = [row[1] for row in selected]
        # Numeric logprobs are diagnostic only. Invalid/missing numeric values
        # become null; exact selected token identities remain the acceptance gate.
        logprobs = [row[0] if type(row[0]) in (int, float) and math.isfinite(row[0]) else None
                    for row in selected]
    ids = value.get("output_ids", selected_ids)
    if (not isinstance(ids, list) or len(ids) != count
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ClientError("acceptance_missing_generated_token_ids")
    if selected_ids is not None and selected_ids != ids:
        raise ClientError("acceptance_token_id_sources_disagree")
    return {"request_index": index, "request_id": f"weave-c32-acceptance-{index:04d}",
            "output_ids": ids, "selected_token_logprobs": logprobs,
            "output_tokens": count, "output_ids_sha256": _digest(ids)}


async def run_acceptance(session, base_url, workload, output_dir, *, concurrency=CONCURRENCY):
    base_url = validate_base_url(base_url)
    if concurrency != CONCURRENCY or len(workload.prompts) < CONCURRENCY:
        raise ClientError("acceptance_requires_32_requests")
    output_dir = Path(output_dir)
    summary = {"schema_version": SCHEMA_VERSION, **workload.identity, "status": "running",
               "mode": "acceptance", "concurrency": CONCURRENCY,
               "generation_settings": generation_settings(ACCEPTANCE_TOKENS),
               "acceptance_order_sha256": _digest(list(range(CONCURRENCY))),
               "logprob_policy": "record_only", "raw_content_exported": False,
               "requests_completed": 0, "generated_tokens": 0, "error_codes": []}
    owned, records, tasks = set(), [], []
    # SGLang may issue a delayed abort after a connection closes. Never reuse a
    # wire ID in a later run; stable acceptance IDs below are logical identities.
    wire_prefix = f"weave-c32-{uuid.uuid4().hex}"
    cancelled = None
    try:
        summary["warmup"] = await _warmup(session, base_url, workload, wire_prefix)
        async def one(index):
            rid = f"{wire_prefix}-acceptance-{index:04d}"
            owned.add(rid)
            value = await _json_generate(session, base_url, _payload(
                workload.prompts[index], rid, tokens=ACCEPTANCE_TOKENS, acceptance=True))
            record = extract_acceptance(value, index)
            records.append(record)
            owned.remove(rid)
        tasks = [asyncio.create_task(one(index)) for index in range(CONCURRENCY)]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=900.0)
        summary["status"] = "passed"
    except asyncio.CancelledError as error:
        cancelled = error
        summary["status"], summary["error_codes"] = "failed", ["client_cancelled"]
    except Exception as error:
        summary["status"], summary["error_codes"] = "failed", [_code(error)]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        failures = await _abort_owned(session, base_url, owned)
        summary["abort_requests"], summary["abort_failures"] = len(owned), failures
        if failures:
            summary["status"] = "failed"
            summary["error_codes"].append("abort_failed")
        records.sort(key=lambda row: row["request_index"])
        summary["requests_completed"] = len(records)
        summary["generated_tokens"] = sum(row["output_tokens"] for row in records)
        summary["generated_ids_sha256"] = _digest([[row["request_index"], row["output_ids"]] for row in records])
        private = {"schema_version": SCHEMA_VERSION, **workload.identity,
                   "generation_settings": generation_settings(ACCEPTANCE_TOKENS), "records": records}
        atomic_json(output_dir / "acceptance-private.json", private)
        summary["private_artifact_sha256"] = hashlib.sha256((output_dir / "acceptance-private.json").read_bytes()).hexdigest()
        atomic_json(output_dir / "acceptance-summary.json", summary)
    if cancelled:
        raise cancelled
    return summary


async def _stream_request(session, base_url, payload, index, meter, stall=None):
    parser = SSEParser()
    async with session.post(base_url + "/generate", json=payload, allow_redirects=False) as response:
        if response.status != 200:
            raise ClientError("generate_http_error")
        async for block in response.content.iter_any():
            # One timestamp for the bytes received by this read, even when the
            # transport batches several SSE events. Never use server timestamps.
            received = time.monotonic()
            if received >= meter.deadline:
                return False
            for value in parser.feed(block):
                if value is None:
                    raise ClientError("stream_done_without_terminal")
                _validate_response_identity(value, payload["rid"])
                if stall is not None:
                    stall.observe(index, value, received)
                if meter.observe(index, value, received):
                    return True
        parser.finish()
        raise ClientError("stream_eof_without_terminal")


async def run_measurement(session, base_url, workload, output_dir, *, seconds=600.0,
                          concurrency=CONCURRENCY, output_tokens=OUTPUT_TOKENS, stall_port_base=None):
    base_url = validate_base_url(base_url)
    if concurrency != CONCURRENCY or output_tokens != OUTPUT_TOKENS:
        raise ClientError("measurement_requires_C32_output1024")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ClientError("invalid_measurement_seconds")
    output_dir = Path(output_dir)
    summary = {"schema_version": SCHEMA_VERSION, **workload.identity,
               "status": "running", "mode": "measurement", "concurrency": CONCURRENCY,
               "generation_settings": generation_settings(OUTPUT_TOKENS),
               "requested_seconds": seconds, "raw_content_exported": False,
               "error_codes": []}
    owned, tasks = set(), []
    wire_prefix = f"weave-c32-{uuid.uuid4().hex}"
    meter = progress = None
    cancelled = None
    reached_deadline = False
    from .stall_client import StallClient, arm_ranks
    stall = StallClient() if stall_port_base is not None else None
    if stall is not None:
        summary.update(instrumented_only=True, instrumented_diagnostic=True, used_for_formal_performance=False, numerical_policy="record_only")
    try:
        summary["warmup"] = await _warmup(session, base_url, workload, wire_prefix)
        if stall is not None:
            try:
                await arm_ranks(session, stall_port_base)
            except ValueError as error:
                raise ClientError(str(error)) from None
        start = time.monotonic()
        meter = WindowMetrics(start, start + seconds, output_tokens=output_tokens)
        atomic_json(output_dir / "load-window.json", {
            "schema_version": SCHEMA_VERSION, "status": "load_started",
            "started_monotonic": start, "deadline_monotonic": meter.deadline,
            "dataset_sha256": workload.identity["dataset_sha256"],
            "request_order_sha256": workload.identity["request_order_sha256"]})
        next_index = 0

        async def worker():
            nonlocal next_index
            while True:
                dispatched = time.monotonic()
                if dispatched >= meter.deadline:
                    return
                # No await between reading and advancing the shared cursor:
                # asyncio tasks consume exactly one common deterministic queue.
                if next_index >= len(workload.prompts):
                    raise ClientError("load_exhausted_before_deadline")
                index = next_index
                next_index += 1
                rid = f"{wire_prefix}-measure-{index:04d}"
                meter.start(index, dispatched)
                if stall is not None:
                    stall.start(index, dispatched)
                owned.add(rid)
                try:
                    args = (session, base_url, _payload(workload.prompts[index], rid), index, meter)
                    complete = await _stream_request(*args, stall=stall) if stall is not None else await _stream_request(*args)
                    if complete:
                        owned.remove(rid)
                    else:
                        return
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    meter.fail(index, _code(error))
                    raise

        async def publish_progress():
            while True:
                # Progress writes occur at most once per second, independent
                # of token frequency. The final summary is a separate artifact.
                await asyncio.sleep(1.0)
                if time.monotonic() >= meter.deadline:
                    return
                value = meter.summary()
                value.update(schema_version=SCHEMA_VERSION, status="running")
                atomic_json(output_dir / "load-progress.json", value)

        tasks = [asyncio.create_task(worker()) for _ in range(CONCURRENCY)]
        progress = asyncio.create_task(publish_progress())
        done, pending = await asyncio.wait(tasks + [progress], timeout=max(0.0, meter.deadline - time.monotonic()),
                                           return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            error = task.exception()
            if error is not None:
                raise error
        reached_deadline = time.monotonic() >= meter.deadline
        if not reached_deadline:
            raise ClientError("load_ended_before_deadline")
        if meter.peak_active != CONCURRENCY:
            raise ClientError("client_never_reached_C32")
        if not meter.decode_tokens:
            raise ClientError("no_decode_tokens_received")
        summary["status"] = "passed"
    except asyncio.CancelledError as error:
        cancelled = error
        summary["status"], summary["error_codes"] = "failed", ["client_cancelled"]
    except Exception as error:
        summary["status"], summary["error_codes"] = "failed", [_code(error)]
    finally:
        for task in tasks + ([progress] if progress is not None else []):
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, *([progress] if progress is not None else []), return_exceptions=True)
        aborts, abort_failures = len(owned), await _abort_owned(session, base_url, owned)
        summary.update(abort_requests=aborts, abort_failures=abort_failures,
                       full_window_observed=reached_deadline)
        if abort_failures:
            summary["status"] = "failed"
            summary["error_codes"].append("abort_failed")
        if meter is not None:
            codes = summary["error_codes"]
            summary.update(meter.summary())
            summary["error_codes"] = sorted(set(codes) | meter.errors)
            summary["elapsed_until_cleanup_finished_seconds"] = time.monotonic() - meter.started
            atomic_json(output_dir / "load-window.json", {
                "schema_version": SCHEMA_VERSION, "status": "load_finished",
                "started_monotonic": meter.started, "deadline_monotonic": meter.deadline,
                "finished_monotonic": time.monotonic(), "client_status": summary["status"],
                "dataset_sha256": workload.identity["dataset_sha256"],
                "request_order_sha256": workload.identity["request_order_sha256"]})
        if summary["status"] != "passed":
            summary["decode_tokens_per_second"] = None
            summary["output_tokens_per_second"] = None
        if stall is not None:
            atomic_json(output_dir / "client-stall.json", stall.report(summary["status"], summary["error_codes"]))
        atomic_json(output_dir / "client-summary.json", summary)
    if cancelled:
        raise cancelled
    return summary
