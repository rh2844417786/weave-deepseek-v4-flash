import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    from src.weave.native_p0_trace import (
        LaunchTrace, NativeP0TraceError, parse_chrome_trace, parse_marlin_template,
    )
except ImportError:
    LaunchTrace = NativeP0TraceError = parse_chrome_trace = parse_marlin_template = None


MARLIN = (
    "void marlin::Marlin<cutlass::detail::scalar<nv_bfloat16, 2>, "
    "(host::ScalarTypeId)5, (host::ScalarTypeId)0, "
    "256, 1, 16, 4, true, 4, -1, false, true, false>"
    "(int4 const*, int4 const*, int4*, int*)"
)


def event(name=MARLIN, ts=100):
    return {
        "ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 7,
        "ts": ts, "dur": 12.5,
        "args": {
            "device": 0, "context": 1, "stream": 7, "correlation": 19,
            "grid": [132, 1, 1], "block": [256, 1, 1],
            "registers per thread": 128, "shared memory": 98304,
            "address": "0xdeadbeef", "Input Dims": [[32, 2048]],
        },
    }


class NativeP0TraceTest(unittest.TestCase):
    def test_balanced_template_and_m8_tile(self):
        self.assertIsNotNone(parse_marlin_template, "trace parser is not implemented")
        parsed = parse_marlin_template(MARLIN)
        self.assertEqual(parsed["threads"], 256)
        self.assertEqual(parsed["tile_m"], 8)
        self.assertEqual(parsed["tile_n"], 256)
        self.assertEqual(parsed["tile_k"], 64)
        self.assertEqual(parsed["stages"], 4)
        self.assertEqual(parsed["group_blocks"], -1)
        self.assertEqual(parsed["scalar_type"], "cutlass::detail::scalar<nv_bfloat16, 2>")
        self.assertEqual(parse_marlin_template(MARLIN.replace("4, true, 4", "4, false, 4"))["tile_m"], 16)

    def test_native_reorders_kernels_and_maps_exact_43_layers(self):
        self.assertIsNotNone(parse_chrome_trace, "trace parser is not implemented")
        events = [event(ts=1000 + i * 20) for i in reversed(range(86))]
        cpu = event(ts=0)
        cpu["cat"] = "cpu_op"
        memcpy = event("Memcpy HtoD", ts=1)
        memcpy["cat"] = "gpu_memcpy"
        result = parse_chrome_trace({"traceEvents": [cpu, memcpy] + events})
        self.assertEqual(result["kernel_count"], 86)
        self.assertEqual(result["marlin_kernel_count"], 86)
        first, last = result["kernels"][0], result["kernels"][-1]
        self.assertEqual((first["layer"], first["role"], first["start_us"]), (0, "gate_up", 1000))
        self.assertEqual((last["layer"], last["role"]), (42, "down"))
        self.assertEqual(first["grid"], [132, 1, 1])
        self.assertEqual(first["shared_memory_bytes"], 98304)
        self.assertNotIn("dynamic_shared_memory_bytes", first)
        public = json.dumps(result)
        for private in ("0xdeadbeef", "Input Dims", "cpu_op", "correlation"):
            self.assertNotIn(private, public)

    def test_generic_mode_retains_actual_resources_without_marlin_requirement(self):
        self.assertIsNotNone(parse_chrome_trace, "trace parser is not implemented")
        result = parse_chrome_trace([event("void vectorized_elementwise_kernel<float>()")], expected_layers=None)
        row = result["kernels"][0]
        self.assertEqual(row["threads_per_block"], 256)
        self.assertEqual(row["registers_per_thread"], 128)
        self.assertNotIn("layer", row)

    def test_rejects_partial_replay_and_wrong_block_size(self):
        self.assertIsNotNone(parse_chrome_trace, "trace parser is not implemented")
        with self.assertRaisesRegex(NativeP0TraceError, "86"):
            parse_chrome_trace([event()])
        wrong = event(ts=200)
        wrong["args"]["block"] = [128, 1, 1]
        with self.assertRaisesRegex(NativeP0TraceError, "threads"):
            parse_chrome_trace([event(), wrong], expected_layers=1)

    def test_missing_or_invalid_launch_fields_fail_in_generic_smoke(self):
        self.assertIsNotNone(parse_chrome_trace, "trace parser is not implemented")
        for key in ("grid", "block", "registers per thread", "shared memory"):
            with self.subTest(key=key):
                broken = event("elementwise")
                del broken["args"][key]
                with self.assertRaisesRegex(NativeP0TraceError, "CUPTI"):
                    parse_chrome_trace([broken], expected_layers=None)
        for bad in ([0, 1, 1], [256, 1], [256.5, 1, 1], "256,1,1"):
            broken = event()
            broken["args"]["block"] = bad
            with self.assertRaises(NativeP0TraceError):
                parse_chrome_trace([broken], expected_layers=None)

    def test_unknown_templates_and_cpu_only_trace_cannot_masquerade_as_launches(self):
        self.assertIsNotNone(parse_chrome_trace, "trace parser is not implemented")
        for name in ("void Marlin<half, 1, 0, BAD>()", MARLIN.replace(", true, 4", ", maybe, 4")):
            with self.assertRaisesRegex(NativeP0TraceError, "template"):
                parse_chrome_trace([event(name), event(name, ts=200)], expected_layers=1)
        cpu = event()
        cpu["cat"] = "cuda_runtime"
        with self.assertRaisesRegex(NativeP0TraceError, "CUPTI"):
            parse_chrome_trace([cpu], expected_layers=None)

    def test_explicit_shared_memory_split_is_preserved_without_guessing(self):
        self.assertIsNotNone(parse_chrome_trace, "trace parser is not implemented")
        row = event("elementwise")
        row["args"].update({"static shared memory": 1024, "dynamic shared memory": 97280})
        parsed = parse_chrome_trace([row], expected_layers=None)["kernels"][0]
        self.assertEqual(parsed["static_shared_memory_bytes"], 1024)
        self.assertEqual(parsed["dynamic_shared_memory_bytes"], 97280)
        del row["args"]["shared memory"]
        parsed = parse_chrome_trace([row], expected_layers=None)["kernels"][0]
        self.assertIsNone(parsed["shared_memory_bytes"])

    def test_nonfinite_timing_and_ambiguous_native_order_fail(self):
        for key in ("ts", "dur"):
            row = event()
            row[key] = float("nan")
            with self.assertRaises(NativeP0TraceError):
                parse_chrome_trace([row], expected_layers=None)
        with self.assertRaisesRegex(NativeP0TraceError, "ambiguous"):
            parse_chrome_trace([event(), event()], expected_layers=1)

    def test_partial_start_failure_attempts_profiler_cleanup(self):
        calls = []
        def fail_start():
            raise RuntimeError("partial profiler startup")
        profile = SimpleNamespace(start=fail_start, stop=lambda: calls.append("stop"))
        torch = SimpleNamespace(profiler=SimpleNamespace(
            profile=lambda **kw: profile, ProfilerActivity=SimpleNamespace(CPU=1, CUDA=2)))
        with tempfile.TemporaryDirectory() as temp:
            trace = LaunchTrace(torch, 0, temp)
            with self.assertRaisesRegex(NativeP0TraceError, "could not start"):
                trace.begin()
            trace.abort()
        self.assertEqual(calls, ["stop"])

    def test_capture_lifecycle_exports_private_trace_and_public_whitelist(self):
        self.assertIsNotNone(LaunchTrace, "trace collector is not implemented")
        # Torch is an external GPU dependency; fake only its profiler boundary.
        calls = []
        class Profile:
            def start(self):
                calls.append("start")
            def stop(self):
                calls.append("stop")
            def export_chrome_trace(self, path):
                calls.append("export")
                Path(path).write_text(json.dumps({"traceEvents": [event("elementwise")]}))
        def make_profile(**kwargs):
            calls.append(kwargs)
            return Profile()
        torch = SimpleNamespace(profiler=SimpleNamespace(
            profile=make_profile,
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
        ))
        with tempfile.TemporaryDirectory() as temp:
            private = Path(temp) / "private"
            trace = LaunchTrace(torch, 2, private, expected_layers=None)
            trace.begin()
            result = trace.finish()
            self.assertEqual(result["rank"], 2)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(calls[1:], ["start", "stop", "export"])
            self.assertEqual(calls[0]["activities"], ["cpu", "cuda"])
            for option in ("record_shapes", "profile_memory", "with_stack"):
                self.assertFalse(calls[0][option])
            files = list(private.glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].stat().st_mode & 0o777, 0o600)
            self.assertNotIn(temp, json.dumps(result))
            with self.assertRaises(NativeP0TraceError):
                trace.begin()

    def test_abort_stops_active_profiler_without_export(self):
        self.assertIsNotNone(LaunchTrace, "trace collector is not implemented")
        calls = []
        profile = SimpleNamespace(start=lambda: calls.append("start"), stop=lambda: calls.append("stop"))
        torch = SimpleNamespace(profiler=SimpleNamespace(
            profile=lambda **kw: profile, ProfilerActivity=SimpleNamespace(CPU=1, CUDA=2)))
        with tempfile.TemporaryDirectory() as temp:
            trace = LaunchTrace(torch, 0, temp)
            trace.begin()
            trace.abort()
            trace.abort()
        self.assertEqual(calls, ["start", "stop"])


if __name__ == "__main__":
    unittest.main()
