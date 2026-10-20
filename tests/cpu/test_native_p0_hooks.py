"""Native hook semantics with fake tensors; no CUDA execution claims."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.weave import native_p0_hooks as hooks

ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    dtype = "bfloat16"
    def __init__(self, shape=(32, 8)):
        self.shape = shape
    def item(self):
        raise AssertionError("device scalar read in capture")
    def cpu(self):
        raise AssertionError("device copy in capture")


class Profiler:
    def __init__(self):
        self.metadata, self.calls, self.grouping = {}, [], []
    def register_layer(self, layer, metadata):
        self.metadata[layer] = metadata
    def set_metadata(self, layer, metadata):
        self.metadata[layer].update(metadata)
    @contextmanager
    def span(self, layer, name):
        self.calls.append((layer, name, "start"))
        yield
        self.calls.append((layer, name, "end"))
    def bind_grouping(self, *args):
        self.grouping.append(args)


def fixture(path="forward_normal_dual_stream"):
    originals = []
    def kernel(*args, **kwargs):
        originals.append((args, kwargs))
        return args[0] if args else None
    modules = {
        "marlin": SimpleNamespace(**{name: kernel for name in
            ("moe_wna16_marlin_gemm", "moe_sum_reduce", "silu_and_mul", "situ_and_mul",
             "swiglu_limit_func", "swiglu_gpt_oss_sigmoid_alpha_contiguous")}),
        "grouping": SimpleNamespace(moe_align_block_size=kernel),
        "sum": SimpleNamespace(moe_topk_sum=kernel),
        "model": SimpleNamespace(maybe_fuse_routed_scale_and_shared_add=kernel,
                                 tensor_model_parallel_all_reduce=kernel),
    }
    class Module:
        def __init__(self, forward=kernel):
            self.forward = forward
    class MLP:
        _shared_expert_tp1 = True
        def __init__(self):
            self.gate, self.topk = Module(), Module()
            self.experts = Module(self.routed)
            self.shared_experts = Module()
        def forward(self, hidden_states):
            return getattr(self, path)(hidden_states)
        def forward_normal(self, hidden):
            self.gate.forward(hidden)
            self.topk.forward(hidden)
            result = self.experts.forward(hidden)
            self._forward_shared_experts(hidden)
            modules["model"].maybe_fuse_routed_scale_and_shared_add(self.experts, result, None, 1.0)
            return result
        forward_normal_dual_stream = forward_normal
        forward_deepep = forward_normal
        def _forward_shared_experts(self, hidden):
            return hidden
        def routed(self, hidden):
            modules["grouping"].moe_align_block_size(hidden, 8, 256)
            args = [hidden, Tensor(), Tensor(), None, Tensor(), None, None, None,
                    None, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()]
            for index in range(2):
                modules["marlin"].moe_wna16_marlin_gemm(
                    *args, size_m=32 if index == 0 else 256, size_n=16, size_k=8,
                    moe_block_size=8, top_k=8 if index == 0 else 1,
                    mul_topk_weights=bool(index), is_ep=True)
                if index == 0:
                    modules["marlin"].silu_and_mul(hidden, Tensor())
            modules["sum"].moe_topk_sum(hidden, Tensor())
            return hidden
    mlp = MLP()
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]))
    cuda = SimpleNamespace(capturing=False)
    cuda.is_current_stream_capturing = lambda: cuda.capturing
    return model, SimpleNamespace(cuda=cuda), modules, originals


class HookTests(unittest.TestCase):
    def test_only_b32_capture_and_original_returns_and_arguments(self):
        model, torch, modules, originals = fixture()
        profiler = Profiler()
        hooks._install_hooks(model, torch, profiler, modules, layers=1)
        hidden = Tensor()
        self.assertIs(model.model.layers[0].mlp.forward(hidden), hidden)
        torch.cuda.capturing = True
        model.model.layers[0].mlp.forward(Tensor((31, 8)))
        self.assertEqual(profiler.calls, [])
        self.assertIs(model.model.layers[0].mlp.forward(hidden), hidden)
        spans = {entry[1] for entry in profiler.calls}
        self.assertEqual(spans, {"moe", "routed", "shared", "router", "topk", "grouping",
                                 "gate_up", "down", "activation", "routed_finalize", "shared_merge"})
        metadata = profiler.metadata[0]
        self.assertEqual(metadata["captured_path"], "forward_normal_dual_stream")
        self.assertEqual([entry["role"] for entry in metadata["marlin_calls"]], ["gate_up", "down"])
        self.assertEqual(profiler.grouping[0][-2:], (8, 256))
        self.assertFalse(metadata["shared_merge_includes_shared_add"])
        self.assertTrue(metadata["shared_tp1"])
        self.assertIsNone(hooks.CURRENT.get())
        count = len(profiler.calls)
        modules["sum"].moe_topk_sum(hidden)
        self.assertEqual(len(profiler.calls), count)
        self.assertIs(originals[-1][0][0], hidden)

    def test_actual_path_recorded_and_missing_optional_callable_explicit(self):
        model, torch, modules, _ = fixture(path="forward_deepep")
        model.model.layers[0].mlp._forward_shared_experts = None
        # A native alternative path can omit shared compute altogether.
        model.model.layers[0].mlp.forward_deepep = lambda hidden: hidden
        profiler = Profiler()
        hooks._install_hooks(model, torch, profiler, modules, layers=1)
        torch.cuda.capturing = True
        model.model.layers[0].mlp.forward(Tensor())
        self.assertEqual(profiler.metadata[0]["captured_path"], "forward_deepep")
        self.assertFalse(profiler.metadata[0]["callable_availability"]["shared"])
        self.assertEqual(profiler.metadata[0]["marlin_calls"], [])

    def test_exception_resets_context_and_propagates_original_failure(self):
        model, torch, modules, _ = fixture()
        def fail(*args, **kwargs):
            raise LookupError("native failure")
        model.model.layers[0].mlp.forward_normal_dual_stream = fail
        profiler = Profiler()
        hooks._install_hooks(model, torch, profiler, modules, layers=1)
        torch.cuda.capturing = True
        with self.assertRaisesRegex(LookupError, "native failure"):
            model.model.layers[0].mlp.forward(Tensor())
        self.assertIsNone(hooks.CURRENT.get())

    def test_invalid_installation_changes_no_aliases(self):
        model, torch, modules, _ = fixture()
        original = modules["marlin"].moe_wna16_marlin_gemm
        model.model.layers[0].mlp.forward = None
        with self.assertRaisesRegex(RuntimeError, "missing native callable"):
            hooks._install_hooks(model, torch, Profiler(), modules, layers=1)
        self.assertIs(modules["marlin"].moe_wna16_marlin_gemm, original)
        self.assertFalse(hasattr(model, "weave_native_p0_profiler"))

    def test_source_identity_before_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "pkg").mkdir()
            (root / "pkg/native.py").write_bytes(b"pinned\n")
            pin = hashlib.sha256(b"pinned\n").hexdigest()
            (root / "configs/native_p0_sources.json").write_text(json.dumps(
                {"commit": "abc", "sha256": {"native.py": pin}}))
            (root / "configs/runtime.json").write_text('{"sglang_commit":"abc"}')
            with patch.object(hooks, "ROOT", root):
                self.assertEqual(hooks.verify_native_p0_sources(root / "pkg"), {"native.py": pin})
                (root / "pkg/native.py").write_bytes(b"changed\n")
                with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                    hooks.verify_native_p0_sources(root / "pkg")

    def test_environment_opt_in_conflicts_and_defaults(self):
        self.assertIsNone(hooks.native_p0_options({}))
        env = dict(WEAVE_NATIVE_P0="1", WEAVE_GRAPH_C32="1", WEAVE_REQUEST_TIMING="1",
                   WEAVE_NATIVE_P0_PRIVATE_DIR="/work/run/native-p0-private")
        self.assertEqual(hooks.native_p0_options(env), dict(stride=128, capacity=64,
                         private_dir="/work/run/native-p0-private"))
        for key, value in (("WEAVE_GRAPH_STAGE_PROFILE", "1"),
                           ("SGLANG_EXTERNAL_MODEL_PACKAGE", "custom"), ("WEAVE_GRAPH_C32", "0")):
            with self.assertRaises(ValueError):
                hooks.native_p0_options(dict(env, **{key: value}))
        with self.assertRaises(ValueError):
            hooks.native_p0_options({"WEAVE_NATIVE_P0_STRIDE": "5"})

    def launch(self, *args):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3",
                   WEAVE_MAX_RUNNING_REQUESTS="32", WEAVE_CUDA_GRAPH_MAX_BS_DECODE="32")
        return subprocess.run([sys.executable, str(ROOT / "scripts/run_native_ep.py"), *args],
                              env=env, capture_output=True, text=True)

    def test_launcher_default_does_not_enable_p0(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any("WEAVE_NATIVE_P0" in value for value in json.loads(result.stdout)["docker_argv"]))

    def test_launcher_opt_in_and_conflicts(self):
        args = ("--native-p0", "--graph-c32", "--decode-graph", "full", "--request-timing")
        result = self.launch(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads(result.stdout)["docker_argv"]
        for value in ("WEAVE_NATIVE_P0=1", "WEAVE_NATIVE_P0_STRIDE=128",
                      "WEAVE_NATIVE_P0_CAPACITY=64",
                      "WEAVE_NATIVE_P0_PRIVATE_DIR=/work/preview/native-p0-private"):
            self.assertIn(value, argv)
        for extra in (("--graph-stage-profile",), ("--backend", "weave_functional"),
                      ("--execution-profile",)):
            rejected = self.launch(*args, *extra)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("native P0 requires", rejected.stderr)
        self.assertNotEqual(self.launch("--native-p0-stride", "2").returncode, 0)


if __name__ == "__main__":
    unittest.main()
