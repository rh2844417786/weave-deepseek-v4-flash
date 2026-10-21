"""Opt-in observation of the pinned native Marlin path, without replacing work.

Tensor inspection here is restricted to host shape/dtype metadata. GPU grouping
buffers are retained for the sampler to read after the sampled graph completes.
"""
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ContextVar("weave_native_p0_layer", default=None)
MODULES = {
    "marlin": "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe",
    "grouping": "sglang.srt.layers.moe.fused_moe_triton",
    "sum": "sglang.kernels.ops.moe.moe_topk_sum",
    "model": "sglang.srt.models.deepseek_v2",
}


def native_p0_options(environ=None):
    env = os.environ if environ is None else environ
    enabled = env.get("WEAVE_NATIVE_P0", "0")
    if enabled not in ("0", "1"):
        raise ValueError("WEAVE_NATIVE_P0 must be 0 or 1")
    if enabled == "0":
        if any(key in env for key in ("WEAVE_NATIVE_P0_STRIDE", "WEAVE_NATIVE_P0_CAPACITY",
                                      "WEAVE_NATIVE_P0_PRIVATE_DIR", "WEAVE_NATIVE_P0_MODE",
                                      "WEAVE_NATIVE_P0_READBACK")):
            raise ValueError("native P0 settings require WEAVE_NATIVE_P0=1")
        return None
    if (env.get("WEAVE_GRAPH_C32") != "1" or env.get("WEAVE_REQUEST_TIMING") != "1"
            or env.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
            or any(env.get(key, "0") != "0" for key in
                   ("WEAVE_GRAPH_STAGE_PROFILE", "WEAVE_EXECUTION_PROFILE",
                    "WEAVE_FORWARD_PROFILE", "WEAVE_STALL_DIAGNOSTIC"))
            or env.get("WEAVE_PROFILE") or env.get("WEAVE_REPLAY_CAPTURE_DIR")):
        raise ValueError("native P0 requires native C32 request timing and no other profiles")
    stride = int(env.get("WEAVE_NATIVE_P0_STRIDE", "128"))
    capacity = int(env.get("WEAVE_NATIVE_P0_CAPACITY", "64"))
    if not 1 <= stride <= 65536 or not 1 <= capacity <= 256:
        raise ValueError("native P0 stride must be in 1..65536 and capacity in 1..256")
    mode = env.get("WEAVE_NATIVE_P0_MODE", "full")
    readback = env.get("WEAVE_NATIVE_P0_READBACK", "1")
    if mode not in ("full", "events", "buffers"):
        raise ValueError("native P0 mode must be full, events, or buffers")
    if readback not in ("0", "1"):
        raise ValueError("WEAVE_NATIVE_P0_READBACK must be 0 or 1")
    if mode != "full" and readback == "1":
        raise ValueError("native P0 events/buffers modes require no readback")
    if env.get("WEAVE_FIXED_PREFIX") == "1" and readback == "1":
        raise ValueError("fixed-prefix native P0 requires readback disabled")
    directory = env.get("WEAVE_NATIVE_P0_PRIVATE_DIR", "")
    if not directory or not Path(directory).is_absolute() or ".." in Path(directory).parts:
        raise ValueError("native P0 requires an absolute private trace directory")
    options = dict(stride=stride, capacity=capacity, private_dir=directory)
    if "WEAVE_NATIVE_P0_MODE" in env or "WEAVE_NATIVE_P0_READBACK" in env:
        options.update(mode=mode, readback=readback == "1")
    return options


def verify_native_p0_sources(package_root):
    manifest = json.loads((ROOT / "configs/native_p0_sources.json").read_text())
    contract = json.loads((ROOT / "configs/runtime.json").read_text())
    if manifest["commit"] != contract["sglang_commit"]:
        raise RuntimeError("native P0 source commit differs from runtime contract")
    for name, expected in manifest["sha256"].items():
        path = Path(package_root) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("native P0 source mismatch: " + name)
    return manifest["sha256"]


def tensor_metadata(value):
    if value is None or not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    return {"shape": [int(size) for size in value.shape], "dtype": str(value.dtype)}


def module_metadata(module):
    if module is None:
        return None
    result = {"class": type(module).__module__ + "." + type(module).__name__}
    for name in ("weight", "qweight", "weight_packed", "weight_scale",
                 "w13_weight", "w2_weight", "w13_qweight", "w2_qweight",
                 "w13_scales", "w2_scales"):
        metadata = tensor_metadata(getattr(module, name, None))
        if metadata is not None:
            result[name] = metadata
    return result


def _install_hooks(model, torch, profiler, modules, *, layers=43):
    """Prepare all targets before mutating aliases; caller has verified sources."""
    blocks = list(model.model.layers)
    if len(blocks) != layers or getattr(model, "weave_native_p0_profiler", None) is not None:
        raise ValueError("native P0 requires exactly one installation on all model layers")
    patches = []

    def queue(owner, name, wrapper, required=True):
        original = getattr(owner, name, None)
        if not callable(original):
            if required:
                raise RuntimeError("native P0 missing native callable: " + name)
            return False
        if getattr(original, "_weave_native_p0", False):
            raise RuntimeError("native P0 alias already instrumented: " + name)
        replacement = wrapper(original)
        replacement._weave_native_p0 = True
        patches.append((owner, name, replacement))
        return True

    def observed(name, extra=None):
        def decorate(original):
            @wraps(original)
            def wrapped(*args, **kwargs):
                state = CURRENT.get()
                if state is None or state["profiler"] is not profiler:
                    return original(*args, **kwargs)
                if extra is not None:
                    extra(state, args, kwargs)
                with profiler.span(state["layer"], name):
                    return original(*args, **kwargs)
            return wrapped
        return decorate

    def marlin(original):
        @wraps(original)
        def wrapped(*args, **kwargs):
            state = CURRENT.get()
            if state is None or state["profiler"] is not profiler:
                return original(*args, **kwargs)
            index = state["marlin_calls"]
            if index >= 2:
                raise RuntimeError("native P0 observed more than two routed Marlin calls")
            role = ("gate_up", "down")[index]
            state["marlin_calls"] += 1
            fields = ("size_m", "size_n", "size_k", "moe_block_size", "top_k",
                      "mul_topk_weights", "is_ep", "is_k_full", "use_atomic_add",
                      "use_fp32_reduce", "is_zp_float")
            metadata = {name: kwargs.get(name) for name in fields}
            metadata.update(role=role, input=tensor_metadata(args[0]),
                            weight=tensor_metadata(args[2]), scale=tensor_metadata(args[4]),
                            b_q_type_id=getattr(kwargs.get("b_q_type"), "id", None),
                            bias_present=args[3] is not None,
                            global_scale_present=args[5] is not None,
                            zero_point_present=args[6] is not None)
            state["marlin_metadata"].append(metadata)
            profiler.set_metadata(state["layer"], {"marlin_calls": list(state["marlin_metadata"])})
            if index == 0:
                profiler.bind_grouping(state["layer"], args[10], args[11], args[12],
                                       kwargs["moe_block_size"], kwargs["size_m"] * kwargs["top_k"])
            with profiler.span(state["layer"], role):
                return original(*args, **kwargs)
        return wrapped

    def merge_metadata(state, args, kwargs):
        shared = args[2] if len(args) > 2 else kwargs.get("shared_output")
        profiler.set_metadata(state["layer"], {"shared_merge_includes_shared_add": shared is not None})

    queue(modules["marlin"], "moe_wna16_marlin_gemm", marlin)
    for name in ("silu_and_mul", "situ_and_mul", "swiglu_limit_func",
                 "swiglu_gpt_oss_sigmoid_alpha_contiguous"):
        queue(modules["marlin"], name, observed("activation"))
    queue(modules["marlin"], "moe_sum_reduce", observed("routed_finalize"))
    # The pinned fused function imports these at call time, so these are the
    # package aliases actually resolved by the native implementation.
    queue(modules["grouping"], "moe_align_block_size", observed("grouping"))
    queue(modules["sum"], "moe_topk_sum", observed("routed_finalize"))
    queue(modules["model"], "maybe_fuse_routed_scale_and_shared_add",
          observed("shared_merge", merge_metadata))
    queue(modules["model"], "tensor_model_parallel_all_reduce", observed("all_reduce"))

    registrations = []
    for layer, block in enumerate(blocks):
        mlp = block.mlp
        metadata = {
            "layer": layer, "shared_tp1": bool(getattr(mlp, "_shared_expert_tp1", False)),
            "captured_path": None, "marlin_calls": [],
            "shared_tp1_post_reduce_add_uninstrumented": bool(getattr(mlp, "_shared_expert_tp1", False)),
            "modules": {name: module_metadata(getattr(mlp, name, None))
                        for name in ("experts", "shared_experts", "gate", "topk")},
        }
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None:
            metadata["modules"]["shared_projections"] = {
                name: module_metadata(getattr(shared, name, None))
                for name in ("gate_up_proj", "down_proj")}
        available = {}
        for method in ("forward_normal", "forward_normal_dual_stream", "forward_deepep"):
            def path_decorator(original, method=method):
                @wraps(original)
                def wrapped(*args, **kwargs):
                    state = CURRENT.get()
                    if state is not None and state["profiler"] is profiler:
                        profiler.set_metadata(state["layer"], {"captured_path": method})
                    return original(*args, **kwargs)
                return wrapped
            available[method] = queue(mlp, method, path_decorator, required=False)
        available["shared"] = queue(mlp, "_forward_shared_experts", observed("shared"), required=False)
        for name, span in (("experts", "routed"), ("gate", "router"), ("topk", "topk")):
            available[span] = queue(getattr(mlp, name, None), "forward", observed(span), required=False)
        metadata["callable_availability"] = available
        registrations.append((layer, metadata))

        def mlp_decorator(original, layer=layer):
            @wraps(original)
            def wrapped(*args, **kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if (hidden is None or len(hidden.shape) != 2 or int(hidden.shape[0]) != 32
                        or not torch.cuda.is_current_stream_capturing()):
                    return original(*args, **kwargs)
                state = {"profiler": profiler, "layer": layer, "marlin_calls": 0,
                         "marlin_metadata": []}
                token = CURRENT.set(state)
                try:
                    profiler.set_metadata(layer, {"input": tensor_metadata(hidden)})
                    with profiler.span(layer, "moe"):
                        return original(*args, **kwargs)
                finally:
                    CURRENT.reset(token)
            return wrapped
        queue(mlp, "forward", mlp_decorator)

    for layer, metadata in registrations:
        profiler.register_layer(layer, metadata)
    for owner, name, replacement in patches:
        setattr(owner, name, replacement)
    model.weave_native_p0_profiler = profiler
    return profiler


def install_native_p0(model, torch, rank, *, stride=128, capacity=64, private_dir,
                      package_root=None, mode='full', readback=True):
    import sglang
    from .native_p0 import NativeP0Profiler
    package_root = Path(package_root or Path(sglang.__file__).parent).resolve()
    pins = verify_native_p0_sources(package_root)
    modules = {key: importlib.import_module(name) for key, name in MODULES.items()}
    for module in modules.values():
        actual = Path(module.__file__).resolve()
        relative = actual.relative_to(package_root).as_posix()
        if relative not in pins:
            raise RuntimeError("native P0 loaded an unpinned module: " + str(actual))
    # Verify concrete method implementations as well as import aliases before
    # changing anything; this rejects custom-model substitutions.
    for block in model.model.layers:
        for owner in (block.mlp, getattr(block.mlp, "experts", None),
                      getattr(block.mlp, "gate", None), getattr(block.mlp, "topk", None)):
            method = getattr(owner, "forward", None)
            code = getattr(getattr(method, "__func__", method), "__code__", None)
            if code is None:
                continue
            try:
                relative = Path(code.co_filename).resolve().relative_to(package_root).as_posix()
            except ValueError as exc:
                raise RuntimeError("native P0 callable outside pinned SGLang") from exc
            if relative not in pins:
                raise RuntimeError("native P0 unpinned callable source: " + relative)
    trace = None
    if readback:
        from .native_p0_trace import LaunchTrace
        trace = LaunchTrace(torch, rank, private_dir)
    profiler = NativeP0Profiler(torch, rank, stride=stride, capacity=capacity, layers=43,
                                trace=trace, mode=mode, readback=readback)
    return _install_hooks(model, torch, profiler, modules)
