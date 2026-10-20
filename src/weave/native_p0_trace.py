"""Private Torch/CUPTI trace collection for one native P0 graph replay.

No Torch import is needed to parse traces or run the CPU tests. The caller starts
the profiler outside capture, before its outer GPU event, and finishes only after
the replay's final GPU event has synchronized. This module does not synchronize,
change streams, or infer a launch grid from model dimensions.
"""

import json
import math
import os
from pathlib import Path
import re
import tempfile


class NativeP0TraceError(RuntimeError):
    """Launch evidence is unavailable, malformed, or incomplete."""


_HELP = (
    " Verify CUDA graph kernel activity and CUPTI launch metadata with the "
    "native P0 GPU smoke in the existing server image; use a compatible "
    "pre-provisioned image if unavailable. Do not install packages online."
)
_MARLIN = re.compile(r"\bMarlin\s*<")


def _fail(message):
    raise NativeP0TraceError(message + _HELP)


def _template_arguments(name):
    match = _MARLIN.search(name)
    if match is None:
        _fail("Marlin template name is unavailable.")
    start = match.end()
    stack = [">"]
    args = []
    token_start = start
    pairs = {"<": ">", "(": ")", "[": "]", "{": "}"}
    for i in range(start, len(name)):
        char = name[i]
        if char in pairs:
            stack.append(pairs[char])
        elif char in ">)]}":
            if not stack or char != stack.pop():
                _fail("Marlin template delimiters are malformed.")
            if not stack:
                args.append(name[token_start:i].strip())
                return args
        elif char == "," and len(stack) == 1:
            args.append(name[token_start:i].strip())
            token_start = i + 1
    _fail("Marlin template is truncated.")


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("CUPTI " + label + " must be an integer.")
    if not math.isfinite(value) or int(value) != value or value < minimum:
        _fail("CUPTI " + label + " is invalid.")
    return int(value)


def _template_integer(value, label):
    # Demanglers can retain an explicit fundamental-type cast.
    value = re.sub(r"^\((?:int|unsigned int|long|unsigned long)\)\s*", "", value)
    if not re.fullmatch(r"[+-]?\d+[uUlL]*", value):
        _fail("Marlin template " + label + " is unknown.")
    return int(re.sub(r"[uUlL]+$", "", value))


def _template_bool(value, label):
    value = re.sub(r"^\(bool\)\s*", "", value)
    if value not in ("true", "false", "0", "1"):
        _fail("Marlin template " + label + " is unknown.")
    return value in ("true", "1")


def parse_marlin_template(name):
    """Parse the pinned 13-argument Marlin kernel signature, without guessing.

    Positions follow marlin_template.h: scalar, weight ID, scale ID, threads,
    M/N/K blocks, M8 flag, stages, group blocks, zero-point flag, EP, bias.
    Scalar type IDs remain demangled strings because their enum meanings are
    defined by the pinned native source, not by CUPTI.
    """
    args = _template_arguments(name)
    if len(args) != 13 or any(not arg for arg in args):
        _fail("Marlin template requires the pinned 13-argument signature.")
    result = {
        "scalar_type": args[0],
        "weight_type_id": args[1],
        "scale_type_id": args[2],
    }
    for index, field in (
        (3, "threads"), (4, "thread_m_blocks"), (5, "thread_n_blocks"),
        (6, "thread_k_blocks"), (8, "stages"), (9, "group_blocks"),
    ):
        result[field] = _template_integer(args[index], field)
    for index, field in (
        (7, "m_block_size_8"), (10, "is_zp_float"), (11, "is_ep"), (12, "has_bias"),
    ):
        result[field] = _template_bool(args[index], field)
    if any(result[key] <= 0 for key in (
        "threads", "thread_m_blocks", "thread_n_blocks", "thread_k_blocks", "stages",
    )):
        _fail("Marlin template tile/thread/pipeline dimensions are invalid.")
    if result["m_block_size_8"] and result["thread_m_blocks"] != 1:
        _fail("Marlin template M8 requires one M block.")
    result.update(
        tile_m=8 if result["m_block_size_8"] else 16 * result["thread_m_blocks"],
        tile_n=16 * result["thread_n_blocks"],
        tile_k=16 * result["thread_k_blocks"],
    )
    return result


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _dimensions(value, label):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        _fail("CUPTI " + label + " launch dimensions are missing or malformed.")
    return [_integer(item, label, 1) for item in value]


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        _fail("CUPTI " + label + " is missing or nonfinite.")
    return value


def parse_chrome_trace(payload, *, expected_layers=43):
    """Return only public kernel resources and GPU-clock timing scalars.

    ``expected_layers=None`` is the generic GPU smoke mode. Native mode requires
    exactly two Marlin kernels per layer in GPU timestamp order. This mapping
    assumes the caller brackets one complete replay; matching captured Python
    call metadata is a separate check performed by the native profiler.
    """
    if expected_layers is not None and (
        isinstance(expected_layers, bool) or not isinstance(expected_layers, int) or expected_layers <= 0
    ):
        _fail("Expected native layer count must be a positive integer.")
    events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        _fail("CUPTI Chrome traceEvents are missing.")
    kernels = []
    for event in events:
        if not isinstance(event, dict):
            continue
        categories = {item.strip().lower() for item in str(event.get("cat", "")).split(",")}
        if "kernel" not in categories or event.get("ph") != "X":
            continue
        name = event.get("name")
        args = event.get("args")
        if not isinstance(name, str) or not name or not isinstance(args, dict):
            _fail("CUPTI kernel name or launch arguments are missing.")
        args = {_key(key): value for key, value in args.items()}
        grid = _dimensions(args.get("grid"), "grid")
        block = _dimensions(args.get("block"), "block")
        registers = _integer(args.get("registersperthread"), "registers per thread")
        row = {
            # Kernel names are public, but a profiler must never leak addresses.
            "name": re.sub(r"0x[0-9a-fA-F]+", "<address>", name),
            "grid": grid, "block": block,
            "threads_per_block": math.prod(block),
            "registers_per_thread": registers,
            "start_us": _number(event.get("ts"), "kernel timestamp"),
            "duration_us": _number(event.get("dur"), "kernel duration"),
        }
        if "sharedmemory" in args:
            row["shared_memory_bytes"] = _integer(args["sharedmemory"], "shared memory")
        else:
            row["shared_memory_bytes"] = None
        for source, target in (
            ("staticsharedmemory", "static_shared_memory_bytes"),
            ("dynamicsharedmemory", "dynamic_shared_memory_bytes"),
        ):
            if source in args:
                row[target] = _integer(args[source], source)
        if row["shared_memory_bytes"] is None and not any(
            key in row for key in ("static_shared_memory_bytes", "dynamic_shared_memory_bytes")
        ):
            _fail("CUPTI shared memory launch metadata is missing.")
        if _MARLIN.search(name):
            row["template"] = parse_marlin_template(name)
            if row["threads_per_block"] != row["template"]["threads"]:
                _fail("CUPTI block threads disagree with the Marlin template threads.")
        kernels.append(row)
    if not kernels:
        _fail("No complete CUDA kernel records were exported by CUPTI.")
    kernels.sort(key=lambda row: row["start_us"])
    marlin = [row for row in kernels if "template" in row]
    if expected_layers is not None:
        expected = 2 * expected_layers
        if len(marlin) != expected:
            _fail(f"Native replay requires exactly {expected} Marlin kernels; observed {len(marlin)}.")
        for index, row in enumerate(marlin):
            if index and row["start_us"] == marlin[index - 1]["start_us"]:
                _fail("Marlin GPU timestamps tie; layer/role ordering is ambiguous.")
            row.update(layer=index // 2, role="gate_up" if index % 2 == 0 else "down")
    return {
        "status": "complete", "expected_layers": expected_layers,
        "kernel_count": len(kernels), "marlin_kernel_count": len(marlin),
        "kernels": kernels,
        "timing_clock": "CUPTI GPU timestamp, microseconds",
        "used_for_formal_performance": False,
    }


class LaunchTrace:
    """One-use profiler; raw output is restricted to ``private_dir``."""

    def __init__(self, torch, rank, private_dir, *, expected_layers=43):
        self._torch = torch
        self.rank = _integer(rank, "rank")
        self._private_dir = Path(private_dir)
        self._expected_layers = expected_layers
        self._profile = None
        self._state = "new"

    def begin(self):
        if self._state != "new":
            _fail("LaunchTrace can collect only one replay.")
        self._state = "starting"
        try:
            self._private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._private_dir.chmod(0o700)
            profiler = self._torch.profiler
            self._profile = profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False,
            )
            self._profile.start()
            self._state = "active"
        except Exception as exc:
            if self._profile is not None:
                try:
                    self._profile.stop()
                except Exception:
                    pass
            self._state = "failed"
            raise NativeP0TraceError("Torch CUDA profiler could not start." + _HELP) from exc

    def finish(self):
        if self._state != "active":
            _fail("LaunchTrace.finish requires an active profiler.")
        self._state = "stopping"
        try:
            self._profile.stop()
            self._state = "stopped"
            descriptor, raw_path = tempfile.mkstemp(
                prefix=f"native-p0-rank{self.rank}-", suffix=".json", dir=self._private_dir,
            )
            os.close(descriptor)
            self._profile.export_chrome_trace(raw_path)
            os.chmod(raw_path, 0o600)
            with open(raw_path, encoding="utf-8") as handle:
                result = parse_chrome_trace(json.load(handle), expected_layers=self._expected_layers)
            result["rank"] = self.rank
            self._state = "complete"
            return result
        except NativeP0TraceError:
            self._state = "failed"
            raise
        except Exception as exc:
            self._state = "failed"
            raise NativeP0TraceError("Torch CUDA trace stop/export/parse failed." + _HELP) from exc

    def abort(self):
        """Best-effort cleanup; preserve the caller's original runtime failure."""
        if self._state == "active":
            try:
                self._profile.stop()
            except Exception:
                pass
        if self._state != "complete":
            self._state = "aborted"
