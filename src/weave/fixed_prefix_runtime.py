"""Opt-in, source-pinned fixed-prefix sampling evidence outside CUDA capture.

Import this module as ``src.weave.fixed_prefix_runtime`` on both the HTTP client
and server. It deliberately imports neither Torch nor SGLang at module load.
Only original CPU logits and private teacher files contain token information;
phase proofs contain scalar metadata and commitments. Finite differences are
record-only. This module inserts no layer events and changes no native kernels.
"""
import atexit
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import re
import sys
import tempfile

_ROOT = Path(__file__).resolve().parents[2]
_TARGET = 'sglang.srt.model_executor.model_runner'
_CONTEXT = ContextVar('weave_fixed_prefix_sample', default=None)
_SOURCE_FILES = ('src/weave/fixed_prefix_runtime.py', 'configs/fixed_prefix_sources.json',
                 'src/weave/native_p0_trace.py', 'timing_hooks/sitecustomize.py')


class FixedPrefixError(RuntimeError):
    """Malformed source, prefix, coverage, or runtime evidence."""


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def fixed_prefix_source_sha256():
    manifest = {name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest()
                for name in _SOURCE_FILES}
    return canonical_sha256(manifest)


_LOADED_SOURCE_SHA256 = fixed_prefix_source_sha256()


def verify_fixed_prefix_sources(package_root):
    pins = json.loads((_ROOT / 'configs/fixed_prefix_sources.json').read_text())['sha256']
    for name, expected in pins.items():
        path = Path(package_root) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise FixedPrefixError('Fixed-prefix native source mismatch: ' + name)
    return pins


def tensor_sha256(tensor):
    """Commit dtype, shape and exact CPU bytes without publishing token content."""
    import torch
    cpu = tensor.detach().cpu().contiguous()
    header = json.dumps({'dtype': str(cpu.dtype), 'shape': list(cpu.shape)},
                        sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header + b'\0' + cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise FixedPrefixError('Invalid ' + label)
    return value


def _sha256(value, label):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise FixedPrefixError('Invalid ' + label)
    return value


def _ints(tensor, label):
    values = tensor.detach().cpu().tolist()
    if not isinstance(values, list) or any(type(v) is not int for v in values):
        raise FixedPrefixError('Malformed actual ' + label)
    return values


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o770)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o664)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_tensor(torch, path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o770)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    os.close(fd)
    try:
        torch.save(tensor, temporary)
        os.chmod(temporary, 0o664)
        # Persist bytes before making this step visible to the controller.
        with open(temporary, 'rb') as handle:
            os.fsync(handle.fileno())
            digest = hashlib.sha256(handle.read()).hexdigest()
        os.replace(temporary, path)
        return digest
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_config(config):
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise FixedPrefixError('Invalid fixed-prefix configuration schema')
    _sha256(config.get('selection_sha256'), 'selection commitment')
    if _integer(config.get('steps'), 'steps', 1) != 8:
        raise FixedPrefixError('Fixed-prefix probe requires eight prediction positions')
    _integer(config.get('vocab_size'), 'vocabulary size', 2)
    requests = config.get('requests')
    if not isinstance(requests, list) or len(requests) != 32:
        raise FixedPrefixError('Fixed-prefix configuration requires 32 requests')
    slots, indices = [], []
    for request in requests:
        slots.append(_integer(request.get('row_slot'), 'row slot'))
        indices.append(_integer(request.get('original_row_index'), 'original row index'))
        _integer(request.get('input_tokens'), 'input length', 1)
        for name in ('prompt_sha256', 'input_ids_sha256'):
            _sha256(request.get(name), name)
    if sorted(slots) != list(range(32)) or len(set(indices)) != 32:
        raise FixedPrefixError('Duplicate or incomplete configured row mapping')
    return {request['row_slot']: request for request in requests}


class FixedPrefixRuntime:
    def __init__(self, torch, rank, world_size, private_dir, teacher_path,
                 native_sources, image_id):
        self.torch = torch
        self.rank = _integer(rank, 'rank')
        self.world_size = _integer(world_size, 'world size', 1)
        if self.world_size != 4 or self.rank >= self.world_size:
            raise FixedPrefixError('Fixed-prefix runtime requires four native TP ranks')
        self.private_dir, self.teacher_path = Path(private_dir), Path(teacher_path)
        self.config = json.loads((self.private_dir / 'config.json').read_text())
        self.requests = _validate_config(self.config)
        self.config_sha256 = canonical_sha256(self.config)
        self.native_sources = dict(native_sources)
        self.image_id = image_id
        if not isinstance(image_id, str) or not image_id.startswith('sha256:'):
            raise FixedPrefixError('Resolved Docker image identity is required')
        self.phases, self.pending, self.trace, self.trace_factory = {}, None, None, None
        self.seed_outputs = {}
        self.package_root = None

    def _phase(self, phase, steps, teacher):
        if phase not in self.phases:
            self.phases[phase] = dict(schema_version=1, status='running', phase=phase,
                numerical_policy='record_only', raw_content_exported=False,
                rank=self.rank, world_size=self.world_size, image_id=self.image_id,
                selection_sha256=self.config['selection_sha256'],
                config_sha256=self.config_sha256,
                teacher_sha256=None if teacher is None else teacher['sha256'],
                expected_steps=steps, completed_steps=[], records=[],
                fixed_prefix_source_sha256=_LOADED_SOURCE_SHA256,
                native_sources=self.native_sources,
                sampler_source_closure_sha256=canonical_sha256(self.native_sources),
                logical_kv_positions_enforced=True,
                physical_kv_slots_policy='record_commitments_only',
                kv_numerical_contents_claimed_identical=False,
                stage_events_inserted_by_fixed_prefix_runtime=0,
                disable_overlap_schedule=True,
                positions_semantics='per_row_seq_len_minus_one',
                trace_includes_native_sampler_extras=(phase == 'trace'))
        proof = self.phases[phase]
        if proof['status'] == 'failed' or proof['expected_steps'] != steps or proof['teacher_sha256'] != (None if teacher is None else teacher['sha256']):
            raise FixedPrefixError('Fixed-prefix phase identity changed or failed')
        return proof

    def _write(self, proof):
        _atomic_json(self.private_dir / (proof['phase'] + f'-rank{self.rank}.json'), proof)

    def _teacher(self, phase):
        if phase == 'seed':
            return None
        if not self.teacher_path.is_file():
            raise FixedPrefixError('Private teacher is missing; complete native seed first')
        teacher = json.loads(self.teacher_path.read_text())
        ids = teacher.get('output_ids')
        if (teacher.get('schema_version') != 1 or teacher.get('steps') != self.config['steps']
                or teacher.get('selection_sha256') != self.config['selection_sha256']
                or not isinstance(ids, list) or len(ids) != 32
                or any(not isinstance(row, list) or len(row) != self.config['steps'] for row in ids)):
            raise FixedPrefixError('Private teacher identity or coverage is malformed')
        if any(type(token) is not int or not 0 <= token < self.config['vocab_size']
               for row in ids for token in row):
            raise FixedPrefixError('Private teacher has invalid token IDs')
        if canonical_sha256(ids) != teacher.get('sha256'):
            raise FixedPrefixError('Private teacher digest mismatch')
        return teacher

    def process(self, logits, custom_param_list, batch):
        # Nothing below may execute while capturing a CUDA graph.
        if self.torch.cuda.is_current_stream_capturing():
            raise FixedPrefixError('Fixed-prefix sampling must execute outside CUDA capture')
        if self.pending is not None:
            raise FixedPrefixError('Previous native sample has not completed')
        if not isinstance(custom_param_list, list) or len(custom_param_list) != 32:
            raise FixedPrefixError('Processor requires all 32 request rows')
        if batch.batch_size != 32 or tuple(logits.shape) != (32, self.config['vocab_size']):
            raise FixedPrefixError('Native batch or original logits shape is malformed')
        if custom_param_list != batch.sampling_info.custom_params:
            raise FixedPrefixError('Processor mask does not cover the complete native batch')
        metadata = [param.get('weave_fixed_prefix') if isinstance(param, dict) else None
                    for param in custom_param_list]
        if any(not isinstance(item, dict) for item in metadata):
            raise FixedPrefixError('Fixed-prefix request metadata is missing')
        phase = metadata[0].get('phase')
        steps = metadata[0].get('steps')
        if phase not in ('seed', 'measure', 'trace') or type(steps) is not int or steps != (2 if phase == 'trace' else 8):
            raise FixedPrefixError('Unsupported fixed-prefix phase or length')
        for item in metadata:
            if (item.get('phase') != phase or item.get('steps') != steps or
                    item.get('selection_sha256') != self.config['selection_sha256']):
                raise FixedPrefixError('Mixed fixed-prefix phase or selection')
        order = [_integer(item.get('row_slot'), 'row slot') for item in metadata]
        if sorted(order) != list(range(32)):
            raise FixedPrefixError('Processor row mapping must cover each request once')
        # Stop the profiler before any logit snapshot, CPU tensor transfer, or
        # source-file verification. The preceding model decode is now complete.
        if phase == 'trace' and self.trace is not None:
            self.torch.cuda.synchronize()
            trace_result = self.trace.finish()
            self.trace = None
            self.phases['trace']['launch_trace'] = trace_result
        if (not batch.sampling_info.is_all_greedy or
                len(batch.sampling_info.custom_logit_processor) != 1):
            raise FixedPrefixError('Fixed-prefix probe requires native greedy sampling and one processor')
        if fixed_prefix_source_sha256() != _LOADED_SOURCE_SHA256:
            raise FixedPrefixError('Fixed-prefix implementation changed after import')
        if self.package_root is not None:
            if verify_fixed_prefix_sources(self.package_root) != self.native_sources:
                raise FixedPrefixError('Native source identity changed after setup')
        if canonical_sha256(json.loads((self.private_dir / 'config.json').read_text())) != self.config_sha256:
            raise FixedPrefixError('Fixed-prefix configuration changed after launch')
        teacher = self._teacher(phase)
        proof = self._phase(phase, steps, teacher)
        seq_lens = _ints(batch.seq_lens, 'sequence lengths')
        positions_actual = _ints(batch.positions, 'positions')
        inputs = _ints(batch.input_ids, 'input IDs')
        if len(seq_lens) != 32:
            raise FixedPrefixError('Native sequence lengths do not cover 32 rows')
        derived = [length - self.requests[slot]['input_tokens'] for slot, length in zip(order, seq_lens)]
        if len(set(derived)) != 1 or not 0 <= derived[0] < steps:
            raise FixedPrefixError('Native rows are not aligned at one prediction position')
        step = derived[0]
        if step != len(proof['completed_steps']):
            raise FixedPrefixError('Missing, duplicate, or reordered prediction step')
        positions = [length - 1 for length in seq_lens]
        rows = []
        if step == 0:
            if not batch.forward_mode.is_extend() or batch.forward_mode.is_decode():
                raise FixedPrefixError('Prediction zero must use native extend')
            expected_positions = [p for length in seq_lens for p in range(length)]
            if positions_actual != expected_positions or len(inputs) != sum(seq_lens):
                raise FixedPrefixError('Prefill must contain the complete uncached native input')
            offset = 0
            for slot, length in zip(order, seq_lens):
                row = inputs[offset:offset + length]
                if canonical_sha256(row) != self.requests[slot]['input_ids_sha256']:
                    raise FixedPrefixError('Actual prefill tokens differ from the selected request')
                rows.append(row)
                offset += length
        else:
            if not batch.forward_mode.is_decode() or positions_actual != positions or len(inputs) != 32:
                raise FixedPrefixError('Decode input or logical KV position is malformed')
            previous = ([self.seed_outputs[slot][step - 1] for slot in order] if teacher is None
                        else [teacher['output_ids'][slot][step - 1] for slot in order])
            if inputs != previous:
                raise FixedPrefixError('Actual native decode input differs from the fixed prefix')
            rows = [[token] for token in inputs]
        if not self.torch.isfinite(logits).all().item():
            raise FixedPrefixError('Original logits contain nonfinite values')
        cpu = logits.detach().to(device='cpu', copy=True).contiguous()
        top2 = cpu.double().topk(2, dim=-1).values.tolist()
        margins = [row[0] - row[1] for row in top2]
        argmax = _ints(cpu.argmax(dim=-1), 'native argmax')
        record = dict(step=step, batch_size=32, mode='extend' if step == 0 else 'decode',
            row_slot_order=order, seq_lens=seq_lens, positions=positions,
            original_logits_finite=True, original_logits_dtype=str(cpu.dtype),
            original_logits_shape=list(cpu.shape), logits_tensor_sha256=tensor_sha256(cpu),
            input_ids_sha256=canonical_sha256(inputs),
            input_ids_by_row_sha256=[canonical_sha256(row) for row in rows],
            req_pool_indices_sha256=tensor_sha256(batch.req_pool_indices),
            out_cache_loc_sha256=tensor_sha256(batch.out_cache_loc),
            native_argmax_sha256=canonical_sha256(argmax),
            top2_margin_min=min(margins), top2_margin_max=max(margins),
            top2_margin_mean=sum(margins) / len(margins),
            captured=False)
        path = self.private_dir / phase / f'rank{self.rank}-step{step:03d}.pt'
        if path.exists():
            raise FixedPrefixError('Duplicate private logit snapshot')
        record['logits_file_sha256'] = _save_tensor(self.torch, path, cpu)
        chosen = None if teacher is None else [teacher['output_ids'][slot][step] for slot in order]
        self.pending = (proof, record, chosen)
        self._write(proof)
        if chosen is not None:
            indices = self.torch.tensor([[token] for token in chosen], dtype=self.torch.int64,
                                        device=logits.device)
            logits.fill_(float('-inf'))
            logits.scatter_(1, indices, 0.0)
        return logits

    def after_sample(self, native_result):
        """Commit only after native sampling and native ngram update succeeded."""
        if self.pending is None:
            return
        proof, record, chosen = self.pending
        output = _ints(native_result, 'native sampled token IDs')
        if len(output) != 32 or any(token < 0 or token >= self.config['vocab_size'] for token in output):
            raise FixedPrefixError('Native sampler returned malformed token IDs')
        if chosen is not None and output != chosen:
            raise FixedPrefixError('Native sampler did not apply the requested teacher prefix')
        if proof['phase'] == 'seed':
            for slot, token in zip(record['row_slot_order'], output):
                self.seed_outputs.setdefault(slot, []).append(token)
        record['sampled_ids_sha256'] = canonical_sha256(output)
        proof['records'].append(record)
        proof['completed_steps'].append(record['step'])
        self.pending = None
        complete = len(proof['completed_steps']) == proof['expected_steps']
        if proof['phase'] == 'trace' and record['step'] == 0:
            from .native_p0_trace import LaunchTrace
            factory = self.trace_factory or LaunchTrace
            self.torch.cuda.synchronize()
            self.trace = factory(self.torch, self.rank,
                                 self.private_dir / f'trace-private-rank{self.rank}', expected_layers=43)
            # Start only after the native prefill sample and ngram operations.
            self.trace.begin()
        if complete:
            if proof['phase'] == 'trace' and 'launch_trace' not in proof:
                raise FixedPrefixError('Native trace did not complete one decode')
            proof['status'] = 'passed'
        self._write(proof)

    def fail(self, error):
        if self.trace is not None:
            self.trace.abort()
            self.trace = None
        proofs = [self.pending[0]] if self.pending is not None else [
            proof for proof in self.phases.values() if proof['status'] != 'passed']
        for proof in proofs:
            proof['status'] = 'failed'
            proof['error_type'] = type(error).__name__
            self._write(proof)
        self.pending = None

    def finish(self):
        if self.pending is not None or self.trace is not None or any(
                proof['status'] == 'running' for proof in self.phases.values()):
            self.fail(FixedPrefixError('Incomplete native phase at shutdown'))


class FixedPrefixProcessor:
    """Native CustomLogitProcessor-compatible callable, serialized by class."""
    @classmethod
    def to_str(cls):
        if cls.__module__ != 'src.weave.fixed_prefix_runtime':
            raise FixedPrefixError('Serialize the canonical src.weave.fixed_prefix_runtime class')
        import dill
        return json.dumps({'callable': dill.dumps(cls).hex()})

    def __call__(self, logits, custom_param_list=None):
        context = _CONTEXT.get()
        if context is None:
            raise FixedPrefixError('Fixed-prefix processor requires native sample context')
        runtime = context['runtime']
        if runtime is None:
            runtime = context['setup'](context['runner'])
            context['runtime'] = runtime
        return runtime.process(logits, custom_param_list, context['batch'])


def patch_model_runner(cls, setup):
    if getattr(cls, '_weave_fixed_prefix_sample', False):
        return
    original = cls.sample
    @wraps(original)
    def sample(self, logits_output, forward_batch):
        if _CONTEXT.get() is not None:
            raise FixedPrefixError('Nested native sampling is unsupported')
        context = dict(runner=self, batch=forward_batch, setup=setup, runtime=None)
        token = _CONTEXT.set(context)
        try:
            result = original(self, logits_output, forward_batch)
            if context['runtime'] is not None:
                context['runtime'].after_sample(result)
            return result
        except BaseException as error:
            if context['runtime'] is not None:
                context['runtime'].fail(error)
            raise
        finally:
            _CONTEXT.reset(token)
    cls.sample = sample
    cls._weave_fixed_prefix_sample = True


def _setup_runner(runner):
    if hasattr(runner, '_weave_fixed_prefix_runtime'):
        return runner._weave_fixed_prefix_runtime
    import sglang
    import torch
    pins = verify_fixed_prefix_sources(Path(sglang.__file__).parent)
    private_dir = Path(os.environ['WEAVE_FIXED_PREFIX_DIR']).resolve()
    teacher_path = Path(os.environ['WEAVE_FIXED_PREFIX_TEACHER']).resolve()
    # The controller explicitly mounts its private output under /work.
    if Path('/work') not in private_dir.parents or Path('/work') not in teacher_path.parents:
        raise FixedPrefixError('Fixed-prefix private paths must be under /work')
    if not runner.server_args.enable_custom_logit_processor:
        raise FixedPrefixError('Native custom logit processor flag is disabled')
    if not runner.server_args.disable_overlap_schedule:
        raise FixedPrefixError('Fixed-prefix evidence requires disabled overlap scheduling')
    runtime = FixedPrefixRuntime(torch, runner.ps.tp_rank, runner.ps.tp_size,
                                 private_dir, teacher_path, pins, os.environ['WEAVE_IMAGE_ID'])
    runtime.package_root = Path(sglang.__file__).parent
    runner._weave_fixed_prefix_runtime = runtime
    atexit.register(runtime.finish)
    return runtime


class _Loader(importlib.abc.Loader):
    def __init__(self, loader): self.loader = loader
    def create_module(self, spec): return self.loader.create_module(spec)
    def exec_module(self, module):
        self.loader.exec_module(module)
        # Verify all sampling interfaces before assigning any wrapper.
        verify_fixed_prefix_sources(Path(module.__file__).resolve().parents[2])
        patch_model_runner(module.ModelRunner, _setup_runner)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError('Pinned native ModelRunner is unavailable')
        spec.loader = _Loader(spec.loader)
        return spec


def install_fixed_prefix_hook():
    if os.environ.get('WEAVE_FIXED_PREFIX') != '1':
        return
    if __name__ != 'src.weave.fixed_prefix_runtime':
        raise FixedPrefixError('Use the canonical src.weave.fixed_prefix_runtime import')
    if _TARGET in sys.modules:
        raise FixedPrefixError('Fixed-prefix hook must precede native ModelRunner import')
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
