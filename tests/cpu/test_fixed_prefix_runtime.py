"""CPU contract tests; fake Torch tensors do not claim GPU execution."""
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from src.weave import fixed_prefix_runtime as runtime
except ImportError:
    runtime = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Tensor:
    def __init__(self, data, dtype='torch.float32'):
        self.data = copy.deepcopy(data)
        self.dtype = dtype
        self.device = 'cpu'
        self.shape = (len(data), len(data[0])) if data and isinstance(data[0], list) else (len(data),)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def to(self, **kwargs): return Tensor(self.data, self.dtype)
    def tolist(self): return copy.deepcopy(self.data)
    def view(self, dtype): return self
    def numpy(self): return self
    def tobytes(self):
        flat = [v for row in self.data for v in row] if len(self.shape) == 2 else self.data
        return struct.pack('<' + ('f' * len(flat)), *flat)
    def float(self): return self
    def double(self): return self
    def all(self): return SimpleNamespace(item=lambda: all(self.data))
    def argmax(self, dim=-1): return Tensor([max(range(len(row)), key=lambda i: row[i]) for row in self.data], 'torch.int64')
    def topk(self, k, dim=-1):
        return SimpleNamespace(values=Tensor([sorted(row, reverse=True)[:k] for row in self.data]))
    def fill_(self, value):
        self.data = [[value] * len(row) for row in self.data]
        return self
    def scatter_(self, dim, index, value):
        for row, ix in zip(self.data, index.data): row[ix[0]] = value
        return self


class FakeTorch:
    uint8 = 'torch.uint8'
    int64 = 'torch.int64'
    def __init__(self):
        self.events = []
        self.capturing = False
        self.cuda = SimpleNamespace(is_current_stream_capturing=lambda: self.capturing,
                                    synchronize=lambda: self.events.append('synchronize'))
    def isfinite(self, tensor):
        return Tensor([math.isfinite(value) for row in tensor.data for value in row])
    def tensor(self, data, **kwargs): return Tensor(data, kwargs.get('dtype', 'torch.float32'))
    def save(self, tensor, target):
        self.events.append('save_logits')
        Path(target).write_text(json.dumps(tensor.data))


class Mode:
    def __init__(self, decode): self.decode = decode
    def is_decode(self): return self.decode
    def is_extend(self): return not self.decode


def fixture(phase='measure', step=0, order=None, steps=8):
    order = list(range(32)) if order is None else order
    prompts = [[i + 1, i + 2] for i in range(32)]
    teacher_ids = [[(row + index) % 40 for index in range(8)] for row in range(32)]
    config = dict(schema_version=1, steps=8, selection_sha256='a' * 64, vocab_size=40,
                  requests=[dict(row_slot=i, original_row_index=i + 7, input_tokens=2,
                                 prompt_sha256='b' * 64, input_ids_sha256=digest(prompts[i])) for i in range(32)])
    teacher = dict(schema_version=1, selection_sha256='a' * 64, steps=8,
                   output_ids=teacher_ids, sha256=digest(teacher_ids))
    params = [{'weave_fixed_prefix': dict(phase=phase, row_slot=i, selection_sha256='a' * 64,
                                         steps=steps)} for i in order]
    inputs = [teacher_ids[i][step - 1] for i in order] if step else [v for i in order for v in prompts[i]]
    batch = SimpleNamespace(batch_size=32, forward_mode=Mode(bool(step)),
        seq_lens=Tensor([2 + step] * 32, 'torch.int64'),
        positions=Tensor([1 + step] * 32 if step else [0, 1] * 32, 'torch.int64'),
        input_ids=Tensor(inputs, 'torch.int64'), req_pool_indices=Tensor(list(range(32)), 'torch.int64'),
        out_cache_loc=Tensor(list(range(len(inputs))), 'torch.int64'),
        sampling_info=SimpleNamespace(custom_params=params, is_all_greedy=True,
                                      custom_logit_processor={0: (None, None)}))
    return config, teacher, params, batch


class FixedPrefixRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runtime, 'fixed-prefix runtime is not implemented')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.torch = FakeTorch()
        self.config, self.teacher, self.params, self.batch = fixture()
        (self.root / 'config.json').write_text(json.dumps(self.config))
        self.teacher_path = self.root / 'teacher.json'
        self.teacher_path.write_text(json.dumps(self.teacher))
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, torch=self.torch).start()
        self.rt = runtime.FixedPrefixRuntime(self.torch, 0, 4, self.root, self.teacher_path,
                                             {'srt/layers/sampler.py': 'd' * 64}, 'sha256:test')
    def logits(self): return Tensor([[float(i) for i in range(40)] for _ in range(32)])
    def call(self, params=None, batch=None, logits=None):
        return self.rt.process(logits or self.logits(), params or self.params, batch or self.batch)
    def test_teacher_forcing_saves_original_logits_and_only_hashes_token_ids(self):
        out = self.call()
        self.assertEqual(out.argmax().tolist(), list(range(32)))
        self.assertEqual(json.loads((self.root / 'measure/rank0-step000.pt').read_text())[0], list(range(40)))
        self.rt.after_sample(Tensor(list(range(32)), 'torch.int64'))
        proof = json.loads((self.root / 'measure-rank0.json').read_text())
        self.assertEqual(proof['completed_steps'], [0])
        self.assertEqual(proof['status'], 'running')
        self.assertEqual(proof['numerical_policy'], 'record_only')
        self.assertEqual(proof['records'][0]['native_argmax_sha256'], digest([39] * 32))
        for key in ('output_ids', 'input_ids', 'logits', 'prompt'):
            self.assertNotIn('"' + key + '":', json.dumps(proof))
    def test_decode_uses_actual_prefix_and_preserves_row_order(self):
        self.call(); self.rt.after_sample(Tensor(list(range(32)), 'torch.int64'))
        order = list(reversed(range(32)))
        _, _, params, batch = fixture(step=1, order=order)
        out = self.call(params, batch)
        self.assertEqual(out.argmax().tolist(), [(i + 1) % 40 for i in order])
        self.rt.after_sample(out.argmax())
        proof = json.loads((self.root / 'measure-rank0.json').read_text())
        self.assertEqual(proof['records'][1]['row_slot_order'], order)
        self.assertEqual(proof['records'][1]['positions'], [2] * 32)
    def test_bad_actual_previous_token_cannot_claim_fixed_prefix(self):
        self.call(); self.rt.after_sample(Tensor(list(range(32))))
        _, _, params, batch = fixture(step=1)
        batch.input_ids.data[0] = 999
        with self.assertRaises(runtime.FixedPrefixError): self.call(params, batch)
    def test_wrong_positions_split_batches_duplicate_rows_and_repeated_steps_fail(self):
        for mutate in (lambda p,b: setattr(b, 'batch_size', 31),
                       lambda p,b: p[1]['weave_fixed_prefix'].update(row_slot=0),
                       lambda p,b: b.positions.data.__setitem__(0, 99),
                       lambda p,b: b.input_ids.data.__setitem__(0, 99),
                       lambda p,b: b.seq_lens.data.__setitem__(0, 3)):
            _, _, params, batch = fixture(); mutate(params,batch)
            with self.assertRaises(runtime.FixedPrefixError): self.call(params,batch)
        self.call(); self.rt.after_sample(Tensor(list(range(32))))
        with self.assertRaises(runtime.FixedPrefixError): self.call()
    def test_seed_preserves_logits_and_does_not_require_teacher(self):
        self.teacher_path.unlink()
        _, _, params, batch = fixture('seed')
        original = self.logits(); out = self.call(params,batch,original)
        self.assertIs(out, original)
        self.rt.after_sample(out.argmax())
        proof = json.loads((self.root / 'seed-rank0.json').read_text())
        self.assertIsNone(proof['teacher_sha256'])
    def test_original_nonfinite_and_capture_fail_before_snapshot(self):
        bad = self.logits(); bad.data[0][0] = float('nan')
        with self.assertRaises(runtime.FixedPrefixError): self.call(logits=bad)
        self.assertFalse((self.root / 'measure/rank0-step000.pt').exists())
        self.torch.capturing = True
        with self.assertRaises(runtime.FixedPrefixError): self.call()
        self.assertFalse((self.root / 'measure/rank0-step000.pt').exists())
    def test_corrupt_teacher_digest_is_rejected(self):
        self.teacher['output_ids'][0][0] = 9
        self.teacher_path.write_text(json.dumps(self.teacher))
        with self.assertRaises(runtime.FixedPrefixError): self.call()
    def test_wrapper_forces_before_native_ngram_update_and_returns_native_result(self):
        observed = []
        rt = self.rt
        class Runner:
            def sample(self, logits_output, forward_batch):
                logits = runtime.FixedPrefixProcessor()(logits_output, forward_batch.sampling_info.custom_params)
                native_result = logits.argmax()
                observed.append(native_result.tolist())  # native ngram update sees forced IDs
                return native_result
        runtime.patch_model_runner(Runner, lambda runner: rt)
        output = Runner().sample(self.logits(), self.batch)
        self.assertEqual(observed, [list(range(32))])
        self.assertEqual(output.tolist(), list(range(32)))
        self.assertIsNone(runtime._CONTEXT.get())
    def test_failed_native_sample_records_sanitized_error_without_completion(self):
        rt = self.rt
        class Runner:
            def sample(self, logits_output, forward_batch):
                runtime.FixedPrefixProcessor()(logits_output, forward_batch.sampling_info.custom_params)
                raise RuntimeError('PRIVATE PROMPT MUST NOT LEAK')
        runtime.patch_model_runner(Runner, lambda runner: rt)
        with self.assertRaises(RuntimeError): Runner().sample(self.logits(), self.batch)
        proof_text = (self.root / 'measure-rank0.json').read_text()
        proof = json.loads(proof_text)
        self.assertEqual(proof['status'], 'failed')
        self.assertEqual(proof['completed_steps'], [])
        self.assertNotIn('PRIVATE', proof_text)
    def test_trace_bounds_surround_one_decode_before_snapshot(self):
        rt, events = self.rt, self.torch.events
        class Trace:
            def __init__(self, *args, **kwargs): pass
            def begin(self): events.append('trace_begin')
            def finish(self): events.append('trace_finish'); return {'marlin_kernel_count': 86}
            def abort(self): events.append('trace_abort')
        rt.trace_factory = Trace
        _, _, params, batch = fixture('trace', steps=2)
        self.call(params,batch); events.append('ngram0'); rt.after_sample(Tensor(list(range(32))))
        self.assertLess(events.index('ngram0'), events.index('trace_begin'))
        events.clear(); events.append('actual_decode')
        _, _, params, batch = fixture('trace', step=1, steps=2)
        out = self.call(params,batch); rt.after_sample(out.argmax())
        self.assertEqual(events[:4], ['actual_decode', 'synchronize', 'trace_finish', 'save_logits'])
        proof = json.loads((self.root / 'trace-rank0.json').read_text())
        self.assertEqual(proof['status'], 'passed')
        self.assertEqual(proof['launch_trace']['marlin_kernel_count'], 86)
    def test_source_mismatch_rejects_before_mutating_model_runner(self):
        with self.assertRaises(runtime.FixedPrefixError): runtime.verify_fixed_prefix_sources(self.root)
        class Runner:
            def sample(self, logits, batch): return logits
        original = Runner.sample
        module = SimpleNamespace(__file__=str(self.root / 'srt/model_executor/model_runner.py'))
        loader = SimpleNamespace(exec_module=lambda target: setattr(target, 'ModelRunner', Runner))
        with self.assertRaises(runtime.FixedPrefixError): runtime._Loader(loader).exec_module(module)
        self.assertIs(Runner.sample, original)
    def test_unknown_shape_or_non_greedy_sampling_does_not_fallback(self):
        with self.assertRaises(runtime.FixedPrefixError): self.call(logits=Tensor([[1.0, 2.0]] * 32))
        self.batch.sampling_info.is_all_greedy = False
        with self.assertRaises(runtime.FixedPrefixError): self.call()
        self.batch.sampling_info.is_all_greedy = True
        self.batch.sampling_info.custom_logit_processor[1] = (None, None)
        with self.assertRaises(runtime.FixedPrefixError): self.call()
    def test_all_eight_steps_are_needed_for_pass_and_forced_result_is_checked(self):
        for step in range(8):
            _, _, params, batch = fixture(step=step)
            out = self.call(params, batch)
            self.rt.after_sample(out.argmax())
            proof = json.loads((self.root / 'measure-rank0.json').read_text())
            self.assertEqual(proof['status'], 'passed' if step == 7 else 'running')
            self.assertEqual(proof['completed_steps'], list(range(step + 1)))
    def test_unfinished_phase_and_wrong_native_sample_cannot_pass(self):
        self.call()
        with self.assertRaises(runtime.FixedPrefixError): self.rt.after_sample(Tensor([39] * 32))
        self.rt.finish()
        proof = json.loads((self.root / 'measure-rank0.json').read_text())
        self.assertEqual(proof['status'], 'failed')
        self.assertEqual(proof['completed_steps'], [])
    def test_import_does_not_eagerly_load_torch_or_sglang(self):
        import subprocess
        result = subprocess.run([sys.executable, '-c',
            'import sys; import src.weave.fixed_prefix_runtime; assert "torch" not in sys.modules; assert "sglang" not in sys.modules'],
            capture_output=True,text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
