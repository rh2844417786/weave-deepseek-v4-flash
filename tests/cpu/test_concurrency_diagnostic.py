import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from scripts.run_concurrency_diagnostic import load_command,validate_dataset,summarize_profiles
except ImportError:
    load_command=validate_dataset=summarize_profiles=None


class ConcurrencyDiagnosticTest(unittest.TestCase):
    def test_warmup_does_not_start_or_reset_the_formal_capture_window(self):
        import asyncio
        from scripts.bench_c32_profile import install_window_hook
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'window.json';observations=[];resets=[]
            async def original(request_func_input,pbar=None):
                observations.append((request_func_input.output_len,marker.exists()))
                return SimpleNamespace(success=len(observations)!=3,output_len=request_func_input.output_len)
            benchmark=SimpleNamespace(ASYNC_REQUEST_FUNCS={'sglang':original})
            with patch('scripts.bench_c32_profile.reset_profiles',side_effect=lambda base:resets.append(base)):
                install_window_hook(benchmark,marker,31000)
                async def run():
                    for n in [32,1024,1024]:
                        await benchmark.ASYNC_REQUEST_FUNCS['sglang'](SimpleNamespace(output_len=n))
                asyncio.run(run())
            self.assertEqual(observations,[(32,False),(1024,True),(1024,True)])
            self.assertEqual(resets,[31000])
            self.assertEqual(json.loads(marker.read_text())['status'],'load_started')
            counts=json.loads((marker.parent/'load-progress.json').read_text())
            self.assertEqual(counts['started'],2)
            self.assertEqual(counts['succeeded'],1)
            self.assertEqual(counts['failed'],1)
            self.assertEqual(counts['completed_output_tokens'],1024)

    def test_batch32_peak_without_retained_batch32_timing_is_incomplete(self):
        import scripts.run_concurrency_diagnostic as module
        self.assertTrue(hasattr(module,'validate_diagnostic_completion'))
        summary={'observed_max_decode_batch':32,'ranks':{str(r):{'aligned_recent_decode':[
            {'valid_rows':1,'outside_persistent_ms':1.}]} for r in range(4)}}
        progress={'started':32,'succeeded':0,'failed':0,'cancelled':0}
        with self.assertRaisesRegex(ValueError,'C32 timing'):
            module.validate_diagnostic_completion(summary,progress,False)
        for value in summary['ranks'].values():
            value['aligned_recent_decode']=[{'valid_rows':32,'outside_persistent_ms':1.}]*2
        module.validate_diagnostic_completion(summary,progress,False)
        with self.assertRaises(ValueError):module.validate_diagnostic_completion(summary,dict(progress,failed=1),False)
        with self.assertRaises(ValueError):module.validate_diagnostic_completion(summary,progress,True)

    def test_load_command_uses_existing_image_local_tokenizer_and_exact_workload(self):
        self.assertIsNotNone(load_command)
        argv=load_command('sha256:existing',Path('/srv/work'),Path('/models/v4'),
                          Path('/srv/work/cases.jsonl'),'diag-123',30001,30110)
        self.assertIn('--pull=never',argv)
        self.assertIn('HF_HUB_OFFLINE=1',argv)
        self.assertIn('NVIDIA_VISIBLE_DEVICES=void',argv)
        self.assertIn('CUDA_VISIBLE_DEVICES=',argv)
        self.assertIn('/models/v4:/model:ro',argv)
        self.assertEqual(argv[argv.index('--max-concurrency')+1],'32')
        self.assertEqual(argv[argv.index('--sharegpt-output-len')+1],'1024')
        self.assertEqual(argv[argv.index('--num-prompts')+1],'1800')
        self.assertEqual(argv[argv.index('--tokenizer')+1],'/model')
        self.assertNotIn('--gpus',argv)

    def test_dataset_identity_rejects_a_silently_changed_probe(self):
        self.assertIsNotNone(validate_dataset)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'dataset.jsonl'
            rows=[{'conversations':[{'value':f'private case {i}'},{'value':'answer'}]} for i in range(9)]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows*200))
            metadata=validate_dataset(path)
            self.assertEqual(metadata['records'],1800)
            self.assertEqual(metadata['unique_prompts'],9)
            self.assertNotIn('private case',json.dumps(metadata))
            path.write_text(json.dumps(rows[0])+'\n')
            with self.assertRaises(ValueError):validate_dataset(path)

    def test_model_rows_are_not_quadrupled_and_layer_times_align_by_sequence(self):
        self.assertIsNotNone(summarize_profiles)
        sample={'sequence':0,'kernel_span_ns':2000000,'schedule':{'c':1,'K':1,'steal':0},
                'workload':{'tasks':48,'groups':24},'phase_envelopes_ns':{'compute':1500000},
                'worker_totals_ns':{'compute_work':10000000}}
        phase={'calls':1,'valid_rows_total':32,'batch_histogram':{'32':1},'sampled_invocations':1,
               'recent_samples':[{'sequence':0,'valid_rows':32,'input_rows':32,'gpu_stream_ms':10.,'host_enqueue_ms':2.}]}
        payload={'rank':0,'forward_profile':{'phases':{'decode':phase}},
                 'layers':{'0':{'decode':{'calls':1,'recent_samples':[sample],
                                        'worker_totals_ns':{'compute_work':10000000}}}}}
        values=[dict(copy.deepcopy(payload),rank=r) for r in range(4)]
        result=summarize_profiles(values)
        self.assertEqual(result['model_decode_rows_rank0'],32)
        self.assertEqual(result['observed_max_decode_batch'],32)
        aligned=result['ranks']['0']['aligned_recent_decode'][0]
        self.assertEqual(aligned['persistent_layers_ms'],2)
        self.assertEqual(aligned['outside_persistent_ms'],8)
        self.assertFalse(result['used_for_formal_performance'])
