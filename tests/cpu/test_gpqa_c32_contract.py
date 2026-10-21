import json
from pathlib import Path
import shutil
import tempfile
import unittest
from src.weave.gpqa_c32 import load_gpqa_workload, prefill_bundle_bytes
ROOT=Path(__file__).resolve().parents[2]
class ContractTests(unittest.TestCase):
    def test_pinned_workload(self):
        value=load_gpqa_workload(ROOT/'data/gpqa/prepared/deepseek-v4-flash/main.jsonl')
        self.assertEqual(len(value.records),448)
        self.assertEqual(value.identity['max_new_tokens'],32768)
    def test_mutated_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            for name in ('main.jsonl','manifest.json'): shutil.copy(ROOT/'data/gpqa/prepared/deepseek-v4-flash'/name,folder/name)
            manifest=json.loads((folder/'manifest.json').read_text());manifest['normal_eos_allowed']=False
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_gpqa_workload(folder/'main.jsonl')
    def test_exact_proof_inventory(self):
        source=ROOT/'reports/server/c32-prefill-63dac24bf44a'
        self.assertEqual(len(prefill_bundle_bytes(source)),99)
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/'proof';shutil.copytree(source,destination)
            (destination/'extra.json').write_text('{}')
            with self.assertRaises(ValueError):prefill_bundle_bytes(destination)

class SummaryTests(unittest.TestCase):
    def summary(self, exhausted=False):
        return dict(schema_version=1, mode='gpqa_measurement', status='passed', concurrency=32, requested_seconds=1200,
          window_seconds=1200, peak_client_inflight=32, started_monotonic=1.,deadline_monotonic=1201.,generation_settings={'max_new_tokens':32768,'temperature':0,'ignore_eos':False},
          workload={'record_sha256':'cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888'},
          numerical_policy='record_only', accuracy_evaluated=False,raw_content_exported=False,
          requests_started=32, requests_completed=2,requests_unfinished=30,requests_failed=0,requests_unattempted=416,
          output_tokens_in_window=320,decode_tokens_in_window=288,decode_tokens_per_second=.24,
          refill_exhausted=False,refill_exhausted_monotonic=None,completed_row_indices=[1,2],last_completed_monotonic=100.,dataset_exhausted_monotonic=None,idle_seconds=0.,dataset_exhausted=exhausted,reached_deadline=True,cleanup_complete=True,error_codes=[])
    def test_incomplete_counted_and_exhaustion_suppresses_ratio(self):
        from src.weave.gpqa_c32 import compare_groups
        a=self.summary();b=self.summary();b['decode_tokens_in_window']=576;b['output_tokens_in_window']=608;b['decode_tokens_per_second']=.48
        result=compare_groups({'native':a,'hopper_prefill':b})
        self.assertEqual(result['prefill_vs_native_ratio'],2)
        b.update(dataset_exhausted=True,requests_started=448,requests_completed=448,requests_unfinished=0,requests_unattempted=0,completed_row_indices=list(range(448)),dataset_exhausted_monotonic=100.,idle_seconds=1101.)
        self.assertIsNone(compare_groups({'native':a,'hopper_prefill':b})['prefill_vs_native_ratio'])
    def test_wrong_eos_or_private_output_rejected(self):
        from src.weave.gpqa_c32 import validate_client_summary
        value=self.summary();value['generation_settings']['ignore_eos']=True
        with self.assertRaises(ValueError):validate_client_summary(value)
        value=self.summary();value['response']='private'
        with self.assertRaises(ValueError):validate_client_summary(value)

class ProofTests(unittest.TestCase):
    def test_real_compute_bundle_valid_with_live_checkpoint_gate(self):
        # Validate this immutable archived bundle in its own source context.
        # New independent CUDA headers legitimately change the current digest.
        from unittest.mock import patch
        from src.weave.gpqa_c32 import validate_prefill_bundle
        root=ROOT/'reports/server/c32-prefill-63dac24bf44a'
        report=json.loads((root/'experiment.json').read_text())
        identity=report['identity'];gate=report['core_gate']
        with patch('scripts.calibrate_weave.validate_core_evidence',return_value=(gate,{})), patch('src.weave.replay_trace.tokenizer_manifest_sha256',return_value=identity['tokenizer_manifest_sha256']), patch('src.weave.ops.kernel_source_sha256',return_value=identity['source_sha256']):
            result=validate_prefill_bundle(root,Path('/missing-on-cpu'),identity['image_id'])
        self.assertEqual(result['manifest_file_count'],98)
        with patch('scripts.calibrate_weave.validate_core_evidence',return_value=(gate,{})), patch('src.weave.replay_trace.tokenizer_manifest_sha256',return_value=identity['tokenizer_manifest_sha256']), patch('src.weave.ops.kernel_source_sha256',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'identity differs'):
                validate_prefill_bundle(root,Path('/missing-on-cpu'),identity['image_id'])
    def test_mutated_records_rehashed_manifest_still_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);source=ROOT/'data/gpqa/prepared/deepseek-v4-flash'
            raw=(source/'main.jsonl').read_bytes()+b'\n';(folder/'main.jsonl').write_bytes(raw)
            manifest=json.loads((source/'manifest.json').read_text());manifest['record_sha256']=hashlib.sha256(raw).hexdigest()
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'pinned original'):load_gpqa_workload(folder/'main.jsonl')
    def test_bundle_symlink_rejected_and_bytes_preserved(self):
        from src.weave.gpqa_c32 import copy_verified_prefill_bundle
        source=ROOT/'reports/server/c32-prefill-63dac24bf44a'
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/'copy';copy_verified_prefill_bundle(source,destination)
            self.assertEqual(prefill_bundle_bytes(source),prefill_bundle_bytes(destination))
            target=destination/'comparison.json';target.unlink();target.symlink_to(source/'comparison.json')
            with self.assertRaises(ValueError):prefill_bundle_bytes(destination)

class ManifestPinTests(unittest.TestCase):
    def test_required_generation_and_source_pins(self):
        source=ROOT/'data/gpqa/prepared/deepseek-v4-flash'
        for key,bad in [('gpqa_main_sha256','0'*64),('model_revision','other'),('order_seed',43),
                        ('reasoning_effort','low'),('thinking_mode','none'),('max_new_tokens',1024),('count',447)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as folder:
                folder=Path(folder);shutil.copy(source/'main.jsonl',folder/'main.jsonl')
                manifest=json.loads((source/'manifest.json').read_text());manifest[key]=bad
                (folder/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):load_gpqa_workload(folder/'main.jsonl')
    def test_observed_config_difference_allowed_and_checked(self):
        import hashlib
        source=ROOT/'data/gpqa/prepared/deepseek-v4-flash'
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);shutil.copy(source/'main.jsonl',folder/'main.jsonl')
            manifest=json.loads((source/'manifest.json').read_text())
            manifest['observed_model_metadata_sha256']['config.json']=hashlib.sha256(b'local config').hexdigest()
            manifest['config_matches_pinned']=False;(folder/'manifest.json').write_text(json.dumps(manifest))
            self.assertFalse(load_gpqa_workload(folder/'main.jsonl').identity['config_matches_pinned'])
            with self.assertRaisesRegex(ValueError,'actual readonly'):load_gpqa_workload(folder/'main.jsonl',folder)

class AggregateIdentityTests(SummaryTests):
    def test_workload_group_identity_and_peak(self):
        from src.weave.gpqa_c32 import compare_groups, validate_client_summary
        a=self.summary();b=self.summary();b['workload']['extra']='changed'
        with self.assertRaises(ValueError):compare_groups({'native':a,'hopper_prefill':b})
        with self.assertRaises(ValueError):compare_groups({'native':{'client_summary':a,'identity':{'image':'a'}},'hopper_prefill':{'client_summary':a,'identity':{'image':'b'}}})
        a=self.summary();a['peak_client_inflight']=31
        with self.assertRaises(ValueError):validate_client_summary(a)
    def test_extra_manifest_private_field_rejected(self):
        source=ROOT/'data/gpqa/prepared/deepseek-v4-flash'
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);shutil.copy(source/'main.jsonl',folder/'main.jsonl')
            manifest=json.loads((source/'manifest.json').read_text());manifest['question']='secret'
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_gpqa_workload(folder/'main.jsonl')

class ReviewRegressionTests(SummaryTests):
    def test_outer_manifest_private_payload_is_rejected(self):
        from src.weave.gpqa_c32 import copy_verified_prefill_bundle
        source=ROOT/'reports/server/c32-prefill-63dac24bf44a'
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/'proof';shutil.copytree(source,destination)
            manifest=json.loads((destination/'manifest.json').read_text());manifest['prompt']='secret raw prompt'
            (destination/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):prefill_bundle_bytes(destination)
            public=Path(folder)/'public'
            with self.assertRaises(ValueError):copy_verified_prefill_bundle(destination,public)
            self.assertFalse(public.exists())
    def test_exhaustion_both_directions_and_missing_timestamps(self):
        from src.weave.gpqa_c32 import validate_client_summary
        value=self.summary();value['dataset_exhausted']=True
        with self.assertRaises(ValueError):validate_client_summary(value)
        value=self.summary();value.update(requests_started=448,requests_completed=448,requests_unfinished=0,requests_unattempted=0,completed_row_indices=list(range(448)))
        with self.assertRaises(ValueError):validate_client_summary(value)
        for field in ('last_completed_monotonic','dataset_exhausted_monotonic','idle_seconds'):
            value=self.summary();del value[field]
            with self.subTest(field=field),self.assertRaises(ValueError):validate_client_summary(value)

class RefillContractTests(SummaryTests):
    def test_refill_exhaustion_suppresses_ratio_with_valid_tail(self):
        from src.weave.gpqa_c32 import compare_groups
        value=self.summary()
        value.update(requests_started=448, requests_completed=417, requests_unfinished=31,
                     requests_unattempted=0, completed_row_indices=list(range(417)),
                     refill_exhausted=True, refill_exhausted_monotonic=101.)
        result=compare_groups({'native':self.summary(),'hopper_prefill':value})
        self.assertIsNone(result['prefill_vs_native_ratio'])
        self.assertEqual(result['ratio_suppression_reason'],'refill_exhausted_before_deadline')
        self.assertEqual(result['decode_tokens_per_second']['hopper_prefill'],.24)
        self.assertFalse(result['dataset_exhausted'])

    def test_last_dispatch_with_32_active_preserves_ratio(self):
        from src.weave.gpqa_c32 import compare_groups
        value=self.summary()
        value.update(requests_started=448, requests_completed=416, requests_unfinished=32,
                     requests_unattempted=0, completed_row_indices=list(range(416)))
        result=compare_groups({'native':self.summary(),'hopper_prefill':value})
        self.assertFalse(result['refill_exhausted'])
        self.assertTrue(result['sustained_c32_ratio_available'])
        self.assertEqual(result['prefill_vs_native_ratio'],1.)
        self.assertIsNone(result['ratio_suppression_reason'])

    def test_refill_evidence_required_and_coherent(self):
        from src.weave.gpqa_c32 import validate_client_summary
        for field in ('refill_exhausted','refill_exhausted_monotonic'):
            value=self.summary();del value[field]
            with self.subTest(field=field),self.assertRaises(ValueError):validate_client_summary(value)
        for updates in ({'refill_exhausted':True,'refill_exhausted_monotonic':101.},
                        {'refill_exhausted_monotonic':101.},
                        {'refill_exhausted':True,'refill_exhausted_monotonic':1201.,
                         'requests_started':448,'requests_completed':417,'requests_unfinished':31,
                         'requests_unattempted':0,'completed_row_indices':list(range(417))}):
            value=self.summary();value.update(updates)
            with self.subTest(updates=updates),self.assertRaises(ValueError):validate_client_summary(value)
