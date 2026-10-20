from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.weave.native_p0_pair import compare_clients, paired_graph_check, normalize_launch


def clients():
    selection=dict(request_count=32,selection_sha256='a'*64,requests=[dict(original_row_index=i,prompt_sha256='b'*64,input_tokens=100,request_payload_sha256='c'*64) for i in range(32)])
    a=dict(status='passed',selection=selection,source_workload_identity={'record_sha256':'d'*64},generation_settings=dict(max_new_tokens=32768,temperature=0,ignore_eos=False),request_count=32,requests_completed=32,
           numerical_policy='record_only',cleanup_complete=True,error_codes=[],warmup={'requests_completed':32},post_warmup_flush={'status':'flushed'},rows=[])
    for i in range(32):a['rows'].append(dict(**selection['requests'][i],status='completed',output_tokens=2000,finish_reason='stop',output_text_sha256='e'*64,output_text_bytes=5000,prefix_digests=[dict(completion_tokens=1,sha256='f'*64)],elapsed_seconds=20.))
    b=deepcopy(a)
    for row in b['rows']:row.update(output_tokens=20,output_text_bytes=40,output_text_sha256='0'*64,prefix_digests=[dict(completion_tokens=1,sha256='1'*64)])
    return a,b


class PairEvidenceTests(unittest.TestCase):
    def test_large_content_and_length_differences_record_only(self):
        a,b=clients();r=compare_clients(a,b)
        self.assertEqual(r['status'],'paired_check_complete');self.assertEqual(r['output_text_digest_mismatched'],32)
        self.assertEqual(r['lengths']['native']['median'],2000);self.assertEqual(r['lengths']['p0']['median'],20)
        self.assertEqual(r['rows'][0]['first_observed_divergent_prefix_tokens'],1)
        self.assertFalse(r['formal_performance']);self.assertEqual(r['numerical_policy'],'record_only')

    def test_wrong_request_identity_or_missing_response_is_invalid(self):
        for mutation in (lambda b:b['rows'][0].update(request_payload_sha256='9'*64),lambda b:b['rows'].pop(),lambda b:b.update(cleanup_complete=False)):
            a,b=clients();mutation(b)
            with self.assertRaises(ValueError):compare_clients(a,b)

    def test_prefix_missing_not_imputed(self):
        a,b=clients();b['rows'][0]['prefix_digests']=[];r=compare_clients(a,b)
        self.assertIsNone(r['rows'][0]['first_observed_divergent_prefix_tokens'])
        self.assertEqual(r['rows'][0]['prefix_comparisons'][0]['status'],'unavailable_in_p0')

    def test_launch_only_p0_and_owned_paths_may_differ(self):
        a=['docker','run','--name','one','-e','HF_HUB_OFFLINE=1','image','python','check','--output-dir','/work/one','--tp-size','4']
        b=['docker','run','--name','two','-e','HF_HUB_OFFLINE=1','-e','WEAVE_NATIVE_P0=1','-e','WEAVE_NATIVE_P0_PRIVATE_DIR=/work/two/private','image','python','check','--output-dir','/work/two','--tp-size','4']
        self.assertEqual(normalize_launch(a),normalize_launch(b));b[-1]='2';self.assertNotEqual(normalize_launch(a),normalize_launch(b))

    def test_no_runtime_sample_is_observed_state_not_failure(self):
        from tests.cpu.test_graph_evidence import snapshots
        before,after=snapshots()
        for rows in (before,after):
            for row in rows:
                row['stage_profile']=dict(kind='native_p0',rank=row['rank'],profile_source_sha256='c'*64,external_events=True,numerical_policy='record_only',configuration=dict(stride=128,capacity=64,layers=43,captured_batch_size=32),b32_seen=row['raw_batch_histogram']['32'],sampled_total=0,aborted_samples=0,launch_trace=None,samples=[],layer_metadata={str(i):dict(captured_path='forward_normal_dual_stream',captured_spans=['moe','routed','gate_up','down'],marlin_calls=[{'role':'gate_up'},{'role':'down'}]) for i in range(43)})
        identity=dict(observer_source_sha256='b'*64,graph_sources={'p':'a'*64},native_p0_source_sha256='c'*64)
        result=paired_graph_check(before,after,p0_enabled=True,identity=identity)
        self.assertEqual(result['p0_runtime_sampled'],False)
        self.assertEqual(result['p0_execution_scope'],'captured_instrumentation_without_runtime_sampling')
        after[0]['stage_profile']['sampled_total']=1
        with self.assertRaises(ValueError):paired_graph_check(before,after,p0_enabled=True,identity=identity)

if __name__=='__main__':unittest.main()
