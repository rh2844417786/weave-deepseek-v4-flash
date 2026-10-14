import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.weave.c32_comparison import (VARIANTS, compare_acceptance, summarize_comparison,
    validate_coverage_delta, scheduler_counts, validate_client_summary,validate_online_gate)


def proof():
    return {'records':[{'request_index':i,'request_id':f'accept-{i}',
                         'output_ids':list(range(8))} for i in range(32)]}


def coverage(rank,mode,after=False):
    return {'rank':rank,'source_sha256':'a'*64,'kernel':'tiled','graph_enabled':False,
        'hot_cache':True,'decode_implementation':mode,'fixed_plan':[1,1,0],
        'profile_checkpoint_content_sha256':'b'*64,'profile_bank_sha256':None,
        'execution_profile_configuration':None,'additional_slot_collectives':86 if after and mode=='staged_direct' else 0,
        'model_batch_counts':{'active':False,'failed_forwards':0,
            'decode_valid_batch_histogram':{'32':2} if after else {}},
        'layers':{str(layer):{'device_calls':[1,2 if after else 0,0],
            'implementation_calls':{'staged_decode':2 if after and mode=='staged_direct' else 0,
                'persistent_decode':2 if after and mode=='persistent' else 0,
                'persistent_prefill':1,'persistent_other':0}} for layer in range(43)}}


class C32ComparisonTest(unittest.TestCase):
    def test_online_gate_requires_all43_batch32_and_full_numeric_cases(self):
        cases=[(-1,v,b) for v,b in enumerate((1,7,17,32,7,1))]+[(x,x,32) for x in range(43)]
        rows=[{'status':'online_staged_checks_passed','rank':r,'source_sha256':'a'*64,
            'checkpoint_content_sha256':'b'*64,'layers':list(range(43)),
            'cases':[{'layer':layer,'variant':v,'batch':b,'metadata':'exact_p1_match',
                'assembly':'exact_fp32_slots_then_ordered_sum_bf16_origin_mask',
                **{k:{'relative_l2':0.,'max_abs':0.} for k in ('local_slots','reference','persistent')}}
                for layer,v,b in cases]} for r in range(4)]
        self.assertEqual(validate_online_gate(rows,'a'*64,'b'*64)['cases_per_rank'],49)
        bad=copy.deepcopy(rows);bad[2]['cases'][-1]['batch']=1
        with self.assertRaises(ValueError):validate_online_gate(bad,'a'*64,'b'*64)
        bad=copy.deepcopy(rows);bad[1]['cases'][5]['persistent']['relative_l2']=.03
        with self.assertRaises(ValueError):validate_online_gate(bad,'a'*64,'b'*64)

    def test_client_rate_includes_inflight_and_uses_declared_window(self):
        value={'status':'passed','dataset_sha256':'e'*64,'request_order_sha256':'f'*64,
            'generation_settings':{'max_new_tokens':1024,'temperature':0,'ignore_eos':True},
            'concurrency':32,'peak_client_inflight':32,'requested_seconds':600.,'window_seconds':600.,
            'full_window_observed':True,'requests_started':32,'requests_completed':0,
            'requests_failed':0,'requests_unfinished':32,'output_tokens_in_window':32000,
            'decode_tokens_in_window':31968,'decode_tokens_per_second':31968/600,
            'error_codes':[],'abort_failures':0}
        validate_client_summary(value,600.,'e'*64)
        value['decode_tokens_per_second']=0.
        with self.assertRaises(ValueError):validate_client_summary(value,600.,'e'*64)

    def test_acceptance_matches_actual_sequences_and_never_exports_tokens(self):
        result=compare_acceptance(proof(),proof())
        self.assertEqual(result['matched_generated_tokens'],256)
        self.assertNotIn('output_ids',json.dumps(result))
        bad=proof();bad['records'][18]['output_ids'][4]=999
        with self.assertRaises(ValueError):compare_acceptance(proof(),bad)
        bad=proof();bad['records'].pop()
        with self.assertRaises(ValueError):compare_acceptance(proof(),bad)

    def test_actual_all43_staged_decode_and_one_collective_per_layer_required(self):
        before=[coverage(r,'staged_direct') for r in range(4)]
        after=[coverage(r,'staged_direct',True) for r in range(4)]
        result=validate_coverage_delta(before,after,'staged_direct','a'*64,'b'*64)
        self.assertEqual(result['decode_batch32_forwards_rank0'],2)
        after[2]['layers']['42']['implementation_calls']['staged_decode']=0
        with self.assertRaises(ValueError):validate_coverage_delta(before,after,'staged_direct','a'*64,'b'*64)

    def test_eager_identity_and_completed_measurement_required_for_ratio(self):
        identity={'dataset_sha256':'c'*64,'image_id':'d'*64,'source_sha256':'a'*64,
                  'seconds':600,'decode_graph':'disabled'}
        groups={v:{'status':'passed','identity':dict(identity),
            'client':{'status':'passed','decode_tokens_per_second':t},
            'acceptance':{'status':'passed'},'batch_evidence':{'max_running_requests':32}}
            for v,t in zip(VARIANTS,(2000.,60.,100.))}
        result=summarize_comparison(groups)
        self.assertAlmostEqual(result['staged_vs_persistent_ratio'],100/60)
        self.assertAlmostEqual(result['staged_vs_native_ratio'],.05)
        groups['native']['identity']['decode_graph']='full'
        with self.assertRaises(ValueError):summarize_comparison(groups)
        groups['native']['identity']['decode_graph']='disabled'
        groups['staged_direct']['client']['status']='failed'
        with self.assertRaises(ValueError):summarize_comparison(groups)

    def test_scheduler_evidence_only_parses_decode_counts(self):
        value=scheduler_counts('Prefill batch #running-req:32\nDecode batch, #running-req: 32, throughput:2000\nDecode batch, #running-req: 17\nprivate prompt')
        self.assertEqual(value['histogram'],{'17':1,'32':1})
        self.assertNotIn('private',json.dumps(value))
        self.assertNotIn('2000',json.dumps(value))
