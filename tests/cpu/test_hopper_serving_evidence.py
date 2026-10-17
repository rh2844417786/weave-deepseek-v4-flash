import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.weave.hopper_serving_evidence import validate_replay_bundle, record_generation_comparison
from src.weave.c32_comparison import HOPPER_VARIANTS, summarize_comparison, validate_coverage_delta
from tests.cpu.test_c32_comparison import coverage

BUNDLE=Path(__file__).resolve().parents[2]/'reports/server/hopper-candidate-115da040aca7'


def current_identity():
    value=json.loads((BUNDLE/'identity.json').read_text())
    return {k:value[k] for k in ('candidate_source_sha256','bf16_source_sha256','checkpoint_content_sha256','image_id')} | {'tokenizer_manifest_sha256':value['capture_identity']['tokenizer_manifest_sha256']}


def generation():
    return {'records':[{'request_index':i,'request_id':f'weave-c32-acceptance-{i:04d}',
        'output_ids':list(range(8)),'output_tokens':8,'output_ids_sha256':hashlib.sha256(json.dumps(list(range(8)),separators=(',',':')).encode()).hexdigest()} for i in range(32)]}


class HopperServingEvidenceTest(unittest.TestCase):
    def test_complete_historical_bundle_retains_python_identity(self):
        proof=validate_replay_bundle(BUNDLE,**current_identity())
        self.assertEqual(proof['point_count'],1376)
        self.assertEqual(proof['manifest_file_count'],29)
        self.assertEqual(proof['numerical_policy'],'record_only')
        self.assertEqual(proof['replay_source_sha256'],json.loads((BUNDLE/'identity.json').read_text())['replay_source_sha256'])

    def test_bundle_rejects_current_identity_change(self):
        for key in current_identity():
            current=current_identity();current[key]='a'*64
            with self.subTest(key=key),self.assertRaises(ValueError):validate_replay_bundle(BUNDLE,**current)

    def test_bundle_rejects_tamper_missing_and_inconsistent_summary(self):
        for change in ('tamper','missing','summary','partial','extra','symlink'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                target=Path(tmp)/'proof';shutil.copytree(BUNDLE,target)
                path=target/'summary.json'
                if change=='tamper':path.write_text('{}')
                elif change=='missing':path.unlink()
                elif change=='extra':(target/'private.json').write_text('{}')
                elif change=='symlink':path.unlink();path.symlink_to(BUNDLE/'summary.json')
                else:
                    value=json.loads(path.read_text());value['status']='partial' if change=='partial' else 'complete';value['point_count']=1;path.write_text(json.dumps(value))
                    manifest=json.loads((target/'manifest.json').read_text());manifest['files']['summary.json']=hashlib.sha256(path.read_bytes()).hexdigest();(target/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):validate_replay_bundle(target,**current_identity())

    def test_generation_differences_recorded_and_no_ids_exported(self):
        candidate=generation();candidate['records'][7]['output_ids'][2]=1234
        candidate['records'][7]['output_ids_sha256']=hashlib.sha256(json.dumps(candidate['records'][7]['output_ids'],separators=(',',':')).encode()).hexdigest()
        result=record_generation_comparison(generation(),candidate)
        self.assertEqual(result['status'],'recorded')
        self.assertEqual(result['mismatched_requests'],1)
        self.assertEqual(result['matched_generated_tokens'],255)
        self.assertNotIn('output_ids',json.dumps(result));self.assertNotIn('1234',json.dumps(result))

    def test_invalid_generation_still_rejected(self):
        for change in (lambda x:x['records'].pop(),lambda x:x['records'][0].update(request_index=40),
                lambda x:x['records'][0].update(request_id='foreign'),lambda x:x['records'][0].update(output_ids=[-1]*8)):
            bad=generation();change(bad)
            with self.assertRaises(ValueError):record_generation_comparison(generation(),bad)

    def test_hopper_coverage_all43_identities_and_collectives(self):
        def rows(after):
            values=[coverage(r,'hopper_fp8',after) for r in range(4)]
            for value in values:
                value.update(candidate_source_sha256='c'*64,assembly_source_sha256='d'*64,numerical_policy='record_only',hopper_max_batch=32,hopper_slot_collective='torch.distributed.all_reduce/SUM/tp_device_group',additional_slot_collectives=86 if after else 0)
                for layer in value['layers'].values():layer['implementation_calls']['hopper_decode']=2 if after else 0
            return values
        before,after=rows(False),rows(True)
        result=validate_coverage_delta(before,after,'hopper_fp8','a'*64,'b'*64,candidate_source_sha256='c'*64,assembly_source_sha256='d'*64)
        self.assertEqual(result['ranks']['0']['additional_slot_collectives'],86)
        for mutation in (lambda x:x.update(assembly_source_sha256='e'*64),lambda x:x.update(additional_slot_collectives=85),lambda x:x['layers']['42']['implementation_calls'].update(hopper_decode=0),lambda x:x['layers']['0']['implementation_calls'].update(staged_decode=1)):
            bad=copy.deepcopy(after);mutation(bad[1])
            with self.assertRaises(ValueError):validate_coverage_delta(before,bad,'hopper_fp8','a'*64,'b'*64,candidate_source_sha256='c'*64,assembly_source_sha256='d'*64)

    def test_hopper_ratios_allow_recorded_differences(self):
        groups={v:{'status':'passed','identity':{'decode_graph':'disabled'},'client':{'status':'passed','decode_tokens_per_second':rate},'acceptance':{'status':'recorded'},'batch_evidence':{'max_running_requests':32}} for v,rate in zip(HOPPER_VARIANTS,(200.,40.,100.))}
        result=summarize_comparison(groups,comparison_suite='hopper')
        self.assertEqual(result['hopper_vs_native_ratio'],.5)
        self.assertEqual(result['hopper_vs_bf16_staged_ratio'],2.5)
        self.assertEqual(result['numerical_policy'],'record_only')

    def test_assembly_requires_actual_four_rank_source_and_case_proof(self):
        from src.weave.hopper_serving_evidence import validate_assembly_reports,ASSEMBLY_CASES
        hashes={'candidate_source_sha256':'a'*64,'bf16_source_sha256':'b'*64,'assembly_source_sha256':'c'*64}
        rows=[{'schema_version':1,'rank':r,'world_size':4,'status':'passed','numerical_policy':'record_only','stop_reason':'none','resources':{},**hashes,
            'source_hashes_after':hashes,'case_count':6,'additional_slot_collectives':6,
            'stable_addresses':True,'binding_lifetime':True,'empty_local_rank_exercised':True,'empty_origin_rank_exercised':True,
            'cases':[{'case':name,'batch':batch,'additional_slot_collectives':1,'assembly_exact':True,'finite':True,
                'stable_addresses':True,'binding_alive':True,'local_slots':0,'origin_rows':0} for name,batch in ASSEMBLY_CASES],
            'ordered_finalizer':{'status':'passed','case_count':4,'batches':[1,7,17,32],'exact':True},
            'runtime':{'image_id':'sha256:image','capability':[9,0],'device_count':4,'device_name':'NVIDIA H100'}} for r in range(4)]
        self.assertEqual(validate_assembly_reports(rows,**hashes,image_id='sha256:image')['ranks'],4)
        for mutate in (lambda r:r['cases'].pop(),lambda r:r.update(source_hashes_after={}),lambda r:r['ordered_finalizer'].update(exact=False),lambda r:r['cases'][0].update(finite=False),lambda r:r['runtime'].update(image_id='foreign')):
            bad=copy.deepcopy(rows);mutate(bad[2])
            with self.assertRaises(ValueError):validate_assembly_reports(bad,**hashes,image_id='sha256:image')

    def test_generation_summary_requires_complete_successful_response_provenance(self):
        from src.weave.hopper_serving_evidence import validate_generation_proof
        from src.weave.c32_comparison import digest
        proof=generation();proof.update(dataset_sha256='a'*64,request_order_sha256='b'*64,generation_settings={'max_new_tokens':8,'temperature':0,'ignore_eos':True})
        summary={k:v for k,v in proof.items() if k!='records'}
        summary.update(status='passed',private_artifact_sha256='c'*64,requests_completed=32,generated_tokens=256,
            concurrency=32,error_codes=[],abort_failures=0,acceptance_order_sha256=digest(list(range(32))),
            generated_ids_sha256=digest([[i,list(range(8))] for i in range(32)]))
        self.assertTrue(validate_generation_proof(summary,proof,private_sha256='c'*64,dataset_sha256='a'*64))
        for key,value in (('status','failed'),('private_artifact_sha256','d'*64),('abort_failures',1),('generated_ids_sha256','e'*64),('error_codes',['response_request_id_mismatch']),('request_order_sha256','foreign')):
            bad=copy.deepcopy(summary);bad[key]=value
            with self.assertRaises(ValueError):validate_generation_proof(bad,proof,private_sha256='c'*64,dataset_sha256='a'*64)
