import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.weave.hopper_persistent import IMPLEMENTATION,MECHANISM_FIELDS


def fixture(root):
    identity={'image_id':'image','checkpoint_content_sha256':'checkpoint'}
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    write('paper-source-manifest.json',{'cuda':'source'})
    write('serving-source-manifest.json',{'python':'source'})
    candidate=[[c,k,100000.,100000.,1.,0.] for c in (1,2) for k in (1,2)]
    numeric={name:{'finite':True,'relative_l2':1e100,'max_abs':1e100} for name in ('original_hopper','fp8_oracle','bf16_oracle')}
    for stage in ('smoke','calibrate','checkpoint'):
        write(stage+'/container-lifecycle.json',{'exit_code':0,'termination_signal':None})
        write(stage+'/runtime.json',{'runtime_status':'passed','resolved_image_id':'image'})
    for rank in range(4):
        bank=dict(schema_version=1,status='measured',implementation=IMPLEMENTATION,numerical_policy='record_only',
            source_sha256='source',device_key='gpu',hidden_size=4096,intermediate_size=2048,experts=256,top_k=6,
            rank=rank,image_id='image',checkpoint_content_sha256='checkpoint',runtime={},
            model_constants={'efficiency':1.,'overlap':0.},profiles=[{'max_local_tokens':2048,'candidates':candidate}])
        digest=write(f'calibrate/paper-profile-bank-rank{rank}.json',bank)
        serving_sha=hashlib.sha256(json.dumps({'python':'source'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        base=dict(serving_source_sha256=serving_sha,serving_source_sha256_after=serving_sha,schema_version=1,status='passed',gpu_checks_status='passed',implementation=IMPLEMENTATION,
            numerical_policy='record_only',raw_content_exported=False,rank=rank,world_size=4,source_sha256='source',
            source_sha256_after='source',image_id='image',checkpoint_content_sha256='checkpoint',
            resources={'implementation':IMPLEMENTATION,'active_blocks_per_sm':1,'grid_blocks':4,'threads_per_block':128},
            cleanup={'status':'complete','graphs_owned':0,'graphs_reset':0},profile_bank_sha256=digest)
        mech={k:0 for k in MECHANISM_FIELDS};mech.update(calls=1,dynamic_calls=1,computed_tiles=2,stolen_tiles=1,
            return_compute_overlap=1,dispatch_compute_overlap=1,k_sum=2,profile_candidates=4)
        case={'mechanisms':mech,'unique_worker_smids':True,'routed_collectives':0}
        smoke=dict(base,shape=dict(hidden=128,intermediate=128,experts=32,capacity=64),cases=[dict(copy.deepcopy(case),case=name) for name in ('serial','dispatch_pipeline','pipeline','stealing','dynamic','dynamic_skew','graph_3','graph_4')])
        smoke_bank=copy.deepcopy(bank);smoke_bank.update(hidden_size=128,intermediate_size=128,experts=32)
        smoke_bank['profiles'][0]['max_local_tokens']=64
        smoke['profile_bank_sha256']=write(f'smoke/paper-profile-bank-rank{rank}.json',smoke_bank)
        smoke['calibration']=[dict(max_local_tokens=64,c=r[0],k=r[1],compute_ms=1.,communication_ms=1.,padded_flops=100,communication_bytes=100,candidate=r) for r in candidate]
        write(f'smoke/paper-smoke-rank{rank}.json',smoke)
        base['shape']=dict(hidden=4096,intermediate=2048,experts=256,capacity=2048)
        calibrated=dict(base,cases=[],calibration=[dict(max_local_tokens=2048,c=r[0],k=r[1],compute_ms=1.,communication_ms=1.,padded_flops=100,communication_bytes=100,candidate=r) for r in candidate])
        write(f'calibrate/paper-calibrate-rank{rank}.json',calibrated)
        cases=[dict(copy.deepcopy(case),case='checkpoint',layer=l,phase=p,batch=b,numerical=numeric,numerical_coverage={'original_hopper':{'rows':b,'scope':'full_global_batch'},'fp8_oracle':{'rows':2,'global_rows':[0,1]},'bf16_oracle':{'rows':2,'global_rows':[0,1]}}) for l,p,b in [(l,'decode',32) for l in range(43)]+[(0,'prefill',128)]]
        write(f'checkpoint/paper-checkpoint-rank{rank}.json',dict(base,cases=cases))
    return identity


class PaperProofTests(unittest.TestCase):
    def test_reject_foreign_checkpoint_profile_and_python_source(self):
        from src.weave.paper_evidence import validate_paper_bundle
        with tempfile.TemporaryDirectory() as tmp,patch('src.weave.paper_evidence.source_sha256',return_value='source'),patch('src.weave.paper_evidence.source_manifest',return_value={'cuda':'source'}):
            root=Path(tmp);identity=fixture(root)
            with patch('scripts.run_c32_comparison.serving_source_manifest',return_value={'python':'source'}):
                self.assertEqual(validate_paper_bundle(root,identity)['status'],'passed')
                path=root/'checkpoint/paper-checkpoint-rank2.json';row=json.loads(path.read_text());row['profile_bank_sha256']='foreign';path.write_text(json.dumps(row))
                with self.assertRaisesRegex(ValueError,'checkpoint.*profile'):validate_paper_bundle(root,identity)
            fixture(root)
            with patch('scripts.run_c32_comparison.serving_source_manifest',return_value={'python':'changed'}):
                with self.assertRaisesRegex(ValueError,'Python|serving'):validate_paper_bundle(root,identity)

    def test_controlled_mechanisms_and_oracle_scope_cannot_be_omitted(self):
        from src.weave.paper_evidence import validate_paper_bundle
        with tempfile.TemporaryDirectory() as tmp,patch('src.weave.paper_evidence.source_sha256',return_value='source'),patch('src.weave.paper_evidence.source_manifest',return_value={'cuda':'source'}),patch('scripts.run_c32_comparison.serving_source_manifest',return_value={'python':'source'}):
            root=Path(tmp)
            changes=[('smoke',lambda r:r['cases'][1]['mechanisms'].update(dispatch_compute_overlap=0)),
                     ('smoke',lambda r:r['cases'][2]['mechanisms'].update(k_sum=1)),
                     ('smoke',lambda r:r['cases'][2]['mechanisms'].update(return_compute_overlap=0)),
                     ('smoke',lambda r:r['cases'][3]['mechanisms'].update(stolen_tiles=0)),
                     ('checkpoint',lambda r:r['cases'][0].pop('numerical_coverage'))]
            for stage,change in changes:
                identity=fixture(root);path=root/stage/f'paper-{stage}-rank0.json'
                row=json.loads(path.read_text());change(row);path.write_text(json.dumps(row))
                with self.subTest(stage=stage),self.assertRaises(ValueError):validate_paper_bundle(root,identity)

    def test_service_requires_all43_dynamic_device_calls_without_fallback(self):
        from src.weave.paper_evidence import validate_paper_coverage
        identity={'checkpoint_content_sha256':'checkpoint'}
        proof={'paper_source_sha256':'source','profile_sha256':{str(r):'profile' for r in range(4)}}
        before=[];after=[]
        for rank in range(4):
            row=dict(rank=rank,method_identity=IMPLEMENTATION,source_sha256='source',profile_bank_sha256='profile',
                schedule_mode='measured_dynamic',fixed_plan=None,decode_implementation='hopper_persistent',
                prefill_implementation='hopper_persistent',graph_enabled=True,graph_mode='full',kernel='tiled',hot_cache=True,
                numerical_policy='record_only',additional_slot_collectives=0,additional_prefill_slot_collectives=0,
                profile_checkpoint_content_sha256='checkpoint',model_batch_counts={'active':False},paper_resources={'active_blocks_per_sm':1,'grid_blocks':4},
                layers={str(l):dict(device_calls=[0,0,0],capture_calls=32,python_decode_calls=64,implementation_calls={'paper_persistent_prefill':0,'paper_persistent_decode':64,'persistent_decode':0}) for l in range(43)},
                mechanism_evidence={'fields':list(MECHANISM_FIELDS),'worker_smids':[0,1,2,3],'phases':{phase:{str(l):{'counters':{k:0 for k in MECHANISM_FIELDS},'plan_histogram':[]} for l in range(43)} for phase in ('prefill','decode')}})
            before.append(row);new=copy.deepcopy(row)
            for layer in new['layers'].values():layer['device_calls']=[1,3,0];layer['implementation_calls']['paper_persistent_prefill']=1
            for phase,n in [('prefill',1),('decode',3)]:
                for layer in new['mechanism_evidence']['phases'][phase].values():
                    layer['counters'].update(calls=n,dynamic_calls=n,profile_candidates=4*n,computed_tiles=2*n)
                    layer['plan_histogram']=[{'c':1,'k':1,'calls':n}]
            after.append(new)
        result=validate_paper_coverage(before,after,{'replays':3},identity,proof)
        self.assertFalse(result['complete_weave_claim']);self.assertFalse(result['ranks']['0']['stealing_observed'])
        changes=[lambda r:r['layers']['42']['device_calls'].__setitem__(1,2),lambda r:r.update(additional_slot_collectives=1),
                 lambda r:r['mechanism_evidence']['phases']['decode']['42']['counters'].update(dynamic_calls=0),
                 lambda r:r['layers']['42']['implementation_calls'].update(persistent_decode=1),
                 lambda r:r.update(profile_bank_sha256='wrong')]
        for change in changes:
            bad=copy.deepcopy(after);change(bad[2])
            with self.assertRaises(ValueError):validate_paper_coverage(before,bad,{'replays':3},identity,proof)
