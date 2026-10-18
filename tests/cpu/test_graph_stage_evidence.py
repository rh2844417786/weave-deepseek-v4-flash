import copy
import unittest
from src.weave.graph_stage_evidence import summarize_graph_stages


def snapshots(kind='weave'):
    before,after=[],[]
    for rank in range(4):
        base={'rank':rank,'successful_replays':2,'stage_profile':dict(schema_version=1,
            measurement_mode='instrumented_diagnostic',profile_source_sha256='a'*64,
            kind=kind,rank=rank,stride=128,capacity=64,layers_expected=43 if kind=='weave' else 0,
            registered_layers=list(range(43)) if kind=='weave' else [],external_events=True,
            numerical_policy='record_only',sampled_total=0,b32_seen=2,samples=[])}
        before.append(base);row=copy.deepcopy(base);row['successful_replays']=386
        profile=row['stage_profile'];profile.update(sampled_total=3,b32_seen=386)
        for seq in (128,256,384):
            compute,comm,final=(1.5,.1,.1) if rank==0 else (.5,1.,.1)
            layers=[dict(layer=i,expert_pipeline_ms=compute,slot_sum_ms=comm,finalize_ms=final,
                         routed_total_ms=compute+comm+final) for i in range(43)] if kind=='weave' else []
            total=sum(x['routed_total_ms'] for x in layers);whole=[100.,200.,150.,160.][rank]
            profile['samples'].append(dict(sequence=seq,b32_ordinal=seq,batch=32,whole_graph_ms=whole,
                layers=layers,routed_total_ms=total,residual_ms=whole-total))
        after.append(row)
    return before,after


class StageEvidenceTests(unittest.TestCase):
    def test_aligned_critical_rank_not_independent_stage_maxima(self):
        a,b=snapshots();r=summarize_graph_stages(a,b,kind='weave',profile_source='a'*64,stride=128,capacity=64)
        self.assertEqual(r['sample_count'],3)
        self.assertEqual(r['critical_rank_histogram'],{'1':3})
        self.assertAlmostEqual(r['critical_rank']['expert_pipeline_ms']['mean'],21.5)
        self.assertAlmostEqual(r['critical_rank']['slot_sum_ms']['mean'],43.)
        self.assertAlmostEqual(r['critical_rank']['whole_graph_ms']['mean'],200.)
        self.assertEqual(len(r['per_rank']['0']['layers']),43)
        self.assertFalse(r['formal_performance'])

    def test_reject_missing_layer_misaligned_sample_source_and_impossible_time(self):
        changes=[lambda p:p['samples'][0]['layers'].pop(),lambda p:p['samples'][0].update(sequence=127),
                 lambda p:p.update(profile_source_sha256='b'*64),lambda p:p['samples'][0].update(whole_graph_ms=1.),
                 lambda p:p['samples'][0]['layers'][0].update(slot_sum_ms=float('nan')),
                 lambda p:p['samples'][0]['layers'][0].update(finalize_ms=-1.),lambda p:p.update(samples=[])]
        for change in changes:
            a,b=snapshots();change(b[2]['stage_profile'])
            with self.subTest(change=change),self.assertRaises(ValueError):
                summarize_graph_stages(a,b,kind='weave',profile_source='a'*64,stride=128,capacity=64)

    def test_native_outer_only(self):
        a,b=snapshots('native');r=summarize_graph_stages(a,b,kind='native',profile_source='a'*64,stride=128,capacity=64)
        self.assertEqual(r['per_rank']['0']['layers'],{})
        self.assertEqual(r['critical_rank']['whole_graph_ms']['mean'],200.)

    def test_samples_before_measurement_are_excluded(self):
        a,b=snapshots()
        for row in a:row['successful_replays']=256
        r=summarize_graph_stages(a,b,kind='weave',profile_source='a'*64,stride=128,capacity=64)
        self.assertEqual(r['sample_count'],1)
        self.assertEqual(r['sequences'],[384])


class StageGateEvidenceTests(unittest.TestCase):
    def fixture(self):
        from tests.gpu.check_graph_stage_events import hashes
        h=hashes();identity={k:v for k,v in h.items() if k not in ('bf16_source_sha256','profile_source_sha256')}
        identity.update(source_sha256=h['bf16_source_sha256'],graph_stage_source_sha256=h['profile_source_sha256'],image_id='sha256:image')
        rows=[]
        for rank in range(4):
            sample=dict(sequence=1,b32_ordinal=1,batch=32,whole_graph_ms=4.,routed_total_ms=3.,residual_ms=1.,
                layers=[dict(layer=0,expert_pipeline_ms=1.,slot_sum_ms=1.,finalize_ms=1.,routed_total_ms=3.)])
            cases=[]
            for seq in range(1,5):
                timed=dict(sample,sequence=seq,b32_ordinal=seq)
                cases.append(dict(sequence=seq,batch=32,variant=seq,stage_sample=timed,finite=True,metadata_exact=True,
                    assembly_exact=True,stable_addresses=True,input_update_verified=True,fresh_events_verified=True,
                    device_collective_delta=1,numerical={k:dict(finite=True,max_abs=1e300,relative_l2=1e300) for k in
                    ('original_graph','original_eager','fp8_oracle','bf16_oracle','bf16_reference')}))
            rows.append(dict(h,schema_version=1,status='passed',gpu_checks_status='passed',rank=rank,world_size=4,
                measurement_mode='instrumented_diagnostic',numerical_policy='record_only',raw_content_exported=False,
                case_count=4,replay_count=4,graph_capture_count=2,external_events=True,replay_device_collective_delta=4,
                image_id='sha256:image',runtime=dict(image_id='sha256:image',capability=[9,0],device_count=4,device_name='H100'),
                shape=dict(batch=32,hidden=128,intermediate=128,experts=32,topk=6,layers=1),
                source_hashes_after=h,cases=cases,cleanup=dict(status='complete',phase='complete',graphs_owned=2,graphs_reset=2,
                    completed_phases=['synchronize','reset_graphs','synchronize_after_reset','rank_barrier','destroy_nccl','destroy_gloo'],
                    started_monotonic=1.,finished_monotonic=2.),
                stage_profile=dict(profile_source_sha256=h['profile_source_sha256'],rank=rank,kind='weave',stride=1,
                    layers_expected=1,sampled_total=4,b32_seen=4,aborted_samples=0,samples=[c['stage_sample'] for c in cases])))
        return identity,rows

    def test_smoke_source_coverage_freshness_and_record_only_errors(self):
        import json
        import tempfile
        from pathlib import Path
        from src.weave.graph_stage_evidence import validate_stage_event_smoke
        identity,rows=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            def write(values):
                for r,row in enumerate(values):(Path(tmp)/f'stage-event-rank{r}.json').write_text(json.dumps(row))
            write(rows);self.assertEqual(validate_stage_event_smoke(tmp,identity)['status'],'passed')
            for mutate in (lambda r:r['cleanup'].update(graphs_reset=1),lambda r:r['cases'].pop(),
                           lambda r:r['cases'][0].update(fresh_events_verified=False),
                           lambda r:r.update(profile_source_sha256='wrong'),
                           lambda r:r['cases'][0]['numerical']['original_graph'].pop('max_abs')):
                changed=copy.deepcopy(rows);mutate(changed[3]);write(changed)
                with self.assertRaises(ValueError):validate_stage_event_smoke(tmp,identity)
