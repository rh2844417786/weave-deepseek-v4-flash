import copy
import importlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import unittest


def surface():
    # Test-only synthetic values: two anchors bracket 24 local rows.
    features = [[0, 0, 0, 0, 0], [48, 16, 16, 144, 3],
                [90, 30, 30, 270, 3], [120, 40, 40, 360, 3],
                [150, 50, 50, 450, 3], [192, 64, 64, 576, 3]]
    return [[f + [100 + index * 10 + anchor * 20, 2, 3]
             for anchor, f in enumerate(features)] for index in range(9)]


def measured_entry(batch):
    rows=surface_for_batch(batch)
    def stats(value):
        return dict(samples_us=[value]*2,median_us=value,mean_us=value,stddev_us=0.,min_us=value,max_us=value)
    def route(index):
        return dict(index=index,seed=100+index,activation_pattern='repeated_row' if index in (4,5,7) else 'spread',
                    features=rows[0][0][:5],total_rows=batch*6,global_blocks=1,
                    native_routing_sha256=f'{index+1:064x}',input_sha256=f'{index+11:064x}',input_ids_sha256=f'{index+21:064x}')
    def candidates(stages):
        return [dict(pair=list(pair),full_moe=stats(100+i),fixed_planner=stats(2),search_planner=stats(3),
                     **({'stages':{role:stats(1) for role in ('gate_up','down','shared','all_reduce')}} if stages else {}))
                for i,pair in enumerate(((8,8),(8,4),(8,2),(4,8),(4,4),(4,2),(2,8),(2,4),(2,2)))]
    anchors=[dict(route(index),candidates=candidates(True)) for index in range(6)]
    heldout=[dict(route(index),fixed_candidates=candidates(False),cost_full_moe=stats(101),
                  prediction=dict(predicted_us=list(range(101,110)),actual_us=list(range(101,110)),selected_index=0,
                                  absolute_error_us=[0]*9,relative_error=[0]*9,selected_actual_us=101,best_actual_us=101,
                                  regret_us=0,regret_fraction=0,numerical_policy='record_only'),
                  native_numerical=dict(finite=True,relative_l2=1e6,max_abs=1e9,numerical_policy='record_only'),
                  selected_pair=[8,8],cpu_selected_index=0,cpu_gpu_selection_agree=True,shares_local_route_with_training=False)
             for index in (6,7)]
    from src.weave.native_cost import workload_difference
    def workload(record):
        return {key:copy.deepcopy(record[key]) for key in
                ('features','total_rows','global_blocks','native_routing_sha256')}
    for anchor in anchors:
        for candidate in anchor['candidates']:
            candidate.update(workload=workload(anchor),workload_difference=workload_difference(anchor,anchor))
        anchor.update(search_workload=workload(anchor),search_workload_difference=workload_difference(anchor,anchor))
    for case in heldout:
        for candidate in case['fixed_candidates']:
            candidate.update(workload=workload(case),workload_difference=workload_difference(case,case))
        case['cost_workload_difference']=workload_difference(case,case)
    geometry=[dict(role=role,size_m=batch*(6 if i else 1),top_k=1 if i else 6,
                   size_n=4096,size_k=2048 if i else 4096,moe_block_size=8,is_ep=True,
                   native_grid=528,n_tiles=32,k_tiles=16 if i else 32)
              for i,role in enumerate(('gate_up','down'))]
    return dict(rows=rows,measurements=dict(complete=True,units='microseconds',repeats=2,geometry=geometry,
        objective='unmodified_native_MLP_routing_shared_join_collective_post_add',objective_graph_component_events=False,routing_comparison_policy='record_only',
        event_overhead_us=dict(samples_us=[0,1],median_us=.5),planner_calls_per_timed_graph=32,
        pattern_protocol='six_native_routed_synthetic_v1_four_spread_two_repeated',anchors=anchors),
        validation=dict(complete=True,numerical_policy='record_only',routing_comparison_policy='record_only',heldout=heldout))


def profile_fixture(cost):
    identity = dict(cost_source_sha256='a'*64, upstream_sha256={'fixture':'b'*64},
                    image_id='sha256:'+'c'*64, checkpoint_content_sha256='d'*64,
                    model_config_sha256='e'*64, sglang_commit='f'*40, rank=0,
                    gpu=dict(name='test H100',capability=[9,0],sm_count=132,total_memory=80000000000),
                    runtime=dict(python='3.12',torch='test',cuda='12',sglang='test',sgl_kernel='test'),
                    precision=dict(activations='bfloat16',routed_weights='mxfp4_e8m0',shared_weights='fp8'),
                    model_geometry=dict(hidden_size=4096,intermediate=2048,experts=256,top_k=6,layers=43),
                    graph_contract=dict(tp=4,ep=4,moe_tp=1,a2a_backend='none',decode_batches=list(range(1,33))))
    profile = dict(schema_version=1,method=cost.METHOD,policy_version=cost.POLICY_VERSION,
                   numerical_policy='record_only',identity=identity,calibration_complete=True,
                   groups={'g0':dict(representative_layer=0,layers=list(range(43)),signature={'topk':'test'})},
                   tables={'g0':{str(b):measured_entry(b) for b in range(1,33)}})
    profile['profile_sha256'] = cost.profile_digest(profile)
    return profile


def surface_for_batch(batch):
    n = batch * 6
    return [[[n, 1, 1, n*n, n, 100+i, 2, 3] for _ in range(6)] for i in range(9)]


def set_large_heldout_errors(case):
    """Test-only self-consistent poor prediction, with no quality acceptance gate."""
    actual=[1000000001.]+list(range(102,110))
    for candidate in case['fixed_candidates'][:1]:
        candidate['full_moe']=dict(samples_us=[1000000000.]*2,median_us=1000000000.,
            mean_us=1000000000.,stddev_us=0.,min_us=1000000000.,max_us=1000000000.)
    case['prediction'].update(actual_us=actual,absolute_error_us=[999999900.]+[0.]*8,
        relative_error=[999999900./1000000001.]+[0.]*8,selected_actual_us=1000000001.,
        best_actual_us=102.,regret_us=999999899.,regret_fraction=999999899./102.)
    return case


class CostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cost = importlib.import_module('src.weave.native_cost')
        except ImportError:
            cls.cost = None

    def test_cost_implementation_exists(self):
        self.assertIsNotNone(self.cost, 'native measured-cost policy is not implemented')

    def setUp(self):
        if self.cost is None and self._testMethodName != 'test_cost_implementation_exists':
            self.skipTest('policy missing')

    def test_prediction_interpolates_complete_moe_and_planner_replacement(self):
        answer = self.cost.predict_cost(surface(), [24,8,8,72,1.5], 192)
        self.assertEqual(answer['candidate_index'], 0)
        self.assertEqual(tuple(answer['pair']), (8,8))
        self.assertAlmostEqual(answer['predicted_us'], 111.)

    def test_each_candidate_can_have_its_own_measured_workload_features(self):
        rows = surface_for_batch(32)
        rows[1][0][:5] = [48, 16, 16, 144, 3]
        self.assertIs(self.cost.validate_table(rows, 192), rows)
        self.assertEqual(self.cost.predict_cost(rows, [48,16,16,144,3], 192)['candidate_index'], 0)

    def test_cross_candidate_route_differences_do_not_reject_complete_profiles(self):
        for change in ('features', 'digest', 'both'):
            profile = profile_fixture(self.cost)
            entry = profile['tables']['g0']['32']; anchor = entry['measurements']['anchors'][2]
            candidate = anchor['candidates'][1]
            observed = {key: copy.deepcopy(anchor[key]) for key in
                        ('features','total_rows','global_blocks','native_routing_sha256')}
            if change in ('features', 'both'): observed['features'] = [48,16,16,144,3]
            if change in ('digest', 'both'): observed['native_routing_sha256'] = 'f'*64
            candidate['workload'] = observed
            candidate['workload_difference'] = self.cost.workload_difference(anchor, observed)
            entry['rows'][1][2][:5] = observed['features']
            profile['profile_sha256'] = self.cost.profile_digest(profile)
            with self.subTest(change=change):
                self.assertIs(self.cost.validate_profile(profile), profile)
                self.assertEqual(candidate['workload_difference']['policy'], 'record_only')

    def test_heldout_cost_replay_can_differ_from_fixed_candidate_workloads(self):
        profile = profile_fixture(self.cost)
        case = profile['tables']['g0']['32']['validation']['heldout'][0]
        reference = case['fixed_candidates'][0]['workload']
        observed = case['fixed_candidates'][-1]['workload']
        observed.update(features=[48,16,16,144,3], native_routing_sha256='d'*64)
        case['fixed_candidates'][-1]['workload_difference'] = self.cost.workload_difference(reference, observed)
        case.update(features=[90,30,30,270,3], native_routing_sha256='e'*64)
        case['cost_workload_difference'] = self.cost.workload_difference(reference, case)
        from src.weave.native_cost_calibration import prediction_record
        prediction = self.cost.predict_cost(profile['tables']['g0']['32']['rows'],case['features'],192)
        case['prediction'] = prediction_record(prediction['predictions_us'],list(range(101,110)),prediction['candidate_index'])
        profile['profile_sha256'] = self.cost.profile_digest(profile)
        self.assertIs(self.cost.validate_profile(profile), profile)
        self.assertFalse(case['cost_workload_difference']['features_match'])

    def test_exact_duplicate_anchor_averages_nearest_two_and_pair_ties_are_stable(self):
        rows = surface_for_batch(32)
        rows[0][0][5], rows[0][1][5] = 130., 110.
        rows[1][0][5], rows[1][1][5] = 120., 120.
        for i in range(2,9):
            for row in rows[i]:row[5] = 200.
        answer = self.cost.predict_cost(rows,[192,1,1,36864,192],192)
        self.assertEqual(answer['candidate_index'],0)
        self.assertEqual(answer['predicted_us'],121.)

    def test_bad_measured_values_and_geometry_raise_without_heuristic_fallback(self):
        for value in (float('nan'),float('inf'),0,-1):
            rows=surface(); rows[1][0][5]=value
            with self.assertRaises(ValueError):self.cost.predict_cost(rows,[0]*5,192)
        rows=surface();rows[2][1][6]=10000
        with self.assertRaisesRegex(ValueError,'corrected'):self.cost.predict_cost(rows,[0]*5,192)
        with self.assertRaises(ValueError):self.cost.predict_cost(surface(),[193,1,1,193**2,193],192)

    def test_strict_profile_identity_digest_coverage_and_record_only(self):
        profile=profile_fixture(self.cost)
        self.assertIs(self.cost.validate_profile(profile,{'rank':0,'gpu':{'sm_count':132}}),profile)
        with self.assertRaisesRegex(ValueError,'identity.*rank'):
            self.cost.validate_profile(profile,{'rank':1})
        changed=copy.deepcopy(profile);changed['tables']['g0']['32']['rows'][0][0][5]+=1
        with self.assertRaisesRegex(ValueError,'digest'):self.cost.validate_profile(changed)
        changed=copy.deepcopy(profile);del changed['tables']['g0']['7']
        changed['profile_sha256']=self.cost.profile_digest(changed)
        with self.assertRaisesRegex(ValueError,'batch'):self.cost.validate_profile(changed)
        changed=copy.deepcopy(profile);changed['groups']['g0']['layers'].append(42)
        changed['profile_sha256']=self.cost.profile_digest(changed)
        with self.assertRaisesRegex(ValueError,'partition'):self.cost.validate_profile(changed)
        changed=copy.deepcopy(profile);changed['tables']['g0']['1']['validation']['complete']=False
        changed['profile_sha256']=self.cost.profile_digest(changed)
        with self.assertRaisesRegex(ValueError,'validation'):self.cost.validate_profile(changed)

    def test_digest_ignores_only_own_field_and_load_reports_missing_generation(self):
        profile=profile_fixture(self.cost)
        digest=self.cost.profile_digest(profile)
        profile['profile_sha256']='tampered'
        self.assertEqual(self.cost.profile_digest(profile),digest)
        profile['identity']['rank']=1
        self.assertNotEqual(self.cost.profile_digest(profile),digest)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'run_native_cost'):
                self.cost.load_profile(Path(td)/'missing.json')

    def test_profile_requires_measured_samples_stages_and_heldout_surfaces(self):
        for mutation in ('samples','stage','heldout','table_binding','geometry'):
            profile=profile_fixture(self.cost);entry=profile['tables']['g0']['32']
            if mutation=='samples':del entry['measurements']['anchors'][2]['candidates'][4]['full_moe']['samples_us']
            if mutation=='stage':del entry['measurements']['anchors'][1]['candidates'][3]['stages']['all_reduce']
            if mutation=='heldout':entry['validation']['heldout'][1]['fixed_candidates'].pop()
            if mutation=='table_binding':entry['rows'][0][0][5]+=1
            if mutation=='geometry':del entry['measurements']['geometry'][1]['native_grid']
            profile['profile_sha256']=self.cost.profile_digest(profile)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.cost.validate_profile(profile)

    def test_heldout_claims_must_match_measured_table_curves_and_deterministic_selection(self):
        mutations={
            'predicted_us':lambda case:case['prediction']['predicted_us'].__setitem__(0,100.),
            'actual_us':lambda case:case['prediction']['actual_us'].__setitem__(0,100.),
            'absolute_error_us':lambda case:case['prediction']['absolute_error_us'].__setitem__(0,1.),
            'relative_error':lambda case:case['prediction']['relative_error'].__setitem__(0,1.),
            'selected_actual_us':lambda case:case['prediction'].__setitem__('selected_actual_us',100.),
            'best_actual_us':lambda case:case['prediction'].__setitem__('best_actual_us',100.),
            'regret_us':lambda case:case['prediction'].__setitem__('regret_us',1.),
            'regret_fraction':lambda case:case['prediction'].__setitem__('regret_fraction',1.),
            'selection':lambda case:(case['prediction'].__setitem__('selected_index',8),case.update(selected_pair=[2,2])),
            'cpu_selection':lambda case:case.update(cpu_selected_index=8),
            'selection_agreement':lambda case:case.update(cpu_gpu_selection_agree=False),
        }
        for field,mutate in mutations.items():
            profile=profile_fixture(self.cost)
            mutate(profile['tables']['g0']['1']['validation']['heldout'][0])
            profile['profile_sha256']=self.cost.profile_digest(profile)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'heldout'):
                self.cost.validate_profile(profile)

    def test_stale_heldout_evidence_rejected_after_consistent_source_curve_changes(self):
        for source in ('table','heldout_raw'):
            profile=profile_fixture(self.cost);entry=profile['tables']['g0']['1']
            if source=='table':
                for anchor,row in zip(entry['measurements']['anchors'],entry['rows'][0]):
                    row[5]=200.
                    anchor['candidates'][0]['full_moe']=dict(samples_us=[200.,200.],median_us=200.,mean_us=200.,
                                                           stddev_us=0.,min_us=200.,max_us=200.)
            else:
                entry['validation']['heldout'][0]['fixed_candidates'][0]['full_moe']=dict(
                    samples_us=[200.,200.],median_us=200.,mean_us=200.,stddev_us=0.,min_us=200.,max_us=200.)
            profile['profile_sha256']=self.cost.profile_digest(profile)
            with self.subTest(source=source),self.assertRaisesRegex(ValueError,'heldout'):
                self.cost.validate_profile(profile)

    def test_self_consistent_large_finite_prediction_errors_and_regret_are_record_only(self):
        profile=profile_fixture(self.cost)
        case=profile['tables']['g0']['1']['validation']['heldout'][0]
        set_large_heldout_errors(case)
        profile['profile_sha256']=self.cost.profile_digest(profile)
        self.assertIs(self.cost.validate_profile(profile),profile)
        self.assertEqual(case['prediction']['regret_us'],999999899.)

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler unavailable')
    def test_host_cpp_prediction_and_worker_cap_match_policy(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'cost.cpp';exe=Path(td)/'cost'
            src.write_text('#include <iostream>\n#include <iomanip>\n#include "native_cost.h"\n'
                'int main(){float table[432],f[5];int total;'
                'for(auto &v:table)std::cin>>v;for(auto &v:f)std::cin>>v;std::cin>>total;'
                'auto p=weave_cost_predict(table,f,total);'
                'std::cout<<p.index<<" "<<std::setprecision(17)<<p.us<<"\\n";'
                'for(int grid:{33,528})for(int bucket:{2,4,8})'
                'std::cout<<weave_cost_workers(grid,bucket,1,2,2)<<" ";}')
            subprocess.run(['c++','-std=c++17','-I',str(root/'csrc'),str(src),'-o',str(exe)],check=True,capture_output=True)
            cases=[(surface(),features) for features in ([24,8,8,72,1.5],[0]*5,[48,16,16,144,3],[192,64,64,576,3])]
            near_tie=surface_for_batch(32)
            for index,candidate in enumerate(near_tie):
                for row in candidate:row[5:]=[100000000, .5 if index==1 else .25, .125]
            cases.append((near_tie,[192,1,1,36864,192]))
            different_anchors=surface()
            for row in different_anchors[1]:row[:]=[48,16,16,144,3,100,2,3]
            cases.append((different_anchors,[48,16,16,144,3]))
            for rows,features in cases:
                data=[v for candidate in rows for row in candidate for v in row]
                data+=list(features)+[192]
                result=subprocess.run([str(exe)],input=' '.join(map(str,data)),text=True,capture_output=True,check=True)
                lines=result.stdout.splitlines();index,value=lines[0].split()
                expected=self.cost.predict_cost(rows,features,192)
                self.assertEqual(int(index),expected['candidate_index'])
                self.assertAlmostEqual(float(value),expected['predicted_us'],places=4)
                self.assertEqual(list(map(int,lines[1].split())),[4]*6)

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler unavailable')
    def test_actual_planner_host_emulation_tracks_replayed_grouping_and_invalid_inputs(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as td:
            exe=Path(td)/'planner'
            subprocess.run(['c++','-std=c++20','-pthread','-I',str(root/'csrc'),
                            str(root/'tests/cpu/native_cost_planner_host.cpp'),'-o',str(exe)],
                           check=True,capture_output=True)
            result=subprocess.run([str(exe)],text=True,capture_output=True,check=True)
            plans=[list(map(int,line.split())) for line in result.stdout.splitlines()]
        empty,heavy,fixed,missing,padding,expert,interior,valid_prefix=plans
        self.assertEqual(empty[:10],[8,8,24,0,0,192,0,0,0,11000])
        self.assertEqual(heavy[:10],[2,2,24,24,192,192,24,1536,8,11000])
        self.assertEqual(fixed[:2],[4,2]);self.assertEqual(fixed[21],5)
        self.assertEqual(fixed[10],3);self.assertEqual(sum(fixed[11:20]),3)
        self.assertEqual([plan[20] for plan in plans],[0,0,0,1,2,3,4,4])
        self.assertEqual([plan[23] for plan in plans],[0,0,0,1,1,1,1,0])
        self.assertEqual(missing[:2],[0,0]);self.assertEqual(padding[2],0)
        self.assertEqual(expert[10],6)
        self.assertEqual(interior[:2],[0,0])
        self.assertEqual(valid_prefix[:9],[4,2,1,1,6,6,1,36,6])


if __name__=='__main__':unittest.main()
