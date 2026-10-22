import unittest
import ast
from types import SimpleNamespace as NS
from src.weave.native_tail_cost import Plan
from src.weave.native_tail_calibration import (curve_from_snapshot, workload_difference,
    prediction_record, verify_current_plan, _median_curve, _observe_join)
from src.weave.native_tail_front import _compile_function


class CalibrationTests(unittest.TestCase):
    def snapshot(self):
        return dict(features=[7,3,3,19,3], local_groups=3,
            start_ns=1000,planner_begin_ns=1200,planner_end_ns=1700,planned_ns=2000,
            completion_begin_ns=10000,finish_ns=11000,role_switch_ns=600,
            chunk_times_ns=[[2000,4000,4000,4500,4500,6500,6500,7000],
                            [3500,6000,6000,6500,6500,8500,8500,9500],
                            [0]*8,[0]*8])

    def test_curve_uses_rank_local_observed_intervals_and_keeps_overlap(self):
        curve=curve_from_snapshot(Plan(2,2,1),self.snapshot(),front_us=3.5)
        self.assertEqual(curve['features'],[7,3,3,19,3])
        self.assertEqual([curve[k] for k in ('front_us','prepare_us','planner_us','finish_us','role_switch_us')],
                         [3.5,.5,.5,1.,.6])
        self.assertEqual(curve['down_us'],[2.,2.5])
        self.assertEqual(curve['finalize_us'],[.5,.5])
        self.assertEqual(curve['fetch_reduce_us'],[2.,2.])
        self.assertEqual(curve['fetch_result_us'],[.5,1.])

    def test_missing_or_reversed_timestamps_cannot_become_fabricated_costs(self):
        for change in ({'planner_end_ns':None},{'planned_ns':1600}, {'role_switch_ns':-1}):
            with self.assertRaises(ValueError):
                curve_from_snapshot(Plan(2,2,1),dict(self.snapshot(),**change),front_us=3.5)

    def test_empty_local_work_has_zero_down_only_with_explicit_empty_groups(self):
        state=self.snapshot();state.update(features=[0]*5,local_groups=0)
        state['chunk_times_ns'][0][:2]=[(1<<64)-1,0]
        state['chunk_times_ns'][1][:2]=[(1<<64)-1,0]
        curve=curve_from_snapshot(Plan(2,2,1),state,front_us=3.5)
        self.assertEqual(curve['down_us'],[0.,0.])
        state['local_groups']=1
        with self.assertRaises(ValueError):curve_from_snapshot(Plan(2,2,1),state,front_us=3.5)

    def test_workload_and_finite_prediction_regressions_are_record_only(self):
        difference=workload_difference(dict(features=[1,1,1,1,1],native_routing_sha256='a'),
            dict(features=[6,3,3,14,3],native_routing_sha256='b'))
        self.assertEqual(difference['features_delta'],[5,2,2,13,2])
        self.assertTrue(difference['routing_digest_changed'])
        self.assertEqual(difference['policy'],'record_only')
        value=prediction_record(1.,[100.,10.],actual_selected_us=100.)
        self.assertEqual(value['regret_us'],90.)
        self.assertEqual(value['numerical_policy'],'record_only')

    def test_planner_validity_uses_current_replay_features(self):
        plan=Plan(1,1,0)
        anchor=dict(plan=plan.to_dict(),features=[1,1,1,1,1],front_us=1.,prepare_us=1.,
                    planner_us=1.,finish_us=1.,role_switch_us=0.,down_us=[1.],
                    finalize_us=[1.],fetch_reduce_us=[1.],fetch_result_us=[1.])
        surfaces=[[[dict(anchor),dict(anchor,features=[6,2,2,20,4])]] for _ in range(4)]
        snapshots=[dict(features=[4,2,2,8,2],chosen_indices=[0]*4,selected_index=0) for _ in range(4)]
        result=verify_current_plan(surfaces,snapshots,[plan],6)
        self.assertEqual(result['indices'],[0]*4)
        snapshots[2]['chosen_indices']=[1]*4
        with self.assertRaisesRegex(RuntimeError,'planner'):
            verify_current_plan(surfaces,snapshots,[plan],6)

    def test_repeated_workload_variation_keeps_one_actual_curve_and_features_together(self):
        plan=Plan(2,2,1)
        first=curve_from_snapshot(plan,self.snapshot(),front_us=3.5)
        second=dict(first,features=[8,3,3,22,4],front_us=10.)
        third=dict(first,features=[4,2,2,8,2],front_us=5.)
        curve,index=_median_curve(plan,[first,second,third])
        self.assertEqual(index,2)
        self.assertEqual(curve,third)

    def test_private_native_join_observer_preserves_operation_order(self):
        source='''def run(self):
    current_stream = self.stream
    self.routed()
    self.shared()
    current_stream.wait_stream(self.alt_stream)
    return self.output
'''
        log=[]
        function=_compile_function(_observe_join(ast.parse(source).body[0]),
            {'_weave_mark':lambda event:log.append(event)},'<test-native-join>')
        mlp=NS(stream=NS(wait_stream=lambda alt:log.append('wait')),alt_stream='shared',
               routed=lambda:log.append('routed'),shared=lambda:log.append('shared'),output='result')
        self.assertEqual(function(mlp),'result')
        self.assertEqual(log,['entry','routed','shared','before_join','wait','after_join'])


if __name__=='__main__':unittest.main()
