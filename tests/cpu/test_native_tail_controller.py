"""Tail controller checks execution plans, protocol coverage and public boundaries."""
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.weave.gpu_allocation import GPUAllocation


def module():
    try:
        return importlib.import_module('scripts.run_native_tail')
    except ModuleNotFoundError:
        return None


class TailControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = module()
        self.assertIsNotNone(self.controller, 'native tail unified controller is missing')

    def preview(self, *extra):
        output = io.StringIO()
        with patch.object(self.controller.subprocess, 'run', side_effect=AssertionError('preview ran a process')):
            with contextlib.redirect_stdout(output):
                self.assertEqual(self.controller.main(['--work-root', '/srv/weave-work', *extra]), 0)
        return json.loads(output.getvalue())

    def test_foundations_preview_needs_no_dataset_and_only_runs_separate_s1_s2(self):
        value = self.preview('--stage', 'foundations')
        self.assertEqual(value['status'], 'preview_not_executed')
        self.assertEqual(value['stages'], ['compute', 'transport'])
        self.assertEqual(set(value['gpu_commands']), {'compute', 'transport'})
        self.assertFalse(value['gpu_mechanisms_verified'])
        self.assertFalse(value['throughput_result_observed'])

    def test_all_preview_has_dependency_order_and_exact_two_formal_arms(self):
        value = self.preview('--records', '/prepared/main.jsonl')
        self.assertEqual(value['stages'], ['compute', 'transport', 'executor', 'executor_memcheck', 'calibration',
                                          'measured_validation', 'native', 'peer_tail'])
        self.assertEqual(value['seconds_per_arm'], 1200)
        self.assertEqual(value['captured_batches'], list(range(1, 33)))
        self.assertEqual(set(value['groups']), {'native', 'peer_tail'})
        self.assertNotIn('--native-tail', value['groups']['native'])
        tail = value['groups']['peer_tail']
        self.assertEqual(tail[tail.index('--native-tail')+1], 'measured')
        self.assertNotIn('--native-budget', tail)
        self.assertFalse(value['complete_weave_claim'])
        memcheck=value['gpu_commands']['executor_memcheck']
        self.assertIn('--sanitizer-smoke',memcheck)
        self.assertIn('--error-exitcode',memcheck)
        self.assertEqual(memcheck[memcheck.index('--error-exitcode')+1],'99')

    def test_explicit_reuse_skips_generation_but_keeps_measured_gpu_validation(self):
        value = self.preview('--records', '/prepared/main.jsonl', '--calibration-dir', '/srv/weave-work/existing')
        self.assertIn('reuse_calibration', value['stages'])
        self.assertIsNone(value['calibration'])
        self.assertIn('measured_validation', value['stages'])

    def test_gpu_command_uses_actual_allocation_existing_image_and_offline_model(self):
        args = SimpleNamespace(work_root=Path('/srv/weave-work'), model_path=Path('/models/local'),
                               table_dir=Path('/srv/weave-work/tables'), gpu_timeout=28800)
        command = self.controller.gpu_command('transport', args, 'sha256:'+'a'*64,
                Path('/srv/weave-work/run/transport'), GPUAllocation.parse('4,6,8,9'), 'b'*64)
        self.assertEqual(command[command.index('--gpus')+1], '"device=4,6,8,9"')
        self.assertIn('--pull=never', command)
        self.assertIn('CUDA_VISIBLE_DEVICES=0,1,2,3', command)
        self.assertIn('/models/local:/model:ro', command)
        self.assertIn('HF_HUB_OFFLINE=1', command)
        self.assertNotIn('--network=none', command)

    def test_public_export_never_copies_raw_reports_or_trace_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'; (run/'transport').mkdir(parents=True)
            (run/'experiment.json').write_text('{"status":"failed"}')
            (run/'transport/check.json').write_text('{"status":"complete","task_bitmap_sha256":"abc"}')
            (run/'transport/rank0.json').write_text('{"task_smids":[1,2],"private":"secret"}')
            manifest = self.controller.export_public(run, Path(tmp)/'public')
            self.assertEqual(set(manifest['files']), {'experiment.json','transport/check.json'})
            self.assertFalse(manifest['raw_content_exported'])
            (run/'transport/check.json').write_text('{"trace":[[1,2]]}')
            with self.assertRaises(ValueError):
                self.controller.export_public(run, Path(tmp)/'bad')

    def test_large_finite_errors_remain_public_records(self):
        self.controller.safe_tail_public({'relative_l2': 1e20, 'max_abs': 1e30,
                                          'numerical_policy':'record_only'})
        for key in ('routes', 'activations', 'task_smids', 'sorted_token_ids', 'trace'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.controller.safe_tail_public({key:[1,2]})



def tail_identity():
    return dict(native_tail_source_sha256='a'*64, native_tail_upstream_sha256={'marlin':'b'*64},
        image_id='sha256:'+'c'*64, checkpoint_content_sha256='d'*64,
        tail_profiles={str(r):dict(profile_sha256=str(r)*64,identity={'rank':r},
                                  groups={'g0':{'layers':list(range(43))}}) for r in range(4)})


def tail_snapshots(calls):
    identity=tail_identity();result=[]
    for rank in range(4):
        profile=identity['tail_profiles'][str(rank)]
        rows=[dict(layer=layer,gate_up_calls=calls,tail_calls=calls,committed_calls=calls,
                   plan_search_calls=calls,empty_calls=0,histogram=[calls]+[0]*21,
                   published_chunks=calls,down_tiles=calls*5,stolen_tiles=0,remote_read_bytes=calls*1024)
              for layer in range(43)]
        executor=dict(schema_version=1,rank=rank,mode='measured',source_sha256='a'*64,
            profile_sha256=profile['profile_sha256'],
            profiles_sha256={r:p['profile_sha256'] for r,p in identity['tail_profiles'].items()},
            identity=profile['identity'],calibration_complete=True,rows=rows)
        result.append(dict(schema_version=1,method='native_marlin_peer_tail_v1',rank=rank,
            mode='measured',layer_count=43,numerical_policy='record_only',source_sha256='a'*64,
            upstream_sha256=identity['native_tail_upstream_sha256'],groups=profile['groups'],executor=executor))
    return result


class TailCoverageTests(unittest.TestCase):
    def test_each_replay_has_one_gate_up_tail_planner_and_committed_protocol(self):
        controller=module();self.assertIsNotNone(controller)
        value=controller.validate_tail_delta(tail_snapshots(2),tail_snapshots(7),{'replays':5},tail_identity())
        self.assertEqual(value['total_gate_up_calls'],860)
        self.assertEqual(value['total_tail_calls'],860)
        self.assertEqual(value['total_plan_search_calls'],860)
        self.assertFalse(value['plan_variation_observed'])
        self.assertEqual(value['ranks']['0']['rows'][0]['remote_read_bytes'],5120)
        self.assertEqual(value['ranks']['0']['rows'][0]['stolen_tiles'],0)

    def test_bad_source_missing_layer_duplicate_down_and_uncommitted_protocol_fail(self):
        controller=module()
        for mutation in ('source','missing','gate_up','tail','committed','planner','histogram','profile'):
            before,after=tail_snapshots(2),tail_snapshots(7)
            row=after[0]['executor']['rows'][0]
            if mutation=='source':after[0]['source_sha256']='f'*64
            if mutation=='missing':after[0]['executor']['rows'].pop()
            if mutation=='gate_up':row['gate_up_calls']=12
            if mutation=='tail':row['tail_calls']=12;row['committed_calls']=12;row['histogram'][0]=12
            if mutation=='committed':row['committed_calls']=6
            if mutation=='planner':row['plan_search_calls']=6
            if mutation=='histogram':row['histogram'][0]=6
            if mutation=='profile':after[0]['executor']['profiles_sha256']['0']='f'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                controller.validate_tail_delta(before,after,{'replays':5},tail_identity())

    def test_metadata_activation_dtype_is_public_but_tensor_is_not(self):
        controller=module()
        controller.safe_tail_public({'precision':{'activations':'bfloat16'}})
        with self.assertRaises(ValueError):controller.safe_tail_public({'activations':[1.]})



class TailCleanupTests(unittest.TestCase):
    def test_two_phase_ipc_close_requests_all_rank_endpoints_concurrently(self):
        import threading
        controller=module();barrier=threading.Barrier(4)
        def endpoint(url, timeout):
            rank=int(url.split(':')[-1].split('/')[0])-30310
            barrier.wait(timeout=2)
            prepared=url.endswith('/tail-close-prepare')
            return dict(rank=rank,status='prepared' if prepared else 'complete',
                        graphs_owned=32,graphs_reset=32,peers_disconnected=True,ipc_closed=not prepared)
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(graph_evidence_port_base=30310,cleanup_timeout=10.)
            with patch.object(controller,'http_json',side_effect=endpoint):
                result=controller.close_tail_service(args,Path(tmp))
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['graphs_reset'],128)
            self.assertTrue(result['ipc_closed'])
            self.assertTrue((Path(tmp)/'tail-cleanup.json').is_file())

    def test_failure_cleanup_does_not_export_private_error_text(self):
        from unittest.mock import MagicMock
        controller=module();owners=[MagicMock(),MagicMock()]
        with tempfile.TemporaryDirectory() as tmp:
            output=io.StringIO()
            with patch.object(controller,'OwnedChild',side_effect=owners),patch.object(controller,'allocated_gpus',return_value=GPUAllocation.parse('4,6,8,9')),patch.object(controller,'load_gpqa_workload',side_effect=ValueError('private input contents')),contextlib.redirect_stdout(output):
                code=controller.main(['--work-root',tmp,'--records',tmp+'/main.jsonl','--execute'])
            self.assertEqual(code,1)
            value=json.loads(output.getvalue())
            exported=json.loads((Path(value['export_directory'])/'experiment.json').read_text())
            self.assertEqual(exported['status'],'failed')
            self.assertNotIn('private input contents',json.dumps(exported))
            self.assertFalse(exported['gpu_mechanisms_verified'])
            for owner in owners:owner.stop.assert_called_once()



def measured_table_fixture(batch):
    from src.weave.native_tail_cost import Plan
    plans=[Plan(0,1,0,True).to_dict(),Plan(1,1,0).to_dict()]
    timing=dict(samples_us=[1.,3.],median_us=2.,mean_us=2.,stddev_us=1.,min_us=1.,max_us=3.)
    surface=[]
    for i,plan in enumerate(plans):
        surface.append([dict(plan=plan,features=[i+1,1,1,(i+1)**2,i+1],front_us=1.,prepare_us=1.,
            planner_us=1.,finish_us=1.,role_switch_us=0.,down_us=[2.],finalize_us=[1.],
            fetch_reduce_us=[1.],fetch_result_us=[1.]) for _ in range(4)])
    anchors=[]
    for position,index in enumerate((0,1,4,5)):
        records=[]
        for candidate,plan in enumerate(plans):
            observed=dict(surface[candidate][position])
            records.append(dict(plan=plan,full_moe=timing,representative_observation=0,
                diagnostic_observations=[dict(protocol_valid=True,curve=observed),dict(protocol_valid=True,curve=observed)]))
            surface[candidate][position]['fixed_planner_us']=1.
        anchors.append(dict(index=index,candidates=records,search_planner=dict(
            representative_observation=0,observations=[dict(planner_us=1.),dict(planner_us=1.)])))
    heldout=[dict(index=index,cpu_gpu_selection_agree=True,native_numerical=dict(finite=True,
                 relative_l2=1e100,max_abs=1e100,numerical_policy='record_only'),
                 native_full_moe=timing,selected_full_moe=timing,
                 fixed_candidates=[dict(plan=plan,full_moe=timing) for plan in plans]) for index in (6,7)]
    return dict(plans=plans,total_rows=batch*6,surface=surface,
                measurements=dict(complete=True,repeats=2,anchors=anchors),heldout=heldout)


class TailProfileTests(unittest.TestCase):
    def write_profiles(self,directory,mutate=None):
        from src.weave.native_tail_cost import profile_digest
        identity=tail_identity()
        for rank in range(4):
            bound=dict(rank=rank,tail_source_sha256=identity['native_tail_source_sha256'],
                upstream_sha256=identity['native_tail_upstream_sha256'],image_id=identity['image_id'],
                checkpoint_content_sha256=identity['checkpoint_content_sha256'])
            profile=dict(method='native_marlin_peer_tail_v1',policy='measured_tail_dag_v1',
                numerical_policy='record_only',status='complete',identity=bound,
                groups={'g0':{'layers':list(range(43))}},
                tables={f'g0:{b}':measured_table_fixture(b) for b in range(1,33)})
            if mutate and rank==0:mutate(profile)
            profile['profile_sha256']=profile_digest(profile)
            (Path(directory)/f'native-tail-rank{rank}.json').write_text(json.dumps(profile))
        return identity

    def test_complete_actual_schema_loads_despite_large_error_and_workload_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity=self.write_profiles(tmp)
            profiles=module().load_profiles(tmp,identity)
            self.assertEqual(len(profiles),4)
            self.assertEqual(len(profiles['0']['tables']),32)

    def test_missing_table_and_incomplete_execution_evidence_rejected(self):
        for mutation in ('batch','measurements','heldout','planner','nonfinite','unmeasured_curve'):
            def mutate(p):
                table=p['tables']['g0:32']
                if mutation=='batch':p['tables'].pop('g0:32')
                if mutation=='measurements':table['measurements']['complete']=False
                if mutation=='heldout':table['heldout'].pop()
                if mutation=='planner':table['heldout'][0]['cpu_gpu_selection_agree']=False
                if mutation=='nonfinite':table['heldout'][0]['native_numerical']['finite']=False
                if mutation=='unmeasured_curve':table['surface'][0][0]['front_us']=4.
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                identity=self.write_profiles(tmp,mutate)
                with self.assertRaises(ValueError):module().load_profiles(tmp,identity)



class TailGpuSummaryTests(unittest.TestCase):
    def test_task_coverage_flags_and_exact_measured_case_matrix_are_required(self):
        import hashlib
        controller=module();identity=tail_identity()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for rank in range(4):
                cases=[dict(batch=32,mode='measured',pattern=p,replay=i,protocol_valid=True,
                    canary_ok=True,native_locks_zero=True,stable_graph_addresses=True,
                    exact_task_coverage=True,front_and_tail_calls=1)
                    for p in ('concentrated','balanced','light','empty_rank2') for i in range(5)]
                report=dict(rank=rank,method='native_marlin_peer_tail_v1',numerical_policy='record_only',
                    status='complete',checks_complete=True,cleanup={'status':'complete'},cases=cases,
                    case_count=20,expected_case_count=20,source_sha256='a'*64,
                    upstream_sha256=identity['native_tail_upstream_sha256'],runtime={'image_id':identity['image_id']},
                    harness_sha256=hashlib.sha256((controller.ROOT/'tests/gpu/check_native_tail_executor.py').read_bytes()).hexdigest(),
                    graph_capture_replay={'verified':True,'replays':20},profile_sha256=str(rank)*64,
                    planner_validation={'complete':True},gpu_mechanisms_verified=True)
                (folder/f'native-tail-executor-rank{rank}.json').write_text(json.dumps(report))
            self.assertEqual(controller.summarize_gpu_reports('measured_validation',folder,identity)['status'],'complete')
            path=folder/'native-tail-executor-rank0.json';bad=json.loads(path.read_text())
            bad['cases'][0]['exact_task_coverage']=False;path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):controller.summarize_gpu_reports('measured_validation',folder,identity)


if __name__ == '__main__':
    unittest.main()
