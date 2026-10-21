"""Cost service boundaries: no guessed table, complete counters, owned cleanup."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tests.cpu.test_native_budget_controller import budgets, graphs, IDENTITY as BASE_IDENTITY
from tests.cpu.test_gpqa_c32_contract import SummaryTests

ROOT = Path(__file__).resolve().parents[2]
if importlib.util.find_spec('scripts.run_native_cost'):
    from scripts import run_native_cost as controller
else:
    controller = None

IDENTITY = dict(native_cost_source_sha256='b'*64, native_cost_upstream_sha256={'marlin.cuh':'a'*64},
                cost_profiles={str(i): {'profile_sha256': str(i)*64, 'identity': {'rank':i}} for i in range(4)})
IDENTITY.update({k:v for k,v in BASE_IDENTITY.items() if not k.startswith('native_budget')})


def snapshots(calls):
    result = budgets(calls)
    for rank, item in enumerate(result):
        item.update(method='native_marlin_measured_cta_v2', mode='cost', world_size=4,
                    calibration_complete=True, profile_sha256=str(rank)*64, identity={'rank':rank},
                    policy={'version':'measured_moe_surface_v1','candidate_pairs':[[8,8],[8,4],[8,2],[4,8],[4,4],[4,2],[2,8],[2,4],[2,2]]})
        item['plans'] = []
        for layer in range(43):
            values = [0]*32
            values[0:2] = [8,8]; values[9] = 12300; values[10] = calls; values[11] = calls
            item['plans'].append({'layer':layer,'values':values})
    return result


class CostLauncherTests(unittest.TestCase):
    def launch(self, *flags):
        env = {**os.environ,'CUDA_VISIBLE_DEVICES':'4,5,6,7',
               'WEAVE_MAX_RUNNING_REQUESTS':'32','WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'}
        return subprocess.run([sys.executable,'scripts/run_native_ep.py','--work-root','/tmp/weave-work',
            '--backend','native','--decode-graph','full','--graph-c32','--request-timing',*flags],
            cwd=ROOT,env=env,capture_output=True,text=True)

    def test_formal_cost_uses_table_and_never_sends_heuristic_target(self):
        value = self.launch('--native-budget','cost','--native-cost-table-dir','/tmp/weave-work/tables')
        self.assertEqual(value.returncode,0,value.stderr)
        command = json.loads(value.stdout)['docker_argv']
        self.assertIn('WEAVE_NATIVE_COST_TABLE_DIR=/work/tables',command)
        self.assertIn('WEAVE_NATIVE_BUDGET=cost',command)
        self.assertFalse(any('WEAVE_NATIVE_BUDGET_TARGET_PERCENT=' in x for x in command))

    def test_calibration_uses_separate_directory_and_default_repeats(self):
        value = self.launch('--native-budget','cost','--native-cost-calibrate-dir','/tmp/weave-work/tables')
        self.assertEqual(value.returncode,0,value.stderr)
        command = json.loads(value.stdout)['docker_argv']
        self.assertIn('WEAVE_NATIVE_COST_CALIBRATE_DIR=/work/tables',command)
        self.assertIn('WEAVE_NATIVE_COST_REPEATS=7',command)

    def test_missing_table_modes_and_paths_fail_actionably(self):
        missing = self.launch('--native-budget','cost')
        self.assertNotEqual(missing.returncode,0)
        self.assertIn('run_native_cost.py',missing.stderr)
        for flags in [('--native-budget','cost','--native-cost-table-dir','/outside'),
                      ('--native-cost-table-dir','/tmp/weave-work/tables'),
                      ('--native-budget','cost','--native-cost-table-dir','/tmp/weave-work/tables','--native-cost-calibrate-dir','/tmp/weave-work/cal'),
                      ('--native-budget','cost','--native-cost-calibrate-dir','/tmp/weave-work/cal','--native-cost-repeats','0'),
                      ('--native-budget','cost','--native-cost-calibrate-dir','/tmp/weave-work/cal','--native-budget-target-percent','50')]:
            with self.subTest(flags=flags): self.assertNotEqual(self.launch(*flags).returncode,0)

    def test_heuristic_alias_preserves_adaptive_runtime(self):
        value=self.launch('--native-budget','heuristic')
        self.assertEqual(value.returncode,0,value.stderr)
        self.assertIn('WEAVE_NATIVE_BUDGET=adaptive',json.loads(value.stdout)['docker_argv'])

    def test_long_calibration_watchdog_does_not_change_formal_watchdog(self):
        calibration=self.launch('--native-budget','cost','--native-cost-calibrate-dir','/tmp/weave-work/cal',
                                '--native-cost-calibration-timeout','43200')
        self.assertEqual(calibration.returncode,0,calibration.stderr)
        formal=self.launch('--native-budget','cost','--native-cost-table-dir','/tmp/weave-work/tables')
        for value,want in ((calibration,'43200'),(formal,'1800')):
            command=json.loads(value.stdout)['docker_argv']
            self.assertEqual(command[command.index('--watchdog-timeout')+1],want)

    def test_execute_cannot_overwrite_existing_rank_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'calibration';directory.mkdir()
            profile=directory/'native-cost-rank0.json';profile.write_text('{"historical":true}')
            result=self.launch('--work-root',tmp,'--native-budget','cost','--native-cost-calibrate-dir',
                               str(directory),'--checkpoint-content-sha256','a'*64,'--execute')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('fresh',result.stderr)
            self.assertEqual(profile.read_text(),'{"historical":true}')

    def test_calibration_requires_repeated_measurements_without_arbitrary_maximum(self):
        for repeats,want in ((1,False),(2,True),(101,True)):
            result=self.launch('--native-budget','cost','--native-cost-calibrate-dir','/tmp/weave-work/cal',
                               '--native-cost-repeats',str(repeats))
            self.assertEqual(result.returncode==0,want,result.stderr)

    def test_written_cost_launch_report_has_measured_policy_without_heuristic_fields(self):
        from scripts import run_native_ep
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir()
            command=['run_native_ep.py','--backend','native','--decode-graph','full','--graph-c32',
                '--request-timing','--native-budget','cost','--native-cost-calibrate-dir',str(root/'tables'),
                '--work-root',tmp,'--model-path',str(model),'--checkpoint-content-sha256','a'*64,'--execute']
            image=SimpleNamespace(stdout='sha256:'+'b'*64)
            lifecycle={'exit_code':7,'termination_signal':None,'cleanup_status':'stopped'}
            env={'CUDA_VISIBLE_DEVICES':'4,5,6,7','WEAVE_MAX_RUNNING_REQUESTS':'32','WEAVE_CUDA_GRAPH_MAX_BS_DECODE':'32'}
            with patch.object(sys,'argv',command),patch.dict(os.environ,env),patch.object(run_native_ep.subprocess,'run',return_value=image),patch.object(run_native_ep,'run_container',return_value=lifecycle),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_native_ep.main(),7)
            report=json.loads(next(root.glob('native-ep-*/launch.json')).read_text())
            self.assertIsNone(report['native_budget_configuration'])
            self.assertEqual(report['native_cost_configuration'].get('policy_version'),'measured_moe_surface_v1')
            self.assertNotIn('target_percent',report['native_cost_configuration'])
            self.assertEqual(report['method'],'native_marlin_measured_cta_v2')


class CostControllerTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(controller,'unified native cost controller must exist')

    def test_offline_preview_orders_calibration_gpu_validation_and_two_1200s_arms(self):
        output=io.StringIO()
        with patch.object(controller.subprocess,'run',side_effect=AssertionError('preview executed')), contextlib.redirect_stdout(output):
            self.assertEqual(controller.main(['--work-root','/work','--records','/prepared/main.jsonl']),0)
        value=json.loads(output.getvalue())
        self.assertEqual(value['stages'],['calibration','gpu_validation','native','cost'])
        self.assertEqual(value['seconds_per_arm'],1200)
        self.assertEqual(set(value['groups']),{'native','cost'})
        self.assertIn('--native-cost-calibrate-dir',value['calibration'])
        self.assertIn('--table-dir',value['gpu_validation'])
        for variant, command in value['groups'].items():
            self.assertIn('--graph-c32',command)
            self.assertNotIn('--native-p0',command)
            self.assertEqual('--native-budget' in command,variant=='cost')
        self.assertFalse(value['complete_weave_claim'])

    def test_unified_entry_requires_two_repeats_and_accepts_larger_counts(self):
        for repeats in ('2','101'):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(controller.main(['--work-root','/work','--records','/prepared/main.jsonl',
                                                  '--native-cost-repeats',repeats]),0)
            self.assertEqual(json.loads(output.getvalue())['calibration_repeats'],int(repeats))
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            controller.main(['--work-root','/work','--records','/prepared/main.jsonl','--native-cost-repeats','1'])

    def test_full_layer_role_and_once_per_moe_planner_replay_coverage(self):
        result=controller.validate_cost_delta(snapshots(2),snapshots(7),{'replays':5},IDENTITY)
        self.assertEqual(result['total_calls'],1720)
        self.assertEqual(result['total_planner_calls'],860)
        self.assertEqual(result['plans_with_pair_variation'],0)
        self.assertFalse(result['pair_variation_observed'])

    def test_bad_or_stale_plan_evidence_cannot_masquerade_as_coverage(self):
        for mutation in ('missing','calls','histogram','invalid','digest','identity','negative','nonfinite','rowcalls'):
            a,b=snapshots(2),snapshots(7)
            plan=b[0]['plans'][0]['values']
            if mutation=='missing': b[0]['plans'].pop()
            if mutation=='calls': plan[10]=6
            if mutation=='histogram': plan[11]=6
            if mutation=='invalid': plan[20]=1
            if mutation=='digest': b[0]['profile_sha256']='x'*64
            if mutation=='identity': b[0]['identity']={'rank':3}
            if mutation=='negative': plan[9]=-1
            if mutation=='nonfinite': plan[9]=float('nan')
            if mutation=='rowcalls': b[0]['rows'][0]['calls']=6
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                controller.validate_cost_delta(a,b,{'replays':5},IDENTITY)

    def test_failure_stops_all_owned_children_and_exports_no_private_error(self):
        from src.weave.gpu_allocation import GPUAllocation
        with tempfile.TemporaryDirectory() as tmp:
            owners=[MagicMock(),MagicMock()];output=io.StringIO()
            with patch.object(controller,'OwnedChild',side_effect=owners), patch.object(controller,'allocated_gpus',return_value=GPUAllocation.parse('0,1,2,3')), patch.object(controller,'load_gpqa_workload',side_effect=ValueError('private contents')), contextlib.redirect_stdout(output):
                code=controller.main(['--work-root',tmp,'--records',tmp+'/main.jsonl','--execute'])
            self.assertEqual(code,1)
            value=json.loads(output.getvalue())
            report=json.loads((Path(value['export_directory'])/'experiment.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertNotIn('private contents',json.dumps(report))
            for owner in owners: owner.stop.assert_called_once()

    def test_public_export_includes_aggregates_profiles_and_excludes_raw_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';(root/'calibration').mkdir(parents=True)
            (root/'experiment.json').write_text('{"status":"failed"}')
            (root/'calibration/native-cost-rank0.json').write_text('{"numerical_policy":"record_only"}')
            (root/'calibration/server-private.log').write_text('secret')
            exported=controller.export_public(root,Path(tmp)/'public')
            self.assertEqual(set(exported['files']),{'experiment.json','calibration/native-cost-rank0.json'})
            (root/'calibration/native-cost-rank1.json').write_text('{"activations":[1,2]}')
            with self.assertRaises(ValueError): controller.export_public(root,Path(tmp)/'bad')

    def test_resume_profiles_require_full_coverage_identity_and_two_heldout_cases(self):
        from src.weave import native_cost
        from tests.cpu.test_native_cost import profile_fixture, set_large_heldout_errors
        with tempfile.TemporaryDirectory() as tmp:
            first=profile_fixture(native_cost)
            expected=dict(native_cost_source_sha256=first['identity']['cost_source_sha256'],
                native_cost_upstream_sha256=first['identity']['upstream_sha256'],
                image_id=first['identity']['image_id'],checkpoint_content_sha256=first['identity']['checkpoint_content_sha256'])
            for rank in range(4):
                item=copy.deepcopy(first);item['identity']['rank']=rank
                for table in item['tables']['g0'].values():
                    for heldout in table['validation']['heldout']:
                        heldout['native_numerical']['relative_l2']=100000.
                        set_large_heldout_errors(heldout)
                item['profile_sha256']=native_cost.profile_digest(item)
                (Path(tmp)/f'native-cost-rank{rank}.json').write_text(json.dumps(item))
            profiles=controller.load_profiles(tmp,expected)
            self.assertEqual(len(profiles),4)
            bad=copy.deepcopy(profiles['2']);bad['tables']['g0'].pop('32')
            bad['profile_sha256']=native_cost.profile_digest(bad)
            (Path(tmp)/'native-cost-rank2.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'batch coverage'): controller.load_profiles(tmp,expected)

    def test_calibration_service_ready_without_complete_profiles_fails_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'experiment';out.mkdir()
            args=SimpleNamespace(work_root=root,table_dir=out/'calibration',model_path=root/'model',
                port=30001,evidence_port_base=30110,graph_evidence_port_base=30310,native_cost_repeats=7,
                calibration_timeout=43200.)
            owner=MagicMock();server=MagicMock();owner.spawn.return_value=server
            folder=root/'native-ep-cal';folder.mkdir()
            controller.atomic_json(folder/'runtime.json',dict(runtime_status='passed',resolved_image_id='sha256:pinned',
                kernel_source_sha256='c'*64,host_gpu_allocation='0,1,2,3'))
            def cleanup(): controller.atomic_json(folder/'container_lifecycle.json',{'cleanup_status':'stopped'})
            owner.stop.side_effect=cleanup
            with patch.object(controller,'assert_ports_free'), patch.object(controller,'wait_ready'), patch.object(controller,'load_profiles',side_effect=ValueError('incomplete')):
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    controller.run_calibration(args,out,'native-ep-cal',IDENTITY,{'CUDA_VISIBLE_DEVICES':'0,1,2,3'},owner,lambda:None)
            check=json.loads((out/'calibration/check.json').read_text())
            self.assertEqual(check['status'],'failed')
            self.assertEqual(json.loads((out/'calibration/server-lifecycle.json').read_text())['cleanup_status'],'stopped')

    def test_whole_window_comparison_retains_exhaustion_scope(self):
        native,cost=SummaryTests().summary(),SummaryTests().summary()
        cost.update(decode_tokens_in_window=576,output_tokens_in_window=608,decode_tokens_per_second=.48,
            requests_started=448,requests_completed=417,requests_unfinished=31,requests_unattempted=0,
            completed_row_indices=list(range(417)),refill_exhausted=True,refill_exhausted_monotonic=1100.)
        value=controller.compare_cost_groups({'native':native,'cost':cost})
        self.assertEqual(value['cost_vs_native_whole_window_ratio'],2.)
        self.assertIsNone(value['cost_vs_native_sustained_c32_ratio'])
        self.assertEqual(value['decode_tokens_per_second'],{'native':.24,'cost':.48})

    def test_budget_reads_bracket_cost_measurement_and_never_touch_native(self):
        for variant in ('native','cost'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);out=root/'experiment';out.mkdir();folder=out/variant
                args=SimpleNamespace(work_root=root,records=root/'main.jsonl',model_path=root/'model',
                    table_dir=root/'tables',native_cost_repeats=7,port=30001,evidence_port_base=30110,
                    graph_evidence_port_base=30310,setup_timeout=7200.,warmup_timeout=960.,drain_timeout=60.)
                summary=SummaryTests().summary();workload=SimpleNamespace(identity=summary['workload'])
                state={'phase':'starting','polls':0};reads=[]
                server=MagicMock();server.poll.return_value=None
                client=MagicMock();client.returncode=0
                owner,client_owner=MagicMock(),MagicMock()
                def spawn_server(command,lifecycle,**kwargs):
                    lifecycle.parent.mkdir()
                    controller.atomic_json(lifecycle.parent/'runtime.json',dict(runtime_status='passed',resolved_image_id='sha256:pinned',kernel_source_sha256='c'*64,host_gpu_allocation='0,1,2,3'))
                    (lifecycle.parent/'server.log').write_text('')
                    return server
                def spawn_client(command,lifecycle,**kwargs):
                    controller.atomic_json(folder/'warmup-ready.json',{'ready_monotonic':100.})
                    controller.atomic_json(folder/'measurement-summary.json',summary)
                    state['phase']='idle_before'
                    return client
                def poll():
                    state['polls']+=1
                    if state['polls']==3:
                        state['phase']='idle_after';return 0
                    return None
                def snapshot(base,*,deadline,endpoint):
                    self.assertIn(state['phase'],('idle_before','idle_after'))
                    reads.append((state['phase'],endpoint))
                    calls=2 if state['phase']=='idle_before' else 7
                    return snapshots(calls) if endpoint=='budget-coverage' else graphs(calls)
                real_atomic=controller.atomic_json
                def publish(path,value):
                    real_atomic(path,value)
                    if Path(path).name=='measurement-go.json':
                        state['phase']='measuring'
                        real_atomic(folder/'load-window.json',{'started_monotonic':100.,'deadline_monotonic':1300.})
                owner.spawn.side_effect=spawn_server;client_owner.spawn.side_effect=spawn_client;client.poll.side_effect=poll
                with patch.object(controller,'wait_ready'),patch.object(controller,'assert_ports_free'),patch.object(controller,'flush_cache_when_idle'),patch.object(controller,'baseline_coverage',side_effect=snapshot),patch.object(controller,'atomic_json',side_effect=publish),patch.object(controller.time,'monotonic',return_value=100.),patch.object(controller.time,'sleep'):
                    result=controller.run_group(args,variant,out,'native-ep-unit-'+variant,IDENTITY,workload,'sha256:pinned',{'CUDA_VISIBLE_DEVICES':'0,1,2,3'},owner,client_owner,lambda:None)
                self.assertEqual(result['status'],'passed')
                budget_reads=[phase for phase,endpoint in reads if endpoint=='budget-coverage']
                self.assertEqual(budget_reads,['idle_before','idle_after'] if variant=='cost' else [])


if __name__=='__main__': unittest.main()
