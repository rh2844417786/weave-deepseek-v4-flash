import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from src.weave.gpu_allocation import GPUAllocation,allocated_gpus
except ImportError:
    GPUAllocation=allocated_gpus=None

ROOT=Path(__file__).resolve().parents[2]


class GPUAllocationTest(unittest.TestCase):
    def test_full_controller_failure_report_records_scheduler_allocation(self):
        from tests.cpu.test_gpqa_performance import make_records
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();records=make_records(root);bindir=root/'bin';bindir.mkdir()
            tool=bindir/'docker';tool.write_text('#!'+sys.executable+'\nimport sys\nsys.exit(7)\n');tool.chmod(0o755)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_full_experiment.py'),
                '--work-root',str(root/'work'),'--records',str(records)],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,2,4,6','PATH':str(bindir)+os.pathsep+os.environ['PATH']},
                capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            reports=list((root/'work').glob('full-experiment-*/experiment.json'))
            self.assertEqual(len(reports),1,result.stderr)
            report=json.loads(reports[0].read_text())
            self.assertEqual(report['physical_gpus'],[0,2,4,6])
            self.assertEqual(report['host_gpu_allocation'],'0,2,4,6')
            self.assertEqual(report['performance'],'not_run')

    def test_model_acceptance_accepts_other_allocations_before_input_gate(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_model_acceptance.py'),
            '--records','/nonexistent/weave-allocation-fixture.jsonl'],
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'1,3,5,7'},capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('prepare the pinned Think High',result.stderr)

    def test_scheduler_allocation_drives_host_selection_and_container_ordinals(self):
        self.assertIsNotNone(GPUAllocation)
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':' 0, 02,4,6 '}):
            allocation=allocated_gpus()
        self.assertEqual(allocation.ids,(0,2,4,6))
        self.assertEqual(allocation.csv,'0,2,4,6')
        self.assertEqual(allocation.docker_argument,'"device=0,2,4,6"')
        self.assertEqual(allocation.container_visible,'0,1,2,3')

    def test_invalid_or_missing_allocation_never_defaults_to_all_cards(self):
        self.assertIsNotNone(GPUAllocation)
        for value in ('','all','0,1,2','0,1,2,3,4','0,1,2,2','0,1,2,-1','0,1,2,3;echo bad'):
            with self.subTest(value=value),self.assertRaises(ValueError): GPUAllocation.parse(value)
        with patch.dict(os.environ,{},clear=True),self.assertRaises(ValueError): allocated_gpus()

    def test_native_preview_uses_arbitrary_four_card_allocations(self):
        for allocation in ('0,1,2,3','1,3,5,7'):
            run=subprocess.run([sys.executable,str(ROOT/'scripts/run_native_ep.py')],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':allocation},capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            command=json.loads(run.stdout)['docker_argv']
            self.assertEqual(command[command.index('--gpus')+1],f'"device={allocation}"')
            self.assertIn('WEAVE_HOST_ALLOCATION='+allocation,command)
            self.assertIn('CUDA_VISIBLE_DEVICES=0,1,2,3',command)
            self.assertEqual(command[command.index('--ep-size')+1],'4')

    def test_calibration_docker_and_export_record_actual_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();bindir=root/'bin';bindir.mkdir();capture=root/'argv.json'
            docker=bindir/'docker'
            docker.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                'if sys.argv[1:3]==["image","inspect"]: print("sha256:"+"a"*64)\n'
                'elif sys.argv[1]=="run":\n'
                '    open(os.environ["DOCKER_CAPTURE"],"w").write(json.dumps(sys.argv[1:]));sys.exit(1)\n'
                'else: print("No such object",file=sys.stderr);sys.exit(1)\n')
            docker.chmod(0o755)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'1,3,5,7','DOCKER_CAPTURE':str(capture),
                 'PATH':str(bindir)+os.pathsep+os.environ['PATH']}
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_calibration_job.py'),
                '--model-path',str(root/'model'),'--work-root',str(root),'--core-run-dir',str(root/'core'),
                '--output-dir',str(root/'diag'),'--lifecycle-path',str(root/'life.json'),
                '--diagnose-case','2048,131,1,compute','--export-dir',str(root/'public')],
                env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertTrue(capture.is_file(),result.stderr)
            command=json.loads(capture.read_text())
            self.assertEqual(command[command.index('--gpus')+1],'"device=1,3,5,7"')
            self.assertIn('WEAVE_HOST_ALLOCATION=1,3,5,7',command)
            report=json.loads((root/'public/experiment.json').read_text())
            self.assertEqual(report['physical_gpus'],[1,3,5,7])

    def test_snapshot_targets_only_the_scheduler_selected_cards(self):
        self.assertIsNotNone(GPUAllocation)
        from src.weave.calibration import capture_gpu_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            tool=Path(tmp)/'nvidia-smi'
            tool.write_text('#!'+sys.executable+'\nimport sys\n'
                'if sys.argv[1:]!=["-q","-x","-i","1,3,5,7"]: sys.exit(7)\n'
                'print("<nvidia_smi_log><gpu id=\\"bus\\"><uuid>GPU-selected</uuid></gpu></nvidia_smi_log>")\n')
            tool.chmod(0o755)
            with patch.dict(os.environ,{'PATH':tmp+os.pathsep+os.environ['PATH'],'CUDA_VISIBLE_DEVICES':'1,3,5,7'}):
                value=capture_gpu_snapshot()
            self.assertEqual(value['status'],'captured')
            self.assertEqual(value['physical_gpus'],[1,3,5,7])
