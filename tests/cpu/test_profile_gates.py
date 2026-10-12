import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.weave.experiment import validate_tiled_snapshot_metadata,validate_model_evidence
from tests.cpu.test_service_acceptance import snapshot

ROOT=Path(__file__).resolve().parents[2]


def evidence():
    config={'stride':128,'capacity':8,'layers':43}
    gate={'source_sha256':'a'*64,'checkpoint_content_sha256':'c'*64}
    profiles={'profiles':[{'rank':r,'sha256':'b'*64} for r in range(4)]}
    snapshots=[]
    for rank in range(4):
        row=snapshot();row.update(rank=rank,kernel='tiled',source_sha256='a'*64,
            schedule_mode='measured_dynamic',profile_checkpoint_content_sha256='c'*64,
            profile_bank_sha256='b'*64,execution_profile_configuration=config)
        snapshots.append(row)
    return config,gate,profiles,snapshots


class ProfileGatesTest(unittest.TestCase):
    def test_formal_snapshots_cannot_have_active_execution_probes(self):
        config,gate,profiles,snapshots=evidence()
        with self.assertRaisesRegex(ValueError,'configuration'):
            validate_tiled_snapshot_metadata(snapshots,gate,profiles)
        validate_tiled_snapshot_metadata(snapshots,gate,profiles,config)
        snapshots[0]['execution_profile_configuration']=None
        with self.assertRaisesRegex(ValueError,'configuration'):
            validate_tiled_snapshot_metadata(snapshots,gate,profiles,config)

    def test_formal_model_gate_cannot_reuse_instrumented_acceptance(self):
        config,gate,profiles,before=evidence()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'acceptance.json').write_text(json.dumps({'status':'passed','source_sha256':'a'*64,
                'kernel':'tiled','execution_profile_configuration':config}))
            native={'status':'passed','input_sha256':'input','response':{'output_ids':[12]*8},
                    'prepared_manifest':{'record_sha256':'d'*64}}
            (root/'native-request.json').write_text(json.dumps(native))
            for name,graph in [('weave_eager',False),('weave_graph',True)]:
                old=copy.deepcopy(before);new=copy.deepcopy(before)
                for rows in (old,new):
                    for row in rows: row['graph_enabled']=graph
                for row in new:
                    for layer in row['layers'].values():
                        layer['device_calls']=[2,11,0]
                        layer['python_decode_calls']=3 if graph else 11
                (root/f'{name}-request.json').write_text(json.dumps({**native,
                    'coverage_before':old,'coverage_after':new}))
            with self.assertRaisesRegex(ValueError,'configuration'):
                validate_model_evidence(root,gate,profiles)
            result=validate_model_evidence(root,gate,profiles,config)
            self.assertEqual(result['status'],'passed')

    def test_execution_without_measured_bank_fails_before_docker(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_native_ep.py'),
            '--backend','weave_functional','--kernel','tiled','--execution-profile','--execute'],
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,2,4,6'},capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('measured profile bank',result.stderr)
