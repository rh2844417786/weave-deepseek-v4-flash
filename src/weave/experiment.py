"""Required experiment phase ordering and source-specific evidence guards."""
import hashlib
import json
from pathlib import Path

from .calibration import load_verified_profile_bundle
from .runtime_contract import load_contract
from .service_acceptance import validate_coverage_delta

PHASES=('core','calibration','model_acceptance','native_formal','weave_formal','comparison')
ALLOWED={'core':{'passed'},'calibration':{'measured'},'model_acceptance':{'passed'},
         'native_formal':{'complete','deadline','incomplete'},
         'weave_formal':{'complete','deadline','incomplete'},
         'comparison':{'compared','no_common_completed'}}


class ExperimentLedger:
    def __init__(self): self.steps={}
    @property
    def completed(self): return list(self.steps)
    def record(self,name,result):
        if len(self.steps)>=len(PHASES) or PHASES[len(self.steps)]!=name:
            raise ValueError('required experiment phase ordering violated')
        if result.get('status') not in ALLOWED[name]:
            raise ValueError(f'{name} has not completed its required gate: {result.get("status")}')
        self.steps[name]=result
    @property
    def outcome(self):
        result=self.steps.get('comparison')
        if result is None: return 'incomplete'
        if result['status']=='no_common_completed': return 'no_common_completed'
        return 'budgeted_comparison_complete' if result.get('relative_decode_speed') is not None else 'no_decode_metric'


def validate_profile_directory(path,core_gate):
    """Host-side guard; each worker also checks its actual hardware/runtime."""
    path=Path(path);contract=load_contract();result=[]
    expected_runtime={'python':contract['python_version'],'torch':contract['torch_version'],'cuda':contract['torch_cuda']}
    for rank in range(4):
        file=path/f'profile-bank-rank{rank}.json'
        data=json.loads(file.read_text())
        if (data.get('checkpoint_content_sha256')!=core_gate['checkpoint_content_sha256'] or
                data.get('image_id')!=core_gate['image_id'] or
                any(data.get('runtime',{}).get(k)!=v for k,v in expected_runtime.items())):
            raise ValueError('calibration bank differs from verified checkpoint/image/runtime')
        buckets=load_verified_profile_bundle(file,kernel='tiled_mxfp4_tile_v1',
            source_sha256=core_gate['source_sha256'],device_key=data['device_key'],
            hidden_size=4096,intermediate_size=2048,rank=rank,workers=data['workers'])
        result.append({'rank':rank,'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                       'device_key':data['device_key'],'buckets':[b['max_local_tokens'] for b in buckets]})
    return {'status':'measured','profiles':result}


def validate_tiled_snapshot_metadata(snapshots,core_gate,profiles):
    expected={v['rank']:v['sha256'] for v in profiles['profiles']}
    if len(snapshots)!=4 or {v.get('rank') for v in snapshots}!=set(range(4)):
        raise ValueError('four tiled worker snapshot identities required')
    for snapshot in snapshots:
        if (snapshot.get('kernel')!='tiled' or snapshot.get('source_sha256')!=core_gate['source_sha256'] or
                snapshot.get('schedule_mode')!='measured_dynamic' or
                snapshot.get('profile_checkpoint_content_sha256')!=core_gate['checkpoint_content_sha256'] or
                snapshot.get('profile_bank_sha256')!=expected[snapshot['rank']]):
            raise ValueError('snapshot is not current measured tiled execution')


def validate_model_evidence(directory,core_gate,profiles):
    """Recheck saved request/coverage evidence before starting formal timing."""
    directory=Path(directory)
    accepted=json.loads((directory/'acceptance.json').read_text())
    if accepted.get('status')!='passed' or accepted.get('source_sha256')!=core_gate['source_sha256'] or accepted.get('kernel')!='tiled':
        raise ValueError('current tiled full-model acceptance must pass before benchmarking')
    native=json.loads((directory/'native-request.json').read_text())
    if native.get('status')!='passed' or len(native['response']['output_ids'])!=8:
        raise ValueError('native eight-token request evidence required')
    proof={}
    for label,graph in [('weave_eager',False),('weave_graph',True)]:
        request=json.loads((directory/f'{label}-request.json').read_text())
        if (request.get('status')!='passed' or request['input_sha256']!=native['input_sha256'] or
                request['response']['output_ids']!=native['response']['output_ids']):
            raise ValueError('tiled/native request identity and generated tokens must match')
        before,after=request['coverage_before'],request['coverage_after']
        validate_tiled_snapshot_metadata(before,core_gate,profiles)
        validate_tiled_snapshot_metadata(after,core_gate,profiles)
        proof[label]=validate_coverage_delta(before,after,require_graph=graph)
    return {'status':'passed','directory':str(directory),'coverage':proof,
            'record_sha256':native['prepared_manifest']['record_sha256']}
