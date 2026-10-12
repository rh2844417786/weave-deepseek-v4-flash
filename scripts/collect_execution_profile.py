"""Collect source-bound idle localhost diagnostics before stopping the service."""
import json
from pathlib import Path
import time

from scripts.check_sglang_service import http_json
from src.weave.calibration import atomic_json
from src.weave.experiment_export import ensure_public
from src.weave.execution_profile import profiling_source_sha256


def reset_profiles(base):
    for rank in range(4):
        value=http_json(f'http://127.0.0.1:{base+rank}/execution-profile/reset',{},timeout=30)
        if value!={'status':'reset','rank':rank}:
            raise ValueError('four execution profile reset acknowledgements required')
    return {'status':'reset','ranks':[0,1,2,3]}


def wait_idle(url,timeout=30):
    """An HTTP200 flush on the standalone service establishes no active request."""
    deadline=time.monotonic()+timeout;error=None
    while time.monotonic()<deadline:
        try:
            value=http_json(url+'/flush_cache',timeout=min(5,deadline-time.monotonic()))
            if value is False: raise RuntimeError('service declined idle cache flush')
            return
        except Exception as exc:
            error=exc
            time.sleep(min(.2,max(0,deadline-time.monotonic())))
    raise TimeoutError(f'service did not become idle for execution profile snapshot: {error}')


def collect_profiles(base,destination,config,core_gate,profiles,allocation,expected_profiler_source=None):
    expected={row['rank']:row for row in profiles['profiles']}
    values=[]
    expected_profiler_source=expected_profiler_source or profiling_source_sha256()
    for rank in range(4):
        value=http_json(f'http://127.0.0.1:{base+rank}/execution-profile',timeout=30)
        bound=expected[rank]
        workers=int(bound['device_key'].rsplit(';',1)[1])
        if (value.get('status')!='profiled' or value.get('rank')!=rank
                or value.get('profiling_source_sha256')!=expected_profiler_source
                or value.get('source_sha256')!=core_gate['source_sha256']
                or value.get('checkpoint_content_sha256')!=core_gate['checkpoint_content_sha256']
                or value.get('image_id')!=core_gate['image_id']
                or value.get('profile_bank_sha256')!=bound['sha256']
                or value.get('device_key')!=bound['device_key']
                or value.get('host_gpu_allocation')!=allocation
                or value.get('kernel')!='tiled' or value.get('graph_enabled') is not True
                or value.get('measurement_mode')!='instrumented_diagnostic'
                or value.get('used_for_formal_performance') is not False
                or value.get('configuration')!={'stride':config.stride,'capacity':config.capacity,
                                              'layers':config.layers,'workers':workers}):
            raise ValueError('execution profile source/checkpoint/bank/device/config identity differs from passed gates')
        if set(value.get('layers',{}))!={str(n) for n in range(config.layers)}:
            raise ValueError('all profiled layer identities required')
        for layer in value['layers'].values():
            decode=layer.get('decode',{})
            if (type(decode.get('calls')) is not int or decode['calls']<=0
                    or type(decode.get('sampled_invocations')) is not int or decode['sampled_invocations']<=0):
                raise ValueError('every layer needs real decode profile samples')
        ensure_public(value);values.append(value)
    # Publish only after all four snapshots pass their provenance gate.
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for value in values:
        atomic_json(destination/f'execution-profile-rank{value["rank"]}.json',value)
    return {'status':'profiled','ranks':[0,1,2,3],'directory':str(destination),
            'used_for_formal_performance':False,
            'stride':config.stride,'capacity':config.capacity}
