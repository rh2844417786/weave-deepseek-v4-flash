import copy
from src.weave.stall_analysis import analyze_stalls
from src.weave.stall_client import StallClient


def fixture():
    ranks=[]
    for r in range(4):
        ranks.append(dict(rank=r,device=r,source_sha256='a'*64,native_source_sha256='b'*64,source_verified=True,
                          identity={'image':'x','candidate':'y'},clock='CLOCK_MONOTONIC',anchor={'host_before_ns':0,'host_after_ns':100},
                          failures=[],overflow={'calls':False,'samples':False},expected_layers=43,coverage={'prefill':[1]*43,'decode':[0]*43},
                          calls=[dict(sequence=0,phase='prefill',rows=2,host_start_ns=0,host_end_ns=20000000,sample_index=0)],
                          samples=[dict(sequence=0,phase='prefill',rows=2,gpu_start_ms=0,gpu_end_ms=10+r,gpu_duration_ms=10+r,
                                        routed=[dict(layer=i,start_ms=i/43,end_ms=(i+1)/43,duration_ms=1/43) for i in range(43)])]))
    client=StallClient().report('passed',[])
    client['requests']=[dict(index=i,start_ns=0,first_ns=0,last_ns=20000000,count=2,finished=False,server_timing={}) for i in range(32)]
    client['top_gaps']=[dict(start_ns=0,end_ns=20000000,delta_ns=20000000,index=0,after_token_position=1,before_token_position=2)]
    return ranks,client,dict(started_monotonic=0,deadline_monotonic=1)


def test_gap_union_and_rank_max():
    a=analyze_stalls(*fixture())
    assert a['status']=='passed'
    assert a['aligned_samples'][0]['gpu_ms']['max']==13
    assert a['gap_overlap'][0]['approximate_prefill_overlap_ns'] <= 13000100
    assert a['request_stages']['queue']['missing']==32


def test_missing_layer_and_rank_drift():
    args=fixture(); args[0][0]['samples'][0]['routed'].pop()
    assert analyze_stalls(*args)['status']=='failed'
    args=fixture(); args[0][0]['samples'][0]['rows']=4
    assert analyze_stalls(*args)['status']=='failed'


def test_client_privacy_and_bound():
    c=StallClient(max_requests=1); c.start(0,1)
    c.observe(0,{'text':'secret','meta_info':{'completion_tokens':1,'weave_stall_timing':{'created_time':1,'prompt':'secret'}}},2)
    c.observe(0,{'meta_info':{'completion_tokens':2,'finish_reason':{'type':'length'}}},3)
    assert 'secret' not in str(c.report('passed',[]))
    assert c.gaps[0]['after_token_position']==1
    try: c.start(1,4)
    except ValueError: pass
    else: assert False


def test_arm_all_ranks_in_order():
    import asyncio
    from src.weave.stall_client import arm_ranks
    class Response:
        status=200
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def json(self): return {'status':'armed'}
    class Session:
        def __init__(self): self.calls=[]
        def post(self,url,**kwargs): self.calls.append((url,kwargs)); return Response()
    session=Session(); asyncio.run(arm_ranks(session,30210))
    assert [x[0] for x in session.calls]==[f'http://127.0.0.1:{30210+i}/arm' for i in range(4)]
    assert all(x[1]=={'json':{},'allow_redirects':False} for x in session.calls)


def test_invalid_and_missing_timing():
    args=fixture(); args[0][0]['samples'][0]['gpu_end_ms']=float('nan')
    assert analyze_stalls(*args)['status']=='failed'
    args=fixture(); args[1]['requests'][0]['server_timing']={'wait_queue_entry_time':2,'forward_entry_time':1}
    assert analyze_stalls(*args)['status']=='failed'


def test_phase_and_window_and_privacy():
    args=fixture(); args[2]['started_monotonic']=0.005
    for request in args[1]['requests']: request.update(start_ns=5000000,first_ns=5000000)
    args[1]['top_gaps'][0].update(start_ns=5000000,delta_ns=15000000)
    a=analyze_stalls(*args)
    assert a['per_rank'][0]['model_minus_routed_union_ms']['prefill']==9
    assert a['routed_layer_gpu_ms']['prefill'][0]['max']==1/43
    assert a['approximate_window_sampled_phase_gpu_ms']['prefill']['max'] < 9
    assert a['request_stage_observability']['queue']=='missing'
    args[1]['requests'][0]['prompt']='PRIVATE'
    assert analyze_stalls(*args)['status']=='failed'


def test_measurement_arm_after_warmup_and_failure_artifact():
    import asyncio, tempfile, json
    from pathlib import Path
    from unittest.mock import patch
    from src.weave import c32_serving as client
    from tests.cpu.test_c32_serving import FakeSession
    class Workload:
        prompts=['private']*100
        identity={'dataset_sha256':'a'*64,'request_order_sha256':'b'*64}
    order=[]
    async def warmup(*args): order.append('warmup'); return {}
    async def arm(*args):
        order.append('arm')
        raise ValueError('stall_arm_failed')
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(client,'_warmup',warmup), patch('src.weave.stall_client.arm_ranks',arm):
            result=asyncio.run(client.run_measurement(FakeSession(),'http://127.0.0.1:3',Workload(),directory,seconds=.1,stall_port_base=30210))
        assert order==['warmup','arm']
        report=Path(directory)/'client-stall.json'
        assert json.loads(report.read_text())['status']=='failed'
        assert report.stat().st_mode & 0o777 == 0o664
        assert result['error_codes']==['stall_arm_failed']


def test_rejects_duplicate_and_orphan_samples():
    for orphan in (False,True):
        args=fixture()
        for rank in args[0]:
            extra=copy.deepcopy(rank['samples'][0])
            if orphan: extra['sequence']=99
            rank['samples'].append(extra)
        assert analyze_stalls(*args)['status']=='failed'


def test_rejects_incomplete_client_contract_and_request_gap_inconsistency():
    for bad in ({'clock':'CLOCK_MONOTONIC'}, dict(fixture()[1],requests=[])):
        args=fixture()
        assert analyze_stalls(args[0],bad,args[2])['status']=='failed'
    for mutation in ('index','timestamp','position','count','contract'):
        args=fixture()
        if mutation=='index': args[1]['top_gaps'][0]['index']=99
        elif mutation=='timestamp': args[1]['top_gaps'][0]['start_ns']=-1
        elif mutation=='position': args[1]['top_gaps'][0]['before_token_position']=3
        elif mutation=='count': args[1]['requests'][0]['count']=1025
        else: args[1]['used_for_formal_performance']=True
        assert analyze_stalls(*args)['status']=='failed'


def test_direct_duration_costs_and_required_finite_fields():
    args=fixture()
    for rank in args[0]:
        sample=rank['samples'][0]
        sample.update(gpu_start_ms=600000.,gpu_end_ms=600000.,gpu_duration_ms=1.)
        for slot in sample['routed']:
            slot.update(start_ms=600000.,end_ms=600000.,duration_ms=.02)
    result=analyze_stalls(*args)
    assert result['status']=='passed'
    assert result['sampled_phase_gpu_ms']['prefill']['max']==1.
    assert result['routed_layer_gpu_ms']['prefill'][0]['max']==.02
    assert abs(result['per_rank'][0]['model_minus_routed_union_ms']['prefill']-.14)<1e-12
    for field in ('gpu_duration_ms','duration_ms'):
        for value in (None,float('nan'),float('inf'),-1):
            bad=copy.deepcopy(args); target=bad[0][0]['samples'][0]
            if field=='duration_ms': target=target['routed'][0]
            if value is None: del target[field]
            else: target[field]=value
            assert analyze_stalls(*bad)['status']=='failed'


def test_routed_slots_must_be_sequential_and_nonoverlapping():
    for mutation in ('order','overlap'):
        args=fixture(); slots=args[0][0]['samples'][0]['routed']
        if mutation=='order': slots[0],slots[1]=slots[1],slots[0]
        else: slots[1]['start_ms']=0
        assert analyze_stalls(*args)['status']=='failed'


import unittest
class StallAnalysisTests(unittest.TestCase):
    pass
for _name, _fn in list(globals().items()):
    if _name.startswith("test_"):
        setattr(StallAnalysisTests, _name, lambda self, fn=_fn: fn())

del _fn
