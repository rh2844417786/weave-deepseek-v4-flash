"""Source-bound graph/control evidence for the bounded fixed-prefix probe."""
import json
from pathlib import Path
from .graph_evidence import validate_graph_delta
from .native_p0_pair import normalize_launch
from .fixed_prefix_data import digest

ARMS=('native_a','native_b','p0_full','p0_events','p0_buffers')


def normalize_fixed_launch(argv):
    result=[];index=0
    while index<len(argv):
        if argv[index]=='-e' and index+1<len(argv):
            key=argv[index+1].split('=',1)[0]
            if key in ('WEAVE_NATIVE_P0_MODE','WEAVE_NATIVE_P0_READBACK'):
                index+=2;continue
            if key=='WEAVE_FIXED_PREFIX_DIR':
                result.extend(['-e','WEAVE_FIXED_PREFIX_DIR=<arm-private>']);index+=2;continue
        result.append(argv[index]);index+=1
    return normalize_launch(result)


def validate_fixed_graph(before,after,*,arm,steps,identity):
    if arm not in ARMS or steps not in (2,8):raise ValueError('unknown fixed-prefix arm/steps')
    result=validate_graph_delta(before,after,expected_observer_source=identity['observer_source_sha256'],expected_sources=identity['graph_sources'])
    if result['replays']!=steps-1 or any(row['decode_batch_histogram']!={'32':steps-1} for row in result['ranks'].values()):
        raise ValueError('exact constant B32 decode trajectory required')
    enabled=arm.startswith('p0_');mode=arm.split('_',1)[1] if enabled else None
    evidence={}
    for old,new in zip(sorted(before,key=lambda r:r['rank']),sorted(after,key=lambda r:r['rank'])):
        for row in (old,new):
            p=row.get('stage_profile')
            if not enabled:
                if p is not None:raise ValueError('native arm unexpectedly has P0 profiler')
                continue
            if not isinstance(p,dict) or p.get('kind')!='native_p0' or p.get('profile_source_sha256')!=identity['native_p0_source_sha256']:
                raise ValueError('fixed-prefix P0 source differs')
            expected=dict(mode=mode,readback=False,event_node_count=0 if mode=='buffers' else 1032,
                          allocated_event_count=0 if mode=='buffers' else 1032,
                          retained_grouping_layer_count=0 if mode=='events' else 43,
                          retained_grouping_tensor_ref_count=0 if mode=='events' else 129)
            if p.get('intervention')!=expected or p.get('sampled_total')!=0 or p.get('aborted_samples')!=0:
                raise ValueError('actual P0 intervention/resources differ')
            if p.get('b32_seen')!=row['raw_batch_histogram'].get('32',0):raise ValueError('P0 actual B32 counts differ')
            metadata=p.get('layer_metadata',{})
            if set(metadata)!={str(i) for i in range(43)} or any(not v.get('captured_path') for v in metadata.values()):
                raise ValueError('actual P0 capture path coverage incomplete')
        if enabled:evidence[str(new['rank'])]=new['stage_profile']['intervention']
    result.update(arm=arm,steps=steps,p0_intervention=evidence,numerical_policy='record_only')
    return result


def validate_phase_proofs(private_dir,*,phase,steps,config,teacher,identity):
    """Validate proof identity on the host; actual tensor bytes checked in image."""
    result={};pins=json.loads((Path(__file__).resolve().parents[2]/'configs/fixed_prefix_sources.json').read_text())['sha256']
    for rank in range(4):
        path=Path(private_dir)/f'{phase}-rank{rank}.json'
        if path.is_symlink():raise ValueError('phase proof symlink forbidden')
        proof=json.loads(path.read_text())
        expected=dict(schema_version=1,status='passed',phase=phase,rank=rank,world_size=4,
            numerical_policy='record_only',expected_steps=steps,completed_steps=list(range(steps)),
            image_id=identity['image_id'],fixed_prefix_source_sha256=identity['fixed_prefix_source_sha256'],
            config_sha256=digest(config),selection_sha256=config['selection_sha256'],native_sources=pins,
            teacher_sha256=None if phase=='seed' else teacher['sha256'],disable_overlap_schedule=True,
            stage_events_inserted_by_fixed_prefix_runtime=0,logical_kv_positions_enforced=True,
            kv_numerical_contents_claimed_identical=False)
        if any(proof.get(k)!=v for k,v in expected.items()):raise ValueError('fixed-prefix phase identity or coverage differs')
        rows=proof.get('records',[])
        if len(rows)!=steps or [r.get('step') for r in rows]!=list(range(steps)):raise ValueError('fixed-prefix phase step records incomplete')
        for step,row in enumerate(rows):
            order=row.get('row_slot_order',[])
            if (sorted(order)!=list(range(32)) or row.get('batch_size')!=32 or row.get('original_logits_finite')is not True
                    or row.get('captured')is not False or row.get('mode')!=('extend' if step==0 else 'decode')):
                raise ValueError('actual fixed-prefix model inputs malformed')
            lengths=[config['requests'][i]['input_tokens']+step for i in order]
            if row.get('seq_lens')!=lengths or row.get('positions')!=[n-1 for n in lengths]:raise ValueError('logical KV positions differ')
            if phase!='seed':
                selected=[teacher['output_ids'][i][step] for i in order]
                if row.get('sampled_ids_sha256')!=digest(selected):raise ValueError('forced token input not actually used')
        if phase=='trace':
            trace=proof.get('launch_trace',{})
            if trace.get('status')!='complete' or trace.get('marlin_kernel_count')!=86 or trace.get('rank')!=rank:
                raise ValueError('complete native decode kernel trace required')
        result[str(rank)]=proof
    return result
