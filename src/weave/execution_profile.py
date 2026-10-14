"""Bounded device diagnostic snapshots; no Torch dependency in aggregation."""
from dataclasses import dataclass,asdict
import hashlib
from pathlib import Path
import re

FIELD_NAMES=('begin','planned','dispatch_begin','dispatch_end','compute_begin','compute_end',
    'return_begin','return_end','reduce_begin','reduce_end','end_wait_begin','end',
    'dispatch_work','dispatch_wait','compute_work','input_wait','compute_queue',
    'return_work','return_wait','return_queue','reduce_work','reduce_wait','end_wait','plan_wait',
    'tiles','stolen','returns','rows','total','plan_work','samples','reserved')
FIELDS=len(FIELD_NAMES)
PHASES=('prefill','decode')
DURATIONS=tuple(range(12,24))+(28,29)
COUNTS=tuple(range(24,28))


def profiling_source_sha256():
    root=Path(__file__).resolve().parents[2]
    names=('src/weave/execution_profile.py','src/weave/forward_profile.py','src/weave/ops.py','src/weave/sglang_backend.py',
           'src/weave/evidence_server.py','src/weave/sglang_ext/models/deepseek_v4.py',
           'scripts/collect_execution_profile.py')
    return hashlib.sha256(b''.join(name.encode()+b'\0'+(root/name).read_bytes() for name in names)).hexdigest()


@dataclass(frozen=True)
class TraceConfig:
    stride:int=128
    capacity:int=8
    layers:int=43

    def __post_init__(self):
        for name,limit in (('stride',65536),('capacity',64),('layers',43)):
            value=getattr(self,name)
            if type(value) is not int or not 1<=value<=limit:
                raise ValueError(f'execution profile {name} must be1..{limit}')

    def shapes(self,workers):
        if type(workers) is not int or not 1<=workers<=256:
            raise ValueError('execution profile worker count must be1..256')
        planes=2*self.layers
        return ((planes,2+workers*8),(planes,workers,FIELDS),
                (planes,self.capacity,workers+1,FIELDS))


def _nonnegative(value):
    if type(value) is not int or not 0<=value<2**63:
        raise ValueError('execution profile requires nonnegative int64 counters')
    return value


def _row(values,size):
    if not isinstance(values,list) or len(values)!=size:
        raise ValueError('execution profile buffer shape mismatch')
    return [_nonnegative(v) for v in values]


def _worker_totals(rows):
    return {FIELD_NAMES[field]:sum(row[field] for row in rows) for field in DURATIONS}


def _sample(rows,meta,sequence,workers):
    meta=_row(meta,FIELDS)
    if meta[0]!=sequence or not 0<=meta[3]<workers or not 1<=meta[4]<=8:
        raise ValueError('execution profile sample sequence or plan mismatch')
    rows=[_row(row,FIELDS) for row in rows]
    for worker,row in enumerate(rows):
        if row[30]!=1 or row[0]<=0 or row[11]<row[0] or row[28]!=row[11]-row[0]:
            raise ValueError('incomplete or nonmonotonic execution profile sample')
        if any(row[field]==0 for field in (1,4,5,6,7,8,9,10)):
            raise ValueError('execution profile mandatory stage timestamp missing')
        if any(row[j]>row[11] or (row[j] and row[j]<row[0]) for j in range(1,11)):
            raise ValueError('execution profile timestamp outside worker interval')
        if any(row[field]>row[28] for field in DURATIONS):
            raise ValueError('execution profile stage exceeds worker interval')
        ordered=[value for value in row[:12] if value]
        if ordered!=sorted(ordered):
            raise ValueError('execution profile stage order is nonmonotonic')
        dispatched=worker<meta[3] or meta[3]==0
        if bool(row[2])!=dispatched or bool(row[3])!=dispatched:
            raise ValueError('execution profile dispatch role mismatch')
        if (row[25]>row[24] or (worker>=meta[3] and row[25])
                or (worker<meta[3] and (row[25]>meta[5] or row[24]!=row[25]))):
            raise ValueError('execution profile stealing count or budget mismatch')
        if sum(row[12:24])>row[28] or row[29]>row[23]:
            raise ValueError('execution profile worker time accounting inconsistent')
    if any(sum(row[field] for row in rows)!=meta[index] for field,index in ((24,12),(26,10),(27,2))):
        raise ValueError('execution profile tile/return/row counts are not conserved')
    begin=min(row[0] for row in rows)
    envelopes={};intervals={}
    for name,left,right,count in (('dispatch',2,3,None),('compute',4,5,24),
                                  ('return',6,7,26),('reduce',8,9,27),('completion_wait',10,11,None)):
        active=[row for row in rows if row[left] and (count is None or row[count])]
        if any(row[right]<row[left] for row in active):
            raise ValueError('execution profile phase ends before it starts')
        intervals[name]=[min(row[left] for row in active)-begin,max(row[right] for row in active)-begin] if active else None
        envelopes[name]=intervals[name][1]-intervals[name][0] if active else 0
    return {'sequence':sequence,'epoch':meta[1],
        'schedule':{'c':meta[3],'K':meta[4],'steal':meta[5]},
        'workload':dict(zip(('local_rows','payloads','local_pairs','remote_pairs','remote_unique',
                            'tasks','groups','tile_jobs'),[meta[j] for j in (2,6,7,8,9,10,11,12)])),
        'kernel_span_ns':max(row[11] for row in rows)-begin,
        'phase_envelopes_ns':envelopes,'phase_intervals_ns':intervals,'worker_totals_ns':_worker_totals(rows),
        'worker_counts':{FIELD_NAMES[field]:sum(row[field] for row in rows) for field in COUNTS}}


def summarize_profile(config,identity,heads,totals,records,*,workers):
    """Aggregate synchronized buffers; envelopes include gaps and may overlap."""
    shapes=config.shapes(workers)
    allowed={'rank','source_sha256','profile_bank_sha256','checkpoint_content_sha256',
             'kernel','graph_enabled','device_key','image_id','host_gpu_allocation',
             'schedule_mode','fixed_plan'}
    if set(identity)-allowed:
        raise ValueError('execution profile identity cannot override diagnostic fields')
    for value,shape in zip((heads,totals,records),shapes):
        if not isinstance(value,list) or len(value)!=shape[0]:
            raise ValueError('execution profile plane count mismatch')
    if (type(identity.get('rank')) is not int or not 0<=identity['rank']<4
            or identity.get('kernel')!='tiled' or type(identity.get('graph_enabled')) is not bool):
        raise ValueError('execution profile identity missing')
    fixed=identity.get('schedule_mode','measured_dynamic')=='explicit_fixed'
    if identity.get('schedule_mode','measured_dynamic') not in ('explicit_fixed','measured_dynamic'):
        raise ValueError('unknown execution profile schedule mode')
    plan=identity.get('fixed_plan')
    if fixed:
        if (not isinstance(plan,list) or len(plan)!=3 or any(type(v) is not int for v in plan)
                or not 0<=plan[0]<workers or not 1<=plan[1]<=8 or plan[2]<0
                or identity.get('profile_bank_sha256') is not None or identity['graph_enabled']):
            raise ValueError('fixed profile requires an explicit bounded plan, no bank and eager execution')
    elif plan is not None:
        raise ValueError('measured profile cannot claim a fixed plan')
    fields=('source_sha256','checkpoint_content_sha256')+(() if fixed else ('profile_bank_sha256',))
    for field in fields:
        if not re.fullmatch('[0-9a-f]{64}',identity.get(field,'')):
            raise ValueError('execution profile source/profile/checkpoint identity missing')
    result={'schema_version':1,'status':'profiled','measurement_mode':'instrumented_diagnostic',
            'profiling_source_sha256':profiling_source_sha256(),
            'used_for_formal_performance':False,'clock':'device_globaltimer','units':'nanoseconds',
            'scope':'persistent_worker_entry_through_completion_barrier_excludes_prepare_and_outer_runtime',
            'aggregation':'worker totals are CTA wall time; concurrent phase envelopes are not additive latency or hardware utilization; plan_wait includes planner work on CTA0',
            'sampling_window':'since explicit reset through idle snapshot; includes warmups and abort/drain completion',
            **identity,'schedule_mode':'explicit_fixed' if fixed else 'measured_dynamic',
            'configuration':{'stride':config.stride,'capacity':config.capacity,
                                        'layers':config.layers,'workers':workers},'layers':{}}
    for phase_index,phase in enumerate(PHASES):
        for layer in range(config.layers):
            plane=phase_index*config.layers+layer
            head=_row(heads[plane],shapes[0][1]);calls,sampled=head[:2]
            expected=(calls+config.stride-1)//config.stride
            if sampled!=expected or sum(head[2:])!=sampled:
                raise ValueError('execution profile sample counters or plan histogram inconsistent')
            if len(totals[plane])!=workers or len(records[plane])!=config.capacity:
                raise ValueError('execution profile worker or ring shape mismatch')
            worker_rows=[_row(row,FIELDS) for row in totals[plane]]
            if any(row[30]!=sampled for row in worker_rows):
                raise ValueError('execution profile incomplete worker totals')
            if any(sum(row[12:24])>row[28] or row[25]>row[24] or row[29]>row[23] for row in worker_rows):
                raise ValueError('execution profile cumulative accounting inconsistent')
            recent=[]
            for index in range(max(0,sampled-config.capacity),sampled):
                slot=index%config.capacity
                rows=records[plane][slot]
                if len(rows)!=workers+1:
                    raise ValueError('execution profile sample worker shape mismatch')
                metadata=_row(rows[-1],FIELDS)
                if metadata[13]!=phase_index or metadata[14]!=layer:
                    raise ValueError('execution profile sample phase/layer mismatch')
                sample=_sample(rows[:-1],rows[-1],index*config.stride,workers)
                if fixed and sample['schedule']!={'c':plan[0],
                        'K':min(plan[1],max(1,sample['workload']['payloads'])),'steal':plan[2]}:
                    raise ValueError('sampled schedule differs from fixed plan')
                recent.append(sample)
            if fixed and any(n and (i//8!=plan[0] or i%8+1>plan[1]) for i,n in enumerate(head[2:])):
                raise ValueError('plan histogram differs from fixed plan')
            for worker,total in enumerate(worker_rows):
                if any(total[field]<sum(records[plane][index%config.capacity][worker][field]
                        for index in range(max(0,sampled-config.capacity),sampled)) for field in range(12,31)):
                    raise ValueError('execution profile totals omit retained samples')
            result['layers'].setdefault(str(layer),{})[phase]={
                'calls':calls,'sampled_invocations':sampled,'retained_samples':len(recent),
                'overwritten_samples':max(0,sampled-config.capacity),
                'plan_histogram':[{'c':i//8,'K':i%8+1,'sampled_count':n}
                                  for i,n in enumerate(head[2:]) if n],
                'worker_totals_ns':_worker_totals(worker_rows),
                'worker_counts':{FIELD_NAMES[field]:sum(row[field] for row in worker_rows) for field in COUNTS},
                'recent_samples':recent}
    return result


def add_profile_arguments(parser):
    parser.add_argument('--execution-profile',action='store_true',help='sampled device diagnostic; excluded from formal performance claims')
    parser.add_argument('--execution-profile-stride',type=int,default=128)
    parser.add_argument('--execution-profile-capacity',type=int,default=8)


def profile_options(args):
    config=TraceConfig(stride=args.execution_profile_stride,capacity=args.execution_profile_capacity)
    return (['--execution-profile','--execution-profile-stride',str(config.stride),
             '--execution-profile-capacity',str(config.capacity)] if args.execution_profile else [])


def profile_configuration(args):
    return asdict(TraceConfig(stride=args.execution_profile_stride,capacity=args.execution_profile_capacity)) if args.execution_profile else None
