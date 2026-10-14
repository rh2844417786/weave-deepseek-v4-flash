"""CPU-only persistence/aggregation for frozen-trace compute controls."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from .calibration import atomic_json
from .experiment_export import ensure_public

VARIANTS=('persistent_raw_no_probe','persistent_raw_probe','persistent_bound_no_probe',
          'staged_132','staged_264','staged_528','staged_direct')
PRIVATE_FIELDS={'x','ids','routing','group_slots','group_meta','slot_outputs','outputs','activations'}


def public_replay(value):
    ensure_public(value)
    def check(item):
        if isinstance(item,dict):
            if PRIVATE_FIELDS.intersection(item):raise ValueError('private replay tensor field cannot be exported')
            for child in item.values():check(child)
        elif isinstance(item,list):
            for child in item:check(child)
    check(value)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def point_key(point):
    if (type(point.get('layer')) is not int or point['layer'] not in range(43)
            or type(point.get('sample')) is not int or point['sample'] not in (0,1) or point.get('variant') not in VARIANTS):
        raise ValueError('invalid replay point identity')
    return f'{point["layer"]}:{point["sample"]}:{point["variant"]}'


def validate_point(point,repetitions):
    point_key(point);public_replay(point)
    if set(point)-{'layer','sample','variant','gpu_ms','host_enqueue_ms','correctness','workload',
                  'stage_gpu_ms','warmups','probe_summary','host_setup_ms'}:
        raise ValueError('unknown public point fields')
    for name in ('gpu_ms','host_enqueue_ms'):
        values=point.get(name)
        if (not isinstance(values,list) or len(values)!=repetitions
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values)):
            raise ValueError('finite positive samples for every repetition required')
    correctness=point.get('correctness',{})
    error=correctness.get('relative_l2')
    if (correctness.get('status')!='passed' or type(error) not in (float,int)
            or not math.isfinite(error) or not 0<=error<=.025):
        raise ValueError('real-trace correctness must pass before timing is saved')
    stages=point.get('stage_gpu_ms',{})
    if set(stages)-{'gate_up','down','reduce'}:raise ValueError('unknown staged timer')
    for samples in stages.values():
        if not samples or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in samples):
            raise ValueError('invalid supplemental stage samples')


class ReplayStore:
    def __init__(self,path,identity):
        self.path=Path(path);self.identity=identity
        if self.path.exists():
            self.value=json.loads(self.path.read_text())
            if self.value.get('identity')!=identity:raise ValueError('replay identity differs; use a fresh output directory')
            if self.value.get('points_sha256')!=digest(self.value.get('points')):
                raise ValueError('replay points digest mismatch')
            for point in self.value['points']:validate_point(point,identity['repetitions'])
            if len(self.keys())!=len(self.value['points']):raise ValueError('duplicate saved replay point')
        else:
            self.value={'schema_version':1,'status':'partial','identity':identity,'points':[],
                        'orphaned_points':[],'measurement_mode':'frozen_trace_microbenchmark',
                        'used_for_formal_performance':False}
            self.save()

    def keys(self):return {point_key(point) for point in self.value['points']}

    def save(self):
        self.value['points_sha256']=digest(self.value['points']);public_replay(self.value)
        atomic_json(self.path,self.value)

    def reconcile(self,common):
        self.value['orphaned_points'] += [p for p in self.value['points'] if point_key(p) not in common]
        self.value['points']=[p for p in self.value['points'] if point_key(p) in common]
        self.value['status']='partial';self.save()

    def append(self,point):
        validate_point(point,self.identity['repetitions'])
        if point_key(point) in self.keys():raise ValueError('duplicate replay point')
        self.value['points'].append(json.loads(json.dumps(point,allow_nan=False)));self.save()


def paired_summary(ranks):
    if len(ranks)!=4 or {r['identity']['rank'] for r in ranks}!={0,1,2,3}:raise ValueError('four rank records required')
    indexes=[{point_key(p):p for p in row['points']} for row in ranks]
    if any(set(index)!=set(indexes[0]) for index in indexes):raise ValueError('rank point sets differ')
    cases={}
    for key in sorted(indexes[0]):
        points=[index[key] for index in indexes];first=points[0]
        case=cases.setdefault((first['layer'],first['sample']),{'layer':first['layer'],'sample':first['sample'],'variants':{}})
        medians=[statistics.median(p['gpu_ms']) for p in points]
        case['variants'][first['variant']]={'max_rank_median_gpu_ms':max(medians),
            'rank_median_gpu_ms':{str(r['identity']['rank']):value for r,value in zip(ranks,medians)},
            'rank_median_host_enqueue_ms':{str(r['identity']['rank']):statistics.median(p['host_enqueue_ms']) for r,p in zip(ranks,points)}}
    return {'schema_version':1,'measurement_mode':'frozen_trace_microbenchmark','used_for_formal_performance':False,
            'comparison_note':'matched layer/sample; persistent includes IPC/protocol; staged excludes them; not serving speedup',
            'timing_note':'max of per-rank medians, not cross-GPU clock alignment or a single global elapsed interval',
            'cases':list(cases.values())}
