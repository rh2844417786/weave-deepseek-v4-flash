"""Measured chunk DAG model for native_marlin_peer_tail_v1 (microseconds).

This table is independent of the old CTA surfaces. Interference and task-boundary
costs come from executions of this executor. The bounded joint search retains two
local candidates per rank/chunk count; it is an approximation, never an optimality
claim. CPU code is also the independent oracle for the captured GPU planner.
"""
from dataclasses import asdict, dataclass
from itertools import product
import hashlib
import json
import math
from pathlib import Path

METHOD='native_marlin_peer_tail_v1'
POLICY='measured_tail_dag_v1'
GENERATE='python3 scripts/run_native_tail.py --work-root /srv/weave-work --records <prepared/main.jsonl> --execute'
FEATURES=('local_pairs','active_experts','native_groups','sum_squared_rows','max_rows')
SCALARS=('front_us','prepare_us','planner_us','finish_us','role_switch_us')
STAGES=('down_us','finalize_us','fetch_reduce_us','fetch_result_us')
MAX_PLANS=22
ANCHORS=4
ROW_SIZE=26  # five features, five scalars, four stages x four chunks

@dataclass(frozen=True)
class Plan:
    communication_workers:int
    chunks:int
    steal_tiles:int
    serial:bool=False

    def __post_init__(self):
        if (type(self.communication_workers) is not int or self.communication_workers<0
            or self.chunks not in (1,2,4) or type(self.steal_tiles) is not int
            or self.steal_tiles<0 or type(self.serial) is not bool
            or (self.serial and (self.communication_workers,self.chunks,self.steal_tiles)!=(0,1,0))
            or (not self.serial and self.communication_workers<1)
            or (self.communication_workers<2 and self.steal_tiles)):
            raise ValueError('unsupported executable tail plan')

    def to_dict(self):return asdict(self)


def candidates(workers,hidden=4096,n_tile=128):
    if type(workers) is not int or workers<2 or hidden<=0 or n_tile<=0 or hidden%n_tile:
        raise ValueError('tail needs resident workers and aligned native N tiles')
    result=[Plan(0,1,0,True)]
    for chunks in (1,2,4):
        if hidden%(chunks*4*n_tile):continue
        for comm in (1,2,4,8):
            if comm>=workers:continue
            for steal in ((0,) if comm==1 else (0,1)):
                result.append(Plan(comm,chunks,steal))
    if len(result)<2:raise ValueError('no aligned resident tail candidates')
    return result


def _number(v):return type(v) in (int,float) and math.isfinite(v) and v>=0


def validate_surface(surface,plans,total_rows):
    if not surface or len(surface)!=len(plans):
        raise ValueError('matching measured tail table required; generate with '+GENERATE)
    if type(total_rows) is not int or total_rows<1:raise ValueError('positive routed capacity required')
    for p,anchors in zip(plans,surface):
        if len(anchors)<2 or len(anchors)>ANCHORS:raise ValueError('two to four measured anchors per candidate required')
        for a in anchors:
            f=a.get('features',[])
            if (a.get('plan')!=p.to_dict() or len(f)!=5 or any(not _number(v) for v in f)
                or f[0]>total_rows or f[2]>total_rows or f[3]>total_rows**2 or f[4]>f[0]):
                raise ValueError('malformed tail anchor workload/plan')
            if any(not _number(a.get(k)) for k in SCALARS):raise ValueError('finite measured scalar costs required')
            for k in STAGES:
                values=a.get(k,[])
                if len(values)!=p.chunks or any(not _number(v) for v in values):
                    raise ValueError('complete finite measured chunk stages required')
            if sum(a['down_us'])+sum(a['fetch_reduce_us'])<=0:
                raise ValueError('measured execution durations are missing; '+GENERATE)
    return surface


def dag_predict(rows):
    """Estimated common-start DAG, not an observed cross-GPU clock timeline.

    Workload exchange joins rank arrivals. Down/finalize is local and sequential
    by channel chunk; each owner's reduction waits for all source publications.
    A rank's gather waits for all owner shards. Per-rank communication service is
    serialized conservatively between chunks. Durations already include observed
    concurrent interference under the calibrated local plan.
    """
    if len(rows)!=4:raise ValueError('four rank curves required')
    chunks=rows[0]['plan']['chunks']
    if any(r['plan']['chunks']!=chunks for r in rows):raise ValueError('joint plan chunk mismatch')
    # Planner already includes waiting for peer front arrivals. Sum each rank's
    # contiguous prefix before the cross-rank maximum; summing phase maxima
    # would count that arrival skew twice.
    start=max(r['front_us']+r['planner_us']+r['prepare_us'] for r in rows)
    ready=[[start] for r in rows]
    comm=[start]*4
    for c in range(chunks):
        for rank,r in enumerate(rows):
            ready[rank].append(ready[rank][-1]+r['down_us'][c]+r['finalize_us'][c])
        published=max(v[-1] for v in ready)
        owner=[max(published,comm[rank])+r['fetch_reduce_us'][c]
               +(r['role_switch_us'] if c==0 else 0.) for rank,r in enumerate(rows)]
        all_owners=max(owner)
        comm=[max(all_owners,owner[rank])+r['fetch_result_us'][c] for rank,r in enumerate(rows)]
    return max(comm[rank]+r['finish_us'] for rank,r in enumerate(rows))


def interpolate(anchors,features,total_rows):
    scales=(total_rows,64,total_rows,total_rows**2,total_rows)
    def distance(a):
        d=0.
        for j in range(5):d+=abs(a['features'][j]-features[j])/scales[j]
        return d
    near=sorted(range(len(anchors)),key=lambda i:(distance(anchors[i]),i))[:2]
    a,b=[anchors[i] for i in near]; da,db=distance(a),distance(b)
    def mix(x,y):return (x+y)/2 if db==0 else x if da==0 else (x*db+y*da)/(da+db)
    result={'plan':a['plan'],'features':list(features)}
    result.update({k:mix(a[k],b[k]) for k in SCALARS})
    result.update({k:[mix(x,y) for x,y in zip(a[k],b[k])] for k in STAGES})
    return result


def select_joint(surfaces,features,plans,total_rows):
    if len(surfaces)!=4 or len(features)!=4:raise ValueError('four rank measured surfaces required')
    for surface in surfaces:validate_surface(surface,plans,total_rows)
    curves=[[interpolate(a,f,total_rows) for a in s] for s,f in zip(surfaces,features)]
    def local(row):
        value=0.
        for k in SCALARS:value+=row[k]
        for k in STAGES:
            for x in row[k]:value+=x
        return value
    options=[]
    # Every serial and homogeneous measured plan remains a candidate. Mixed
    # candidates are bounded to 2^4 for each shared chunk count.
    options.extend(tuple([i]*4) for i in range(len(plans)))
    for chunks in (1,2,4):
        indices=[i for i,p in enumerate(plans) if p.chunks==chunks and not p.serial]
        if not indices:continue
        shortlist=[sorted(indices,key=lambda i:(local(curves[r][i]),i))[:2] for r in range(4)]
        options.extend(product(*shortlist))
    scored=[(dag_predict([curves[r][choice[r]] for r in range(4)]),tuple(choice)) for choice in options]
    cost,choice=min(scored)
    return dict(indices=list(choice),plans=[plans[i].to_dict() for i in choice],predicted_us=cost,
                searched_combinations=len(scored),selection_scope='bounded_joint_dag')


def encode_surface(surface,plans,total_rows):
    validate_surface(surface,plans,total_rows)
    data=[]
    for anchors in surface:
        rows=[]
        for a in anchors:
            row=list(a['features'])+[a[k] for k in SCALARS]
            for k in STAGES:row+=a[k]+[0.]*(4-len(a[k]))
            rows.append(row)
        # Duplicate the last observed anchor only as storage padding. The valid
        # anchor count is uploaded separately; padding is never searched.
        rows += [list(rows[-1]) for _ in range(ANCHORS-len(rows))]
        data.append(rows)
    return data


def profile_digest(profile):
    payload={k:v for k,v in profile.items() if k!='profile_sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_curve_measurements(table):
    """Every uploaded component must be an actual recorded observation.

    This checks table provenance/arithmetic, never equality of different
    candidates' workloads or a numerical/prediction quality cutoff.
    """
    measured=table.get('measurements',{})
    anchors=measured.get('anchors',[])
    if len(anchors)!=ANCHORS:raise ValueError('four actual measured curve anchors required; '+GENERATE)
    for anchor_index,anchor in enumerate(anchors):
        search=anchor.get('search_planner',{})
        observations=search.get('observations',[]);index=search.get('representative_observation')
        if type(index) is not int or not 0<=index<len(observations):raise ValueError('search timing observation missing')
        search_us=observations[index].get('planner_us')
        if not _number(search_us):raise ValueError('finite actual search timing required')
        candidate_records=anchor.get('candidates',[])
        if len(candidate_records)!=len(table['surface']):raise ValueError('candidate timing coverage missing')
        for candidate,record in enumerate(candidate_records):
            observations=record.get('diagnostic_observations',[]);index=record.get('representative_observation')
            if type(index) is not int or not 0<=index<len(observations):raise ValueError('candidate actual observation missing')
            expected=dict(observations[index]['curve'])
            expected['fixed_planner_us']=expected['planner_us'];expected['planner_us']=search_us
            if table['surface'][candidate][anchor_index]!=expected:
                raise ValueError('tail curve differs from its actual measured observation; '+GENERATE)
    return table


def load_profile(path,expected_identity):
    path=Path(path)
    if not path.is_file() or path.is_symlink():raise ValueError('measured native tail profile required; '+GENERATE)
    p=json.loads(path.read_text())
    if (p.get('method')!=METHOD or p.get('policy')!=POLICY or p.get('numerical_policy')!='record_only'
        or p.get('status')!='complete' or p.get('profile_sha256')!=profile_digest(p)):
        raise ValueError('native tail measured profile identity invalid; '+GENERATE)
    for k,v in expected_identity.items():
        if p.get('identity',{}).get(k)!=v:raise ValueError('native tail identity mismatch: '+k+'; '+GENERATE)
    for key,t in p.get('tables',{}).items():
        plans=[Plan(**v) for v in t['plans']]
        validate_surface(t['surface'],plans,t['total_rows'])
        verify_curve_measurements(t)
        if not t.get('measurements') or not t.get('heldout'):
            raise ValueError('actual repeated and held-out execution evidence required; '+GENERATE)
    if not p.get('tables'):raise ValueError('empty native tail measured profile; '+GENERATE)
    return p
