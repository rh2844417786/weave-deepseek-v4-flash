"""Measured tiled profiles, durable raw event points and identity-safe resume.

This module imports no Torch. Rates are FLOP/s and byte/s; event samples are
milliseconds. No synthetic rate or hardware-clock estimate is accepted.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback
import re

KERNEL = 'tiled_mxfp4_tile_v1'
BUCKETS = (1, 64, 256, 1024, 2048)
IDENTITY_KEYS = ('kernel', 'source_sha256', 'device_key', 'hidden_size',
                 'intermediate_size', 'experts', 'top_k', 'rank', 'workers',
                 'ep_size', 'checkpoint_content_sha256', 'image_id', 'runtime', 'measurement')


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic_json(path, value):
    """Publish calibration JSON with a readable host/container ACL mask.

    NamedTemporaryFile starts at0600, masking inherited named-user ACLs.
    Keep the group class readable before replace so the GPUQ host can read
    root-container artifacts. This helper contains no raw GPQA payloads.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _encoded(value) + b'\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(),0o664)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class CalibrationJournal:
    """Durable active-case marker; GPU stages stay local until success/failure.

    Publish begin() through the caller's collective error agreement before any
    rank launches CUDA. Failure recording uses CPU-only I/O after a poisoned
    CUDA context and retains the first local traceback.
    """
    def __init__(self,directory,identity,invocation,start,budget,*,clock=time.monotonic):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',invocation):
            raise ValueError('safe calibration invocation ID required')
        self.directory=Path(directory)
        self.clock,self.start,self.failed=clock,start,False
        self.value={'status':'measuring','rank':identity['rank'],'invocation_id':invocation,
                    'measuring_started_monotonic':start,'budget_seconds':budget,
                    'source_sha256':identity['source_sha256'],'identity':identity,
                    'active_case':None,'iteration':None,'gpu_stage':'ready','completed_cases':0}
        self.path=self.directory/f'progress-rank{identity["rank"]}.json'
        self.failure_path=self.directory/f'failure-rank{identity["rank"]}-{invocation}.json'

    def save(self): atomic_json(self.path,self.value)

    def begin(self,case,iteration,*,elapsed_seconds,completed_cases):
        self.value.update(active_case=dict(case),iteration=iteration,gpu_stage='before_kernel_iteration',
                          elapsed_seconds=elapsed_seconds,completed_cases=completed_cases,
                          remaining_seconds=self.value['budget_seconds']-elapsed_seconds)
        self.save()

    def stage(self,name): self.value['gpu_stage']=name

    def fail(self,error):
        if self.failed: return
        self.failed=True
        elapsed=self.clock()-self.start
        self.value.update(status='failed',elapsed_seconds=elapsed,
            remaining_seconds=self.value['budget_seconds']-elapsed,
            error=f'{type(error).__name__}: {error}',
            traceback=''.join(traceback.format_exception(type(error),error,error.__traceback__)))
        atomic_json(self.failure_path,self.value)
        self.save()

    def committed(self,count):
        self.value.update(completed_cases=count,gpu_stage='case_committed')
        self.save()


def capture_local_action(operation,*,journal=None):
    """Retain the local exception before converting it for CPU rank agreement."""
    try: return operation(),None
    except Exception as error:
        if journal is not None:
            try: journal.fail(error)
            except OSError as evidence_error:
                error.add_note(f'failure evidence I/O failed: {evidence_error}')
        return None,f'{type(error).__name__}: {error}'


def execute_gpu_iteration(launch,synchronize,*,journal=None,kernel_timeout_seconds=60.,
                          begin_event=None,end_event=None):
    """Keep the device protocol guard independent of the measurement deadline.

    The outer controller owns the hard budget; the driver excludes late points.
    Shrinking the device wait guard to remaining budget would turn ordinary
    deadline expiry into a trap/launch failure and poison the CUDA context.
    """
    if not _positive(kernel_timeout_seconds) or kernel_timeout_seconds>1200:
        raise ValueError('kernel wait guard must be finite in (0,1200]')
    def stage(name):
        if journal is not None: journal.stage(name)
    try:
        if begin_event:
            stage('begin_event_record');begin_event()
        stage('kernel_launch');launch(kernel_timeout_seconds)
        if end_event:
            stage('end_event_record');end_event()
        stage('cuda_synchronize');synchronize()
        stage('iteration_synchronized')
    except Exception as error:
        if journal is not None:
            try: journal.fail(error)
            except OSError as evidence_error:
                error.add_note(f'failure evidence I/O failed: {evidence_error}')
        raise


def measuring_deadline(path,invocation):
    try: marker=json.loads(Path(path).read_text())
    except (OSError,ValueError): return None
    if marker.get('invocation_id')!=invocation: return None
    start,budget=marker.get('measuring_started_monotonic'),marker.get('budget_seconds')
    if not _positive(start) or budget!=1200:
        raise ValueError('invalid calibration watchdog marker')
    return start+1200


def run_diagnostic_case(store,case,measure_case,should_stop):
    """One isolated debug case; never mutate raw timings or derive a bank."""
    if case_key(case) not in {case_key(c) for c in required_cases(store.identity['workers'])}:
        raise ValueError('diagnostic case outside calibrated grid')
    result={'status':'diagnostic_partial','used_for_calibration':False,'case':dict(case)}
    if should_stop(): return result
    value=measure_case(case)
    if value is None or should_stop(): return result
    store._validate_point(value)
    return dict(result,status='diagnostic_passed',point=value)


def candidate_plans(workers, tokens):
    if type(workers) is not int or workers < 2 or type(tokens) is not int or tokens < 1:
        raise ValueError('at least two workers and positive integer tokens required')
    cs = sorted({min(c, workers-1) for c in (0,1,4,8,16,32,64,workers-1)})
    return [(c,k) for c in cs for k in (1,2,4,8) if k <= tokens]


def required_cases(workers, buckets=BUCKETS):
    return [{'tokens': n, 'c': c, 'K': k, 'mode': mode}
            for n in buckets for c,k in candidate_plans(workers,n)
            for mode in (('compute',) if c == 0 else ('compute','communication','mixed'))]


def case_key(point):
    return (point['tokens'], point['c'], point['K'], point['mode'])


def _identity(identity):
    if not isinstance(identity, dict) or any(k not in identity for k in IDENTITY_KEYS):
        raise ValueError('complete calibration identity required')
    if (identity['kernel'] != KERNEL or not _digest(identity['source_sha256'])
            or not _digest(identity['checkpoint_content_sha256'])
            or not isinstance(identity['image_id'], str)
            or not identity['image_id'].startswith('sha256:') or not _digest(identity['image_id'][7:])):
        raise ValueError('kernel/source/checkpoint/image identity invalid')
    if ((identity['hidden_size'],identity['intermediate_size'],identity['experts'],identity['top_k'])
            != (4096,2048,256,6) or any(type(identity[k]) is not int for k in
                ('hidden_size','intermediate_size','experts','top_k','rank','workers','ep_size'))
            or not 0 <= identity['rank'] < 4 or identity['ep_size'] != 4 or identity['workers'] < 2):
        raise ValueError('invalid dimensions/rank/workers for original EP4 checkpoint')
    key = identity['device_key']
    if not isinstance(key, str) or not key.endswith(f';{identity["workers"]}') or ';sm90;' not in key:
        raise ValueError('device key must match Hopper workers')
    runtime = identity['runtime']
    if not isinstance(runtime,dict) or any(not isinstance(runtime.get(k),str) or not runtime[k]
                                            for k in ('python','torch','cuda')):
        raise ValueError('actual Python/Torch/CUDA identity required')
    measurement = identity['measurement']
    if (not isinstance(measurement,dict) or measurement.get('buckets') != list(BUCKETS)
            or type(measurement.get('repetitions')) is not int or measurement['repetitions'] < 2
            or type(measurement.get('warmups')) is not int or measurement['warmups'] < 1):
        raise ValueError('required buckets, warmups and repeated measurements required')
    return {k: identity[k] for k in IDENTITY_KEYS}


def validate_profile_bundle(bundle, *, kernel, source_sha256, device_key,
                            hidden_size, intermediate_size, rank, workers):
    """Return sorted normalized bucket rows, rejecting incomplete measured banks."""
    try:
        identity = _identity(bundle)
        expected = dict(kernel=kernel,source_sha256=source_sha256,device_key=device_key,
                        hidden_size=hidden_size,intermediate_size=intermediate_size,rank=rank,workers=workers)
        if any(identity[k] != v for k,v in expected.items()):
            raise ValueError('profile is stale for kernel/source/device/shape/rank/workers')
        if type(bundle.get('schema_version')) is not int or bundle['schema_version'] != 1 or bundle.get('status') != 'measured':
            raise ValueError('only complete measured schema_version=1 profiles accepted')
        raw = bundle['raw_timings']
        if not isinstance(raw,dict) or not isinstance(bundle.get('buckets'),list):
            raise ValueError('raw source metadata and bucket list required')
        if (raw.get('source') != 'cuda_event' or not _digest(raw.get('sha256'))
                or raw.get('file') != f'timings-rank{rank}.json'
                or type(raw.get('count')) is not int or raw['count'] != len(required_cases(workers))):
            raise ValueError('complete raw CUDA-event timing source and hash required')
        buckets, found = [], set()
        for item in bundle['buckets']:
            n = item['max_local_tokens']
            if type(n) is not int or n not in BUCKETS or n in found:
                raise ValueError('invalid or duplicate token bucket')
            found.add(n)
            rows, plans = [], set()
            for row in item['candidates']:
                if not isinstance(row, (list,tuple)) or len(row) != 6:
                    raise ValueError('candidate must have six fields')
                c,k,rate,bw,eff,overlap = row
                if (type(c) is not int or type(k) is not int or not 0 <= c < workers
                        or not 1 <= k <= min(8,n) or (c,k) in plans
                        or not all(type(x) in (int,float) and math.isfinite(x) for x in row[2:])
                        or rate <= 0 or not 0 < eff <= 1 or not 0 <= overlap < 1
                        or (c == 0 and (bw != 0 or overlap != 0)) or (c > 0 and bw <= 0)):
                    raise ValueError('invalid, duplicate or nonfinite candidate')
                plans.add((c,k))
                rows.append([c,k,float(rate),float(bw),float(eff),float(overlap)])
            if plans != set(candidate_plans(workers,n)):
                raise ValueError('incomplete required candidate grid')
            buckets.append({'max_local_tokens': n, 'candidates': sorted(rows,key=lambda r:(r[0],r[1]))})
        if found != set(BUCKETS):
            raise ValueError('incomplete required token buckets including 2048')
        return sorted(buckets,key=lambda b:b['max_local_tokens'])
    except (KeyError,TypeError,OverflowError) as exc:
        raise ValueError(f'malformed profile bundle: {exc}') from exc


def work_units(point, hidden_size, intermediate_size):
    """Convert actual workload counters into logical arithmetic and wire bytes."""
    values = point.get('workload')
    if not isinstance(values,list) or len(values) != 7 or any(type(v) is not int or v < 0 for v in values):
        raise ValueError('seven nonnegative actual workload counters required')
    local,remote,unique,payloads,tasks,groups,tiles = values
    n, mode = point['tokens'], point['mode']
    k=point['K']
    if type(n) is not int or n<1 or type(k) is not int or not 1<=k<=min(8,n):
        raise ValueError('invalid grouping fixture dimensions')
    expected_groups=6*sum((((j+1)*n+k-1)//k-(j*n+k-1)//k+15)//16 for j in range(k))
    if groups != expected_groups:
        raise ValueError('group count differs from deterministic six-expert fixture')
    if (local+remote != 6*n or tasks != 6*n or payloads != n or unique > n
            or (mode == 'compute' and (remote != 0 or unique != 0))
            or (mode != 'compute' and (local != 0 or remote != 6*n or unique != n))
            or groups < (tasks+15)//16 or tiles != groups*((hidden_size+63)//64+(intermediate_size+63)//64)
            or point.get('tile_counters') != [tiles,0]):
        raise ValueError('workload does not match local/next-rank six-expert steal0 calibration')
    dispatch = unique*hidden_size*2
    returned = remote*hidden_size*4
    return {'logical_flops': 6*hidden_size*intermediate_size*tasks,
            'padded_flops': groups*16*(4*hidden_size*((intermediate_size+63)//64*64)+
                                      2*intermediate_size*((hidden_size+63)//64*64)),
            'dispatch_bytes': dispatch, 'return_bytes': returned, 'remote_bytes': dispatch+returned,
            'groups':groups, 'tile_jobs':tiles}


def fit_overlap(compute_ms, communication_ms, mixed_ms, *, return_fraction, chunks):
    """Lower estimate fitting compute + (1-overlap)*return_time/K to mixed.

    Use fastest primitive samples and slowest mixed sample to avoid optimistic
    overlap. Communication-dominated samples cannot identify it, so choose zero.
    Values outside the legal interval are clipped and the derivation is retained.
    """
    if (any(not seq or any(not _positive(x) for x in seq) for seq in
            (compute_ms,communication_ms,mixed_ms)) or not _positive(return_fraction)
            or return_fraction > 1 or type(chunks) is not int or not 1 <= chunks <= 8):
        raise ValueError('finite positive measured samples and legal return fraction/K required')
    compute,communication,mixed = min(compute_ms),min(communication_ms),max(mixed_ms)
    tail = communication*return_fraction/chunks
    unidentified = mixed <= communication
    raw = 0. if unidentified else 1.-max(0.,mixed-compute)/tail
    overlap = min(math.nextafter(1.,0.),max(0.,raw))
    return {'overlap':overlap, 'raw_overlap':raw, 'clipped':'lower' if raw < 0 else ('upper' if raw >= 1 else None),
            'unidentifiable':unidentified, 'compute_ms_min':compute,
            'communication_ms_min':communication,'mixed_ms_max':mixed,'return_tail_ms':tail,
            'derivation':'1-max(0,mixed_max-compute_min)/(communication_min*return_fraction/K); unidentifiable -> 0'}


class TimingStore:
    """One atomic rank-local document; only fully repeated on-time cases commit."""
    def __init__(self,path,identity,*,resume=False,create_missing=True):
        self.path = Path(path)
        self.identity = _identity(identity)
        self.points, self.orphaned_points, self.excluded_points, self.status = [], [], [], 'partial'
        if self.path.exists():
            if not resume:
                raise FileExistsError(f'{self.path} exists; use --resume with matching identity')
            saved = json.loads(self.path.read_text())
            if not isinstance(saved,dict) or type(saved.get('schema_version')) is not int or saved['schema_version'] != 1 or saved.get('identity') != self.identity:
                raise ValueError('resume identity/shape/grid/runtime does not match')
            if saved.get('points_sha256') != hashlib.sha256(_encoded(saved.get('points'))).hexdigest():
                raise ValueError('raw point content hash mismatch')
            orphaned = saved.get('orphaned_points',[])
            if saved.get('orphaned_sha256') != hashlib.sha256(_encoded(orphaned)).hexdigest():
                raise ValueError('orphaned raw point content hash mismatch')
            for p in orphaned:
                self._validate_point(p)
            self.orphaned_points = orphaned
            excluded = saved.get('excluded_points',[])
            if saved.get('excluded_sha256') != hashlib.sha256(_encoded(excluded)).hexdigest():
                raise ValueError('excluded raw point content hash mismatch')
            for p in excluded:
                self._validate_excluded(p)
            self.excluded_points = excluded
            seen = set()
            for p in saved['points']:
                self._validate_point(p)
                if case_key(p) in seen:
                    raise ValueError('duplicate resumed timing case')
                seen.add(case_key(p))
            self.points, self.status = saved['points'], saved.get('status','partial')
            if self.status not in ('partial','measured') or (self.status == 'measured' and seen != {case_key(c) for c in required_cases(self.identity['workers'])}):
                raise ValueError('incomplete or invalid measured raw status')
        elif not create_missing:
            raise ValueError('required measured raw timing file is absent')
        elif resume:
            # A fresh rank in an interrupted invocation is permitted, but all
            # ranks collectively intersect completed keys before continuing.
            self.save()
        else:
            self.save()

    def _validate_point(self,point):
        try:
            key = case_key(point)
            if any(type(point[k]) is not int for k in ('tokens','c','K')):
                raise ValueError('raw tokens/c/K must be integers')
            if key not in {case_key(c) for c in required_cases(self.identity['workers'])}:
                raise ValueError('case outside required calibration grid')
            samples = point['samples_ms']
            if (not isinstance(samples,list) or len(samples) != self.identity['measurement']['repetitions']
                    or any(not _positive(x) for x in samples)
                    or type(point.get('warmups')) is not int or point['warmups'] != self.identity['measurement']['warmups']
                    or point.get('completed_before_deadline') is not True):
                raise ValueError('complete positive on-time CUDA-event samples required')
            work_units(point,self.identity['hidden_size'],self.identity['intermediate_size'])
        except (KeyError,TypeError) as exc:
            raise ValueError(f'malformed raw timing point: {exc}') from exc

    def save(self,*,status=None,reason=None):
        if status is not None:
            if status not in ('partial','measured'):
                raise ValueError('raw store status must be partial or measured')
            if status == 'measured' and {case_key(p) for p in self.points} != {case_key(c) for c in required_cases(self.identity['workers'])}:
                raise ValueError('incomplete raw points cannot claim measured')
            self.status = status
        value = {'schema_version':1,'status':self.status,'identity':self.identity,'points':self.points,
                 'points_sha256':hashlib.sha256(_encoded(self.points)).hexdigest(),
                 'orphaned_points':self.orphaned_points,
                 'orphaned_sha256':hashlib.sha256(_encoded(self.orphaned_points)).hexdigest(),
                 'excluded_points':self.excluded_points,
                 'excluded_sha256':hashlib.sha256(_encoded(self.excluded_points)).hexdigest()}
        if reason:
            value['reason'] = reason
        atomic_json(self.path,value)

    def _validate_excluded(self,point):
        try:
            if (case_key(point) not in {case_key(c) for c in required_cases(self.identity['workers'])}
                    or point.get('completed_before_deadline') is not False
                    or not isinstance(point.get('exclusion_reason'),str) or not point['exclusion_reason']
                    or not isinstance(point.get('samples_ms'),list)
                    or len(point['samples_ms']) > self.identity['measurement']['repetitions']
                    or any(not _positive(v) for v in point['samples_ms'])):
                raise ValueError('invalid excluded timing attempt')
        except (KeyError,TypeError) as exc:
            raise ValueError(f'malformed excluded timing attempt: {exc}') from exc

    def record_excluded(self,point,*,reason):
        value = dict(point,completed_before_deadline=False,exclusion_reason=reason)
        self._validate_excluded(value)
        self.excluded_points.append(json.loads(_encoded(value)))
        self.save(status='partial',reason=reason)

    def reconcile_completed(self,common_keys):
        """Rerun non-universal cases on all ranks, retaining every old sample."""
        kept,orphaned = [],[]
        for p in self.points:
            (kept if case_key(p) in common_keys else orphaned).append(p)
        if orphaned:
            self.points = kept
            self.orphaned_points.extend(orphaned)
            self.save(status='partial',reason='resume intersection excludes cases not committed on every rank')

    def append(self,point,*,elapsed_seconds,budget_seconds):
        if (not _positive(budget_seconds) or budget_seconds > 1200 or
                type(elapsed_seconds) not in (int,float) or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0):
            raise ValueError('finite elapsed time and at most 1200 second budget required')
        if elapsed_seconds > budget_seconds:
            raise TimeoutError('late CUDA-event point excluded from calibration')
        self._validate_point(point)
        if case_key(point) in {case_key(p) for p in self.points}:
            raise ValueError('duplicate calibration case')
        self.points.append(json.loads(_encoded(point)))
        self.status = 'partial'
        self.save()


def derive_profile_bundle(identity,raw_points,raw_metadata):
    """Pure profile arithmetic over complete, validated raw points."""
    identity = _identity(identity)
    points = {case_key(p):p for p in raw_points}
    expected = {case_key(c) for c in required_cases(identity['workers'])}
    if set(points) != expected:
        raise ValueError(f'incomplete calibration: {len(expected-set(points))} required cases missing')
    if len(raw_points) != len(points):
        raise ValueError('duplicate raw points cannot derive a profile')
    buckets,derivations = [],[]
    for n in BUCKETS:
        plans = candidate_plans(identity['workers'],n)
        rates = {(c,k):work_units(points[n,c,k,'compute'],4096,2048)['padded_flops'] /
                        (max(points[n,c,k,'compute']['samples_ms'])*1e-3) for c,k in plans}
        bases = {c:max(rate for (comm,_),rate in rates.items() if comm == c) for c,_ in plans}
        rows = []
        for c,k in plans:
            bandwidth,overlap = 0.,0.
            if c:
                compute,comm,mixed = [points[n,c,k,m] for m in ('compute','communication','mixed')]
                units = work_units(comm,4096,2048)
                bandwidth = units['remote_bytes']/(max(comm['samples_ms'])*1e-3)
                fit = fit_overlap(compute['samples_ms'],comm['samples_ms'],mixed['samples_ms'],
                                  return_fraction=units['return_bytes']/units['remote_bytes'],chunks=k)
                overlap = fit['overlap']
                derivations.append(dict(tokens=n,c=c,K=k,**fit))
            rows.append([c,k,bases[c],bandwidth,rates[c,k]/bases[c],overlap])
        buckets.append({'max_local_tokens':n,'candidates':rows})
    bundle = dict(identity,schema_version=1,status='measured',buckets=buckets,
                  raw_timings=raw_metadata,
                  overlap_derivations=derivations,
                  units={'compute':'padded WMMA FLOP/s; 6*H*I*16*actual_groups; logical FLOPs separately recorded',
                         'communication':'byte/s; actual_remote_unique*H*2 + actual_remote_pairs*H*4; includes protocol overhead',
                         'timing':'repeated synchronized CUDA events in milliseconds; conservative max duration rates',
                         'base_and_efficiency':'per-c max K measured rate; efficiency=measured_K_rate/base'})
    validate_profile_bundle(bundle,**{k:identity[k] for k in
        ('kernel','source_sha256','device_key','hidden_size','intermediate_size','rank','workers')})
    return bundle


def build_profile_bundle(store):
    """Publish raw measured status, then derive its hash-bound profile bank."""
    for point in store.points:
        store._validate_point(point)
    store.save(status='measured')
    metadata = {'file':store.path.name,'sha256':hashlib.sha256(store.path.read_bytes()).hexdigest(),
                'count':len(store.points),'source':'cuda_event'}
    return derive_profile_bundle(store.identity,store.points,metadata)


def load_verified_profile_bundle(path,**validate_kwargs):
    """Read-only verified loading; return normalized bucket dictionaries.

    Verify sibling raw bytes, every sample and identity, and rederive all rates
    and efficiencies. Loading never rewrites evidence or publishes a new bank.
    """
    path = Path(path)
    try:
        bundle = json.loads(path.read_text())
        normalized = validate_profile_bundle(bundle,**validate_kwargs)
        raw_path = path.with_name(bundle['raw_timings']['file'])
        raw_before = raw_path.read_bytes()
        if hashlib.sha256(raw_before).hexdigest() != bundle['raw_timings']['sha256']:
            raise ValueError('raw timing file hash differs from measured profile')
        store = TimingStore(raw_path,_identity(bundle),resume=True,create_missing=False)
        if store.status != 'measured':
            raise ValueError('profile raw timing source is not fully measured')
        derived = derive_profile_bundle(store.identity,store.points,bundle['raw_timings'])
        if normalized != derived['buckets']:
            raise ValueError('profile candidates differ from rates derived from raw CUDA-event points')
        if raw_path.read_bytes() != raw_before:
            raise ValueError('raw timing file changed during read-only verification')
        return normalized
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError(f'missing or malformed measured profile/raw evidence: {exc}') from exc


def run_calibration(store,measure_case,should_stop,elapsed,budget_seconds):
    """Drive measured cases with shared stop decisions before/after every case.

    CUDA/distributed boundaries are supplied by the CLI. CPU tests execute the
    real persistence and deadline decisions without inventing GPU measurements.
    """
    try:
        completed = {case_key(p) for p in store.points}
        for case in required_cases(store.identity['workers']):
            if case_key(case) in completed:
                continue
            if should_stop():
                store.save(status='partial',reason='deadline or signal')
                return {'status':'partial','completed_cases':len(store.points)}
            point = measure_case(case)
            if point is None or should_stop():
                if point is not None:
                    store.record_excluded(point,reason='deadline or signal after events')
                store.save(status='partial',reason='deadline or signal; unfinished/late case excluded')
                return {'status':'partial','completed_cases':len(store.points)}
            try:
                store.append(point,elapsed_seconds=elapsed(),budget_seconds=budget_seconds)
            except TimeoutError:
                store.record_excluded(point,reason='late CUDA-event point')
                raise
        return {'status':'measured','completed_cases':len(store.points),
                'bundle':build_profile_bundle(store)}
    except TimeoutError:
        store.save(status='partial',reason='late CUDA-event point excluded')
        return {'status':'partial','completed_cases':len(store.points)}
    except BaseException as exc:
        store.save(status='partial',reason=f'{type(exc).__name__}: {exc}')
        raise
