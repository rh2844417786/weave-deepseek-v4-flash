"""Bounded eager full-forward observations; events are read only at idle snapshot.

GPU stream elapsed time includes queued GPU work, waits and launch starvation.
Host enqueue time is not completion latency. Neither is HTTP token throughput.
"""
import math
import threading
import time


def validate_forward_profile(value,*,stride,capacity):
    if (not isinstance(value,dict) or value.get('scope')!='full_model_forward_current_stream_eager'
            or value.get('used_for_formal_performance') is not False
            or (value.get('stride'),value.get('capacity'))!=(stride,capacity)
            or set(value.get('phases',{}))!={'prefill','decode','other'}):
        raise ValueError('forward profile identity/configuration missing')
    for phase,state in value['phases'].items():
        calls=state.get('calls');sampled=state.get('sampled_invocations');total=state.get('valid_rows_total')
        if any(type(v) is not int or v<0 for v in (calls,sampled,total)) or sampled!=(calls+stride-1)//stride:
            raise ValueError('forward counters disagree')
        histogram=state.get('batch_histogram',{})
        if (not isinstance(histogram,dict) or any(not k.isdigit() or str(int(k))!=k
                or type(n) is not int or n<=0 for k,n in histogram.items())
                or sum(histogram.values())!=calls or sum(int(k)*n for k,n in histogram.items())!=total):
            raise ValueError('forward batch histogram disagrees with row/call counters')
        recent=state.get('recent_samples',[])
        expected=list(range(max(0,sampled-capacity)*stride,sampled*stride,stride))
        if ([r.get('sequence') for r in recent]!=expected
                or state.get('overwritten_samples')!=max(0,sampled-capacity)):
            raise ValueError('forward sample sequence is incomplete')
        for row in recent:
            n=row.get('input_rows');valid=row.get('valid_rows')
            if (type(n) is not int or type(valid) is not int or not 0<=valid<=n
                    or (phase=='decode' and valid!=n) or str(valid) not in histogram):
                raise ValueError('forward sample row count invalid')
            for key in ('host_enqueue_ms','gpu_stream_ms'):
                v=row.get(key)
                if type(v) not in (int,float) or not math.isfinite(v) or v<0:
                    raise ValueError('forward sample time invalid')
    return value


class ForwardSampler:
    def __init__(self,cuda,*,stride=128,capacity=8,clock=time.perf_counter):
        if (type(stride) is not int or not 1<=stride<=65536
                or type(capacity) is not int or not 1<=capacity<=64):
            raise ValueError('bounded forward sampling configuration required')
        self.cuda,self.stride,self.capacity,self.clock=cuda,stride,capacity,clock
        self.lock=threading.RLock()
        self.events={phase:[(cuda.Event(enable_timing=True),cuda.Event(enable_timing=True))
                            for _ in range(capacity)] for phase in ('prefill','decode','other')}
        self.active=None
        self.reset()

    def reset(self):
        with self.lock:
            if self.active is not None:raise ValueError('forward active during reset')
            self.failed=0
            self.phases={phase:{'calls':0,'valid_rows_total':0,'batch_histogram':{},
                                'sampled_invocations':0,'slots':[None]*self.capacity}
                         for phase in self.events}

    def begin(self,phase,input_rows,valid_rows):
        with self.lock:
            if self.cuda.is_current_stream_capturing():raise ValueError('forward event profiler requires eager execution')
            if (phase not in self.phases or type(input_rows) is not int or type(valid_rows) is not int
                    or not 0<=valid_rows<=input_rows or (phase=='decode' and valid_rows!=input_rows)):
                raise ValueError('eager forward requires explicit valid rows; padded decode is unsupported')
            if self.active is not None:raise ValueError('overlapping host forwards in one rank')
            state=self.phases[phase];sequence=state['calls'];state['calls']+=1
            state['valid_rows_total']+=valid_rows
            key=str(valid_rows);state['batch_histogram'][key]=state['batch_histogram'].get(key,0)+1
            ticket={'phase':phase,'sequence':sequence,'input_rows':input_rows,'valid_rows':valid_rows,'slot':None}
            if sequence%self.stride==0:
                ticket['slot']=state['sampled_invocations']%self.capacity
                state['sampled_invocations']+=1
                state['slots'][ticket['slot']]=None
                self.events[phase][ticket['slot']][0].record()
                ticket['host_start']=self.clock()
            self.active=ticket
            return ticket

    def end(self,ticket):
        with self.lock:
            if self.active is not ticket:raise ValueError('forward sampler ticket mismatch')
            if ticket['slot'] is not None:
                self.events[ticket['phase']][ticket['slot']][1].record()
                elapsed=(self.clock()-ticket['host_start'])*1000
                self.phases[ticket['phase']]['slots'][ticket['slot']]={
                    key:ticket[key] for key in ('sequence','input_rows','valid_rows')}
                self.phases[ticket['phase']]['slots'][ticket['slot']]['host_enqueue_ms']=elapsed
            self.active=None

    def fail(self,ticket):
        with self.lock:
            self.failed+=1
            if self.active is ticket:self.active=None

    def snapshot(self):
        """Caller must synchronize the device and establish service idleness first."""
        with self.lock:
            if self.active is not None or self.failed:raise ValueError('incomplete forward timing window')
            result={'schema_version':1,'scope':'full_model_forward_current_stream_eager',
                    'clock':'cuda_events_and_host_perf_counter','stride':self.stride,'capacity':self.capacity,
                    'used_for_formal_performance':False,'phases':{}}
            for phase,state in self.phases.items():
                recent=[]
                for slot,row in enumerate(state['slots']):
                    if row is None:continue
                    begin,end=self.events[phase][slot]
                    value=dict(row,gpu_stream_ms=float(begin.elapsed_time(end)))
                    if any(not math.isfinite(value[k]) or value[k]<0 for k in ('host_enqueue_ms','gpu_stream_ms')):
                        raise ValueError('invalid forward elapsed time')
                    recent.append(value)
                recent.sort(key=lambda row:row['sequence'])
                if len(recent)!=min(state['sampled_invocations'],self.capacity):
                    raise ValueError('missing forward event samples')
                result['phases'][phase]={key:state[key] for key in ('calls','valid_rows_total','sampled_invocations')}
                result['phases'][phase].update(batch_histogram=dict(state['batch_histogram']),recent_samples=recent,
                    overwritten_samples=max(0,state['sampled_invocations']-self.capacity))
            return result
