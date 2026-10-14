import atexit
import json
import os

from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM as NativeDeepseekV4
from weave.sglang_backend import install_weave_backend, coverage


class DeepseekV4ForCausalLM(NativeDeepseekV4):
    def forward(self, *args, **kwargs):
        if not hasattr(self, 'weave_runtime'):
            model_path = os.environ['WEAVE_MODEL_PATH']
            fixed = tuple(int(x) for x in os.environ.get('WEAVE_FIXED_PLAN', '1,1,0').split(','))
            if len(fixed) != 3:
                raise ValueError('WEAVE_FIXED_PLAN must be c,K,steal-task-count')
            from weave.execution_profile import TraceConfig
            profiling=(TraceConfig(stride=int(os.environ.get('WEAVE_EXECUTION_PROFILE_STRIDE','128')),
                                   capacity=int(os.environ.get('WEAVE_EXECUTION_PROFILE_CAPACITY','8')))
                       if os.environ.get('WEAVE_EXECUTION_PROFILE')=='1' else None)
            result = install_weave_backend(self, model_path,
                capacity=int(os.environ.get('WEAVE_CAPACITY', '2048')),
                profile_path=os.environ.get('WEAVE_PROFILE'), fixed_plan=fixed,
                kernel=os.environ.get('WEAVE_KERNEL', 'scalar'),
                allow_decode_graph=os.environ.get('WEAVE_DECODE_GRAPH') == '1',execution_profile=profiling)
            if os.environ.get('WEAVE_EVIDENCE_PORT_BASE'):
                from weave.evidence_server import start_evidence_server
                self.weave_evidence_server = start_evidence_server(self, int(os.environ['WEAVE_EVIDENCE_PORT_BASE']))
            print(json.dumps(result), flush=True)
            atexit.register(lambda: print(json.dumps({'weave_coverage': coverage(self)}), flush=True))
        batch = kwargs.get('forward_batch', args[2] if len(args) > 2 else None)
        mode = getattr(batch, 'forward_mode', None)
        self.weave_runtime.phase = ('decode' if mode is not None and mode.is_decode()
                                    else 'prefill' if mode is not None and mode.is_extend() else 'other')
        sampler=getattr(self.weave_runtime,'forward_sampler',None)
        replay_capture=getattr(self.weave_runtime,'replay_capture',None)
        if sampler is None and replay_capture is None:return super().forward(*args, **kwargs)
        ids=kwargs.get('input_ids',args[0] if args else None)
        if ids is None:raise ValueError('eager forward diagnostics require input_ids shape')
        rows=int(ids.shape[0])
        valid=getattr(batch,'batch_size',None) if self.weave_runtime.phase=='decode' else rows
        ticket=sampler.begin(self.weave_runtime.phase,rows,valid) if sampler is not None else None
        if replay_capture is not None:replay_capture.lifecycle.begin(self.weave_runtime.phase,rows,valid)
        try:
            result=super().forward(*args, **kwargs)
        except BaseException:
            if sampler is not None:sampler.fail(ticket)
            if replay_capture is not None:replay_capture.lifecycle.end(success=False)
            raise
        if sampler is not None:sampler.end(ticket)
        if replay_capture is not None:replay_capture.lifecycle.end()
        return result


EntryClass = [DeepseekV4ForCausalLM]
