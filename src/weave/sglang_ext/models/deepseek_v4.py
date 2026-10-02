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
            result = install_weave_backend(self, model_path,
                capacity=int(os.environ.get('WEAVE_CAPACITY', '2048')),
                profile_path=os.environ.get('WEAVE_PROFILE'), fixed_plan=fixed,
                kernel=os.environ.get('WEAVE_KERNEL', 'scalar'),
                allow_decode_graph=os.environ.get('WEAVE_DECODE_GRAPH') == '1')
            if os.environ.get('WEAVE_EVIDENCE_PORT_BASE'):
                from weave.evidence_server import start_evidence_server
                self.weave_evidence_server = start_evidence_server(self, int(os.environ['WEAVE_EVIDENCE_PORT_BASE']))
            print(json.dumps(result), flush=True)
            atexit.register(lambda: print(json.dumps({'weave_coverage': coverage(self)}), flush=True))
        batch = kwargs.get('forward_batch', args[2] if len(args) > 2 else None)
        mode = getattr(batch, 'forward_mode', None)
        self.weave_runtime.phase = ('decode' if mode is not None and mode.is_decode()
                                    else 'prefill' if mode is not None and mode.is_extend() else 'other')
        return super().forward(*args, **kwargs)


EntryClass = [DeepseekV4ForCausalLM]
