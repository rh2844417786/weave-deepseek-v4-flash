"""Source-bound, measured native MoE budgeting with graph-stable device state.

Calibration is an explicit startup phase. Formal decode requires a complete
matching profile and performs no Python routing readback or timing sampling.
"""
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re

from .calibration import atomic_json
from .native_budget import ROOT, ROLES, STATE_SIZE, source_pins, state_row
from .native_budget_patch import verify_sources
from .native_cost import (METHOD, POLICY_VERSION, CANDIDATES, PLAN_SIZE, TABLE_SHAPE,
                          load_profile, validate_profile, profile_digest, native_cost_source_sha256)

DECODE = ContextVar('weave_cost_decode', default=False)
CURRENT = ContextVar('weave_cost_layer', default=None)
GENERATE = 'python3 scripts/run_native_cost.py --work-root /srv/weave-work --records <prepared/main.jsonl> --execute'


def cost_options(environ=None):
    env = os.environ if environ is None else environ
    keys = ('WEAVE_NATIVE_COST_TABLE_DIR', 'WEAVE_NATIVE_COST_CALIBRATE_DIR', 'WEAVE_NATIVE_COST_REPEATS')
    if env.get('WEAVE_NATIVE_BUDGET', 'off') != 'cost':
        if any(k in env for k in keys):
            raise ValueError('native cost settings require --native-budget cost')
        return None
    if (env.get('WEAVE_GRAPH_C32') != '1' or env.get('WEAVE_REQUEST_TIMING') != '1'
            or env.get('SGLANG_EXTERNAL_MODEL_PACKAGE')
            or any(env.get(k, '0') != '0' for k in ('WEAVE_NATIVE_P0', 'WEAVE_GRAPH_STAGE_PROFILE',
                    'WEAVE_FIXED_PREFIX', 'WEAVE_EXECUTION_PROFILE', 'WEAVE_FORWARD_PROFILE', 'WEAVE_STALL_DIAGNOSTIC'))
            or env.get('WEAVE_PROFILE') or env.get('WEAVE_REPLAY_CAPTURE_DIR')):
        raise ValueError('native cost requires native full graph C32 without diagnostic/profile intervention')
    table, output = env.get(keys[0]), env.get(keys[1])
    if bool(table) == bool(output):
        raise ValueError('one measured table directory or explicit calibration output required; generate with ' + GENERATE)
    directory = table or output
    if not Path(directory).is_absolute() or '..' in Path(directory).parts:
        raise ValueError('native cost directory must be absolute under WORK_ROOT')
    repeats = int(env.get(keys[2], '7'))
    if repeats < 2:
        raise ValueError('native cost requires at least two repeated timing measurements')
    return dict(mode='cost' if table else 'calibration', directory=directory, repeats=repeats)


def _class_name(module):
    return None if module is None else type(module).__module__ + '.' + type(module).__qualname__


def _module_signature(module):
    if module is None:
        return None
    parameters = getattr(module, 'named_parameters', lambda **_: [])(recurse=False)
    return dict(type=_class_name(module), quant_method=_class_name(getattr(module, 'quant_method', None)),
                parameters={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                            for name, value in sorted(parameters)})


def classify_layers(model):
    layers = model.model.layers
    if len(layers) != 43:
        raise RuntimeError('native cost requires 43 layers')
    groups, mapping, keys = {}, {}, {}
    for index, block in enumerate(layers):
        mlp = block.mlp
        shared = getattr(mlp, 'shared_experts', None)
        signature = dict(topk=_class_name(mlp.topk), is_hash=bool(getattr(mlp, 'is_hash', False)),
            shared_tp1=bool(getattr(mlp, '_shared_expert_tp1', False)),
            routed=_module_signature(mlp.experts), gate=_module_signature(mlp.gate),
            shared=_module_signature(shared), shared_projections={name: _module_signature(getattr(shared, name, None))
                for name in ('gate_up_proj', 'down_proj')})
        key = json.dumps(signature, sort_keys=True)
        if key not in keys:
            name = 'g' + str(len(groups)); keys[key] = name
            groups[name] = dict(representative_layer=index, layers=[], signature=signature)
        name = keys[key]; groups[name]['layers'].append(index); mapping[index] = name
    return groups, mapping


def install_cost_dispatch(model, marlin_module, model_module, runtime, *, layers=43):
    if len(model.model.layers) != layers:
        raise RuntimeError('native cost requires complete layer coverage')
    original_op = marlin_module.moe_wna16_marlin_gemm
    if getattr(original_op, '_weave_cost_dispatch', False) or getattr(original_op, '_weave_budget_dispatch', False):
        raise RuntimeError('native cost cannot stack another budget dispatch')

    @wraps(original_op)
    def dispatch(*args, **kwargs):
        state = CURRENT.get()
        if state is None or state['runtime'] is not runtime:
            return original_op(*args, **kwargs)
        index = state['calls']
        if index >= 2:
            raise RuntimeError('native cost requires exactly two routed Marlin GEMMs')
        if not kwargs.get('is_ep') or kwargs.get('mul_topk_weights') != bool(index) or (index == 1 and kwargs.get('top_k') != 1):
            raise RuntimeError('native cost routed Marlin role/EP contract differs')
        state['calls'] += 1
        return runtime.dispatch(state['layer'], index, args, kwargs,
            observer=state['observer'], original_native=state['original_native'], original=original_op)
    dispatch._weave_cost_dispatch = True
    marlin_module.moe_wna16_marlin_gemm = dispatch

    original_reduce = model_module.tensor_model_parallel_all_reduce
    @wraps(original_reduce)
    def reduce(*args, **kwargs):
        state = CURRENT.get()
        observer = state['observer'] if state and state['runtime'] is runtime else None
        with observer.span('all_reduce') if observer is not None else nullcontext():
            return original_reduce(*args, **kwargs)
    model_module.tensor_model_parallel_all_reduce = reduce

    def invoke(layer, original, args, kwargs, observer=None, original_native=False):
        state = dict(runtime=runtime, layer=layer, calls=0, observer=observer, original_native=original_native)
        token = CURRENT.set(state)
        try:
            result = original(*args, **kwargs)
            if state['calls'] != 2:
                raise RuntimeError('native cost requires exactly two routed Marlin GEMMs')
            return result
        finally:
            CURRENT.reset(token)
    runtime.invoke = invoke

    for index, block in enumerate(model.model.layers):
        mlp = block.mlp
        for name in ('forward_normal_dual_stream', 'forward_normal', 'forward_deepep'):
            method = getattr(mlp, name, None)
            if method is None:
                continue
            @wraps(method)
            def path(*args, _original=method, _layer=index, _name=name, **kwargs):
                state = CURRENT.get()
                if state and state['runtime'] is runtime:
                    if _name == 'forward_deepep':
                        raise RuntimeError('measured native cost profile does not describe A2A dispatch/combine')
                    runtime.path_tags[_layer] = _name
                return _original(*args, **kwargs)
            setattr(mlp, name, path)
        method = getattr(mlp, '_forward_shared_experts', None)
        if method is not None:
            @wraps(method)
            def shared(*args, _original=method, **kwargs):
                state = CURRENT.get()
                observer = state['observer'] if state and state['runtime'] is runtime else None
                with observer.span('shared') if observer is not None else nullcontext():
                    return _original(*args, **kwargs)
            mlp._forward_shared_experts = shared
        original = mlp.forward
        @wraps(original)
        def mlp_forward(*args, _original=original, _layer=index, **kwargs):
            if not DECODE.get():
                return _original(*args, **kwargs)
            if runtime.mode == 'calibration':
                runtime.maybe_calibrate(_layer, _original, args, kwargs)
            result = invoke(_layer, _original, args, kwargs)
            if runtime.mode == 'cost' and hasattr(runtime, 'check_path'):
                hidden = args[0] if args else kwargs['hidden_states']
                runtime.check_path(_layer, int(hidden.shape[0]))
            return result
        mlp.forward = mlp_forward
    original_model = model.forward
    @wraps(original_model)
    def model_forward(*args, **kwargs):
        batch = kwargs.get('forward_batch', args[2] if len(args) > 2 else None)
        if batch is None:
            raise RuntimeError('native cost requires explicit model forward mode')
        token = DECODE.set(bool(batch.forward_mode.is_decode()))
        try:
            return original_model(*args, **kwargs)
        finally:
            DECODE.reset(token)
    model.forward = model_forward


def deployment_identity(torch, rank, package_root):
    from .runtime_contract import load_contract
    contract = load_contract()
    for name in ('sglang-kernel', 'sgl-kernel'):
        try:
            kernel = importlib.metadata.distribution(name)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    else:
        raise RuntimeError('installed kernel distribution missing (sglang-kernel, sgl-kernel)')
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    versions = dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                    sglang=importlib.metadata.version('sglang'), sgl_kernel=kernel.version)
    for key, setting in (('python','python_version'),('torch','torch_version'),('cuda','torch_cuda'),('sglang','sglang_version')):
        if versions[key] != contract[setting]:
            raise RuntimeError('native cost runtime contract differs: ' + key)
    model = Path(os.environ.get('WEAVE_MODEL_PATH', '/model'))
    raw = (model / 'config.json').read_bytes(); config = json.loads(raw)
    actual = tuple(config[k] for k in ('hidden_size','moe_intermediate_size','n_routed_experts','num_experts_per_tok','num_hidden_layers'))
    if actual != (4096,2048,256,6,43) or [props.major,props.minor] != [9,0]:
        raise RuntimeError('native cost supports original V4 Flash on SM90 only')
    image = os.environ.get('WEAVE_IMAGE_ID',''); checkpoint = os.environ.get('WEAVE_CHECKPOINT_CONTENT_SHA256','')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',image) or not re.fullmatch(r'[0-9a-f]{64}',checkpoint):
        raise RuntimeError('native cost requires verified image/checkpoint identity; use ' + GENERATE)
    return dict(cost_source_sha256=native_cost_source_sha256(), upstream_sha256=verify_sources(package_root),
        image_id=image, checkpoint_content_sha256=checkpoint, model_config_sha256=hashlib.sha256(raw).hexdigest(),
        sglang_commit=contract['sglang_commit'], rank=rank,
        gpu=dict(name=props.name,capability=[props.major,props.minor],sm_count=props.multi_processor_count,total_memory=props.total_memory),
        runtime=versions,precision=dict(activations='bfloat16',routed_weights='mxfp4_e8m0',shared_weights='fp8'),
        model_geometry=dict(hidden_size=4096,intermediate=2048,experts=256,top_k=6,layers=43),
        graph_contract=dict(tp=4,ep=4,moe_tp=1,a2a_backend='none',decode_batches=list(range(1,33))))


class NativeCostRuntime:
    def __init__(self, model, torch, rank, package_root, *, mode, directory, repeats):
        self.model, self.torch, self.rank, self.package_root = model, torch, rank, Path(package_root)
        self.mode, self.directory, self.repeats = mode, Path(directory), repeats
        self.device = torch.cuda.current_device()
        self.identity = deployment_identity(torch, rank, package_root)
        self.groups, self.layer_to_group = classify_layers(model)
        self.profile = None; self.calibration_complete = False
        self.completed = {name:{} for name in self.groups}
        self.path_tags, self.bindings = {}, {}
        self.calibrating = False
        if mode == 'cost':
            self.profile = load_profile(self.directory / f'native-cost-rank{rank}.json', self.identity)
            if self.profile['groups'] != self.groups:
                raise RuntimeError('native cost structural class/checkpoint geometry differs; regenerate with ' + GENERATE)
            self.calibration_complete = True
        else:
            self.directory.mkdir(parents=True,exist_ok=True)
            if (self.directory / f'native-cost-rank{rank}.json').exists():
                raise FileExistsError('fresh native cost calibration output required')
        self.tables = {(name,b): (torch.tensor(self.profile['tables'][name][str(b)]['rows'],dtype=torch.float32,device=self.device)
                       if self.profile else torch.zeros(TABLE_SHAPE,dtype=torch.float32,device=self.device))
                       for name in self.groups for b in range(1,33)}
        self.state_buffer = torch.zeros((43,2,STATE_SIZE),dtype=torch.int64,device=self.device)
        self.plan_buffer = torch.zeros((43,PLAN_SIZE),dtype=torch.int64,device=self.device)
        self.control_buffer = torch.tensor([[0 if mode=='cost' else 1,8,8]]*43,dtype=torch.int32,device=self.device)
        self.states = {(i,role):self.state_buffer[i,j] for i in range(43) for j,role in enumerate(ROLES)}
        self.states.update({(i,j):self.states[i,role] for i in range(43) for j,role in enumerate(ROLES)})
        self.plans = {i:self.plan_buffer[i] for i in range(43)}
        self.controls = {i:self.control_buffer[i] for i in range(43)}
        from .native_cost_patch import load_cost_op
        cache = Path(os.environ.get('XDG_CACHE_HOME', str(self.directory/'cache')))
        self.candidate = load_cost_op(package_root,cache/'weave-native-cost'/f'rank{rank}')
        self.candidate.weave_prepare(torch.bfloat16,True,False)

    def dispatch(self, layer, index, args, kwargs, *, observer=None, original_native=False, original=None):
        if self.mode == 'calibration' and self.calibrating:
            if index == 0:self.bindings[layer] = []
            self.bindings[layer].append((args,kwargs))
        if observer is not None:
            span = observer.span(ROLES[index])
        else:
            span = nullcontext()
        with span:
            if original_native:
                return original(*args,**kwargs)
            total = int(kwargs['size_m'])*int(kwargs['top_k'])
            if total%6 or not 1<=total//6<=32:
                raise RuntimeError('native cost requires exact supported decode geometry')
            batch = total//6
            expected_n,expected_k = (4096,4096 if index==0 else 2048)
            if (kwargs['size_n'],kwargs['size_k']) != (expected_n,expected_k):
                raise RuntimeError('native cost GEMM geometry differs from calibrated model')
            return self.candidate(*args,**kwargs,budget_state=self.states[layer,ROLES[index]],
                cost_plan=self.plans[layer],cost_control=self.controls[layer],
                cost_table=self.tables[self.layer_to_group[layer],batch],cost_role=index)

    def check_path(self, layer, batch):
        expected = self.profile['tables'][self.layer_to_group[layer]][str(batch)]['measurements']['captured_path']
        if self.path_tags.get(layer) != expected:
            raise RuntimeError('native cost actual MLP dependency path differs from calibration')

    def maybe_calibrate(self, layer, original, args, kwargs):
        torch = self.torch
        observer = getattr(self.model, 'weave_graph_observer', None)
        # DecodeCudaGraphRunner also runs model_capture_mode autotune BEFORE
        # distributed graph_capture enables PyNccl. Only capture_one's warmups
        # have the native communicator state needed for our nested layer graphs.
        if (observer is None or not observer.capturing
                or torch.cuda.is_current_stream_capturing()):return
        hidden = args[0] if args else kwargs['hidden_states']; batch=int(hidden.shape[0])
        group = self.layer_to_group[layer]
        if self.groups[group]['representative_layer'] != layer or str(batch) in self.completed[group]:return
        from .native_cost_calibration import calibrate_layer
        try:
            self.calibrating = True
            self.completed[group][str(batch)] = calibrate_layer(self,layer,batch,original,args,kwargs)
            if all(set(rows)=={str(b) for b in range(1,33)} for rows in self.completed.values()):
                profile = dict(schema_version=1,method=METHOD,policy_version=POLICY_VERSION,
                    numerical_policy='record_only',calibration_complete=True,identity=self.identity,
                    groups=self.groups,tables=self.completed)
                profile['profile_sha256']=profile_digest(profile)
                validate_profile(profile,self.identity)
                atomic_json(self.directory/f'native-cost-rank{self.rank}.json',profile)
                self.profile=profile; self.calibration_complete=True
        except Exception as exc:
            atomic_json(self.directory/f'native-cost-rank{self.rank}.json',dict(schema_version=1,
                method=METHOD,policy_version=POLICY_VERSION,numerical_policy='record_only',calibration_complete=False,
                identity=self.identity,groups=self.groups,tables=self.completed,error_type=type(exc).__name__))
            raise
        finally:
            self.calibrating = False
            self.bindings.pop(layer,None)

    def snapshot(self):
        self.verify()
        with self.torch.cuda.device(self.device):
            self.torch.cuda.synchronize(self.device)
            states=self.state_buffer.cpu().tolist();plans=self.plan_buffer.cpu().tolist()
        if any(plan[20] or plan[23] for plan in plans):
            raise RuntimeError('native cost planner recorded malformed routing/table/plan')
        return dict(schema_version=1,method=METHOD,rank=self.rank,world_size=4,layer_count=43,mode=self.mode,
            numerical_policy='record_only',calibration_complete=self.calibration_complete,
            source_sha256=self.identity['cost_source_sha256'],upstream_sha256=source_pins(),identity=self.identity,
            groups=self.groups,profile_sha256=self.profile['profile_sha256'] if self.profile else None,
            policy=dict(version=POLICY_VERSION,candidate_pairs=[list(p) for p in CANDIDATES]),
            rows=[state_row(i,role,states[i][j]) for i in range(43) for j,role in enumerate(ROLES)],
            plans=[dict(layer=i,values=values) for i,values in enumerate(plans)])

    def verify(self):
        verify_sources(self.package_root)
        if native_cost_source_sha256()!=self.identity['cost_source_sha256']:
            raise RuntimeError('native cost source changed during execution')


def install_native_cost(model,torch,rank,package_root,**options):
    runtime=NativeCostRuntime(model,torch,rank,package_root,**options)
    marlin=importlib.import_module('sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe')
    native=importlib.import_module('sglang.srt.models.deepseek_v2')
    install_cost_dispatch(model,marlin,native,runtime)
    model.weave_native_cost=runtime
    return runtime
