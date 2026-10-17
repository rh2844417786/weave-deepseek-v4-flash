"""Bounded capture of real replicated MoE inputs; persistence happens at idle.

The CPU lifecycle and manifest gates import no Torch. CUDA storage is allocated
once before service work; selected layer calls only enqueue device copies.
Raw .pt files and request data must stay in the server's private work directory.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid

from .calibration import atomic_json

SCHEMA_VERSION=1
KIND='weave_private_real_decode_trace'
IDENTITY_KEYS={'kernel_source_sha256','checkpoint_content_sha256','dataset_sha256',
               'capture_source_sha256','tokenizer_manifest_sha256','capture_id','image_id'}

# Hash all local text/config/code and conventional encoding assets, even files
# not currently selected by AutoTokenizer. Fixed names retain absence markers.
# Weight bytes have their independent checkpoint-content identity.
TOKENIZER_REQUIRED_PATHS=('tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                          'generation_config.json','tokenizer.model','vocab.json','merges.txt',
                          'added_tokens.json')
TOKENIZER_ASSET_SUFFIXES=('.json','.py','.txt','.model','.vocab','.merges','.tiktoken',
                         '.bpe','.spm','.sentencepiece','.jinja','.jinja2','.yaml','.yml')
TOKENIZER_ASSET_NAMES=('tokenizer','tokenization','vocab','merges','encoding')


@dataclass(frozen=True)
class CaptureConfig:
    sample_sequences: tuple=(64,128)
    layers: int=43
    batch: int=32
    hidden_size: int=4096
    top_k: int=6
    experts: int=256
    ep_size: int=4

    def __post_init__(self):
        if (tuple(self.sample_sequences),self.layers,self.batch,self.hidden_size,
                self.top_k,self.experts,self.ep_size)!=((64,128),43,32,4096,6,256,4):
            raise ValueError('capture requires samples64/128, all43 B32 H4096/top6/E256/EP4')

    def as_dict(self):
        result=asdict(self);result['sample_sequences']=list(self.sample_sequences)
        return result


def capture_source_sha256():
    root=Path(__file__).resolve().parents[2]
    names=('src/weave/replay_trace.py','src/weave/sglang_backend.py',
           'src/weave/sglang_ext/models/deepseek_v4.py','src/weave/evidence_server.py',
           'src/weave/ops.py','src/weave/weights.py','src/weave/runtime_contract.py','src/weave/staged_decode.py',
           'src/weave/hopper_decode.py','src/weave/hopper_finalize.py','src/weave/hopper_candidate.py',
           'configs/runtime.json','configs/sglang_adapter_sources.json',
           'scripts/run_replay_capture.py','scripts/run_native_ep.py',
           'scripts/run_concurrency_diagnostic.py')
    return hashlib.sha256(b''.join(name.encode()+b'\0'+(root/name).read_bytes() for name in names)).hexdigest()


def validate_identity(value,expected=None):
    if not isinstance(value,dict) or set(value)!=IDENTITY_KEYS:
        raise ValueError('capture identity fields differ from schema')
    if any(not isinstance(value[k],str) or not re.fullmatch('[0-9a-f]{64}',value[k])
           for k in IDENTITY_KEYS-{'image_id','capture_id'}):
        raise ValueError('capture identity requires full SHA256 digests')
    if not re.fullmatch('sha256:[0-9a-f]{64}',value['image_id']):
        raise ValueError('capture image identity must be content addressed')
    validate_capture_id(value['capture_id'])
    if expected is not None and value!=expected:
        raise ValueError('capture identity does not match current source/checkpoint/tokenizer/dataset/image/capture ID')


def validate_capture_id(value):
    try:
        parsed=uuid.UUID(value) if isinstance(value,str) else None
    except (ValueError,AttributeError):parsed=None
    if parsed is None or parsed.version!=4 or str(parsed)!=value:
        raise ValueError('capture ID must be a canonical UUID4')
    return value


def tokenizer_manifest(model_path):
    """Content identity of nonhidden local tokenizer/config/encoding assets.

    Includes every file with the listed suffixes (all local Python code), plus
    files named for tokenizer/tokenization/vocab/merges/encoding regardless of
    extension. Hidden/cache directories and weight shards are excluded. The
    manifest has only relative paths and content digests, never token contents.
    """
    root=Path(model_path)
    if not root.is_dir():raise ValueError('local model directory missing; no tokenizer download is permitted')
    files={name:None for name in TOKENIZER_REQUIRED_PATHS}
    for name in TOKENIZER_REQUIRED_PATHS:
        path=root/name
        if path.exists() and not path.is_file():raise ValueError('tokenizer asset path must be a regular file')
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if any(part.startswith('.') or part=='__pycache__' for part in relative.parts):continue
        name=path.name.lower()
        if path.suffix.lower() not in TOKENIZER_ASSET_SUFFIXES and not any(key in name for key in TOKENIZER_ASSET_NAMES):continue
        if not path.is_file():continue
        files[relative.as_posix()]=sha256_file(path)
    value={'schema_version':1,'policy':{'required_paths':list(TOKENIZER_REQUIRED_PATHS),
        'asset_suffixes':list(TOKENIZER_ASSET_SUFFIXES),'asset_name_fragments':list(TOKENIZER_ASSET_NAMES),
        'hidden_and_pycache_excluded':True},'files':dict(sorted(files.items()))}
    value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return value


def tokenizer_manifest_sha256(model_path):
    return tokenizer_manifest(model_path)['sha256']


class CaptureLifecycle:
    """Model-forward selection and coverage bookkeeping; no device reads."""
    def __init__(self,config,rank):
        if type(rank) is not int or not 0<=rank<config.ep_size:raise ValueError('invalid capture rank')
        self.config,self.rank=config,rank
        self.armed=self.sealed=self.active=self.failed=False
        self.eligible_forwards=0
        self.slot=None
        self.bitmap=[[False]*config.layers for _ in config.sample_sequences]
        self.lock=threading.RLock()

    def arm(self):
        with self.lock:
            if self.armed or self.sealed:raise ValueError('capture already armed or sealed')
            if self.active:raise ValueError('cannot arm during an active model forward')
            self.armed=True

    def begin(self,phase,rows,valid_rows):
        with self.lock:
            if self.sealed:raise ValueError('capture is sealed; stop the owned service')
            if self.active:raise ValueError('nested model forward during capture')
            if self.failed:raise ValueError('capture failed; start a fresh service')
            self.active=True;self.slot=None
            if self.armed and phase=='decode' and rows==self.config.batch and valid_rows==rows:
                self.eligible_forwards+=1
                if self.eligible_forwards in self.config.sample_sequences:
                    self.slot=self.config.sample_sequences.index(self.eligible_forwards)
            return self.slot

    def record_layer(self,layer):
        # Called only by the model thread after the selected GPU copy enqueues.
        if not self.active or self.slot is None:raise ValueError('no selected capture forward')
        if type(layer) is not int or not 0<=layer<self.config.layers:raise ValueError('invalid capture layer')
        if self.bitmap[self.slot][layer]:raise ValueError('duplicate capture layer')
        self.bitmap[self.slot][layer]=True

    def end(self,success=True):
        with self.lock:
            if not self.active:raise ValueError('no active model forward')
            incomplete=self.slot is not None and not all(self.bitmap[self.slot])
            self.active=False;self.slot=None
            if not success or incomplete:self.failed=True
            if success and incomplete:raise ValueError('incomplete selected forward layer coverage')

    def seal(self):
        with self.lock:
            if self.active:raise ValueError('cannot persist an active model forward; abort and drain first')
            if self.failed:raise ValueError('failed capture cannot persist')
            if not self.armed or not all(all(row) for row in self.bitmap):
                raise ValueError('incomplete capture coverage')
            self.sealed=True

    def snapshot(self):
        with self.lock:
            return {'armed':self.armed,'sealed':self.sealed,'active':self.active,'failed':self.failed,
                'eligible_forwards':self.eligible_forwards,
                'sample_sequences':list(self.config.sample_sequences),
                'captured_layers':[sum(row) for row in self.bitmap],
                'layer_bitmap':[list(row) for row in self.bitmap],
                # A later forward can be active while selected samples are
                # complete. The owner must abort/drain before seal/persist.
                'complete':not self.failed and all(all(row) for row in self.bitmap)}


def tensor_descriptors(config):
    base=[len(config.sample_sequences),config.layers,config.batch]
    return {'x':{'shape':base+[config.hidden_size],'dtype':'bfloat16'},
            'ids':{'shape':base+[config.top_k],'dtype':'int32'},
            'routing':{'shape':base+[config.top_k],'dtype':'float32'}}


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def validate_rank_manifest(value,expected_identity=None):
    fields={'schema_version','kind','rank','identity','config','coverage','tensor_file',
            'tensor_sha256','tensor_bytes','tensors','used_for_formal_performance'}
    if not isinstance(value,dict) or set(value)!=fields:raise ValueError('rank manifest fields differ from schema')
    config=CaptureConfig()
    if (value['schema_version']!=SCHEMA_VERSION or value['kind']!=KIND
            or value['used_for_formal_performance'] is not False or value['config']!=config.as_dict()):
        raise ValueError('invalid capture manifest schema/configuration')
    rank=value['rank']
    if type(rank) is not int or not 0<=rank<config.ep_size:raise ValueError('invalid capture rank')
    validate_identity(value['identity'],expected_identity)
    if value['tensor_file']!=f'rank{rank}.pt':raise ValueError('private tensor filename differs from rank')
    if not isinstance(value['tensor_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',value['tensor_sha256']):
        raise ValueError('tensor hash required')
    if type(value['tensor_bytes']) is not int or value['tensor_bytes']<=0:raise ValueError('invalid tensor size')
    if value['tensors']!=tensor_descriptors(config):raise ValueError('tensor shape/dtype mismatch')
    state=value['coverage']
    keys={'armed','sealed','active','failed','eligible_forwards','sample_sequences',
          'captured_layers','layer_bitmap','complete'}
    if (not isinstance(state,dict) or set(state)!=keys or state['armed'] is not True
            or state['active'] is not False or state['failed'] is not False or state['complete'] is not True
            or state['sealed'] is not True or type(state['eligible_forwards']) is not int
            or state['eligible_forwards']<128 or state['sample_sequences']!=[64,128]
            or state['captured_layers']!=[43,43] or state['layer_bitmap']!=[[True]*43,[True]*43]
            or any(type(bit) is not bool for row in state['layer_bitmap'] for bit in row)):
        raise ValueError('capture coverage incomplete or malformed')
    return value


def aggregate_manifests(values,expected_identity=None):
    if len(values)!=4 or {v.get('rank') for v in values}!={0,1,2,3}:
        raise ValueError('four distinct rank manifests required')
    expected_identity=expected_identity or values[0]['identity']
    for value in values:validate_rank_manifest(value,expected_identity)
    return {'schema_version':SCHEMA_VERSION,'kind':KIND,'identity':dict(expected_identity),
        'config':CaptureConfig().as_dict(),'rank_count':4,'captured_layer_instances':344,
        'used_for_formal_performance':False,'ranks':sorted(values,key=lambda v:v['rank'])}


def validate_capture_directory(directory,expected_identity=None):
    """Verify committed manifests and private file hashes without importing Torch."""
    directory=Path(directory)
    values=[]
    for rank in range(4):
        value=json.loads((directory/f'rank{rank}.json').read_text())
        validate_rank_manifest(value,expected_identity)
        if value['rank']!=rank:raise ValueError('rank manifest filename mismatch')
        path=directory/value['tensor_file']
        if path.is_symlink() or path.stat().st_size!=value['tensor_bytes']:raise ValueError('private tensor size/path mismatch')
        if sha256_file(path)!=value['tensor_sha256']:raise ValueError('private tensor hash mismatch')
        values.append(value)
    result=aggregate_manifests(values,expected_identity)
    if json.loads((directory/'capture_manifest.json').read_text())!=result:
        raise ValueError('capture commit manifest mismatch')
    return result


def validate_reused_capture(directory,current_identity):
    """Keep the archived run UUID; recompute/pass every other current identity."""
    if not isinstance(current_identity,dict) or set(current_identity)!=IDENTITY_KEYS-{'capture_id'}:
        raise ValueError('current reuse identity requires every field except the archived capture ID')
    try:
        archive=json.loads((Path(directory)/'capture_manifest.json').read_text())
        capture_id=archive['identity']['capture_id']
    except (KeyError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('archived capture identity missing or malformed') from exc
    expected={**current_identity,'capture_id':capture_id}
    validate_identity(expected)
    return validate_capture_directory(directory,expected)


def validate_tensor_payload(payload,manifest,torch):
    validate_rank_manifest(manifest)
    if (not isinstance(payload,dict) or set(payload)!={'schema_version','rank','identity','config','x','ids','routing'}
            or payload['schema_version']!=SCHEMA_VERSION or payload['rank']!=manifest['rank']
            or payload['identity']!=manifest['identity'] or payload['config']!=manifest['config']):
        raise ValueError('private tensor payload identity/schema mismatch')
    for name,desc in manifest['tensors'].items():
        tensor=payload[name]
        if (not isinstance(tensor,torch.Tensor) or list(tensor.shape)!=desc['shape']
                or tensor.dtype!=getattr(torch,desc['dtype']) or not tensor.is_contiguous()):
            raise ValueError(f'private {name} shape/dtype/contiguity mismatch')
    if not torch.isfinite(payload['x']).all().item() or not torch.isfinite(payload['routing']).all().item():
        raise ValueError('private inputs/routing contain nonfinite values')
    ids=payload['ids']
    if (ids<0).any().item() or (ids>=256).any().item():raise ValueError('private expert IDs outside global range')
    ordered=ids.sort(dim=-1).values
    if (ordered[...,1:]==ordered[...,:-1]).any().item():raise ValueError('private per-token expert IDs must be unique')
    return payload


def load_rank_trace(directory,rank,expected_identity=None):
    """Load only a validated weights-only-safe rank payload (CPU map location)."""
    import torch
    if type(rank) is not int or not 0<=rank<4:raise ValueError('invalid trace rank')
    directory=Path(directory)
    summary=validate_capture_directory(directory,expected_identity)
    manifest=summary['ranks'][rank]
    payload=torch.load(directory/manifest['tensor_file'],map_location='cpu',weights_only=True)
    return validate_tensor_payload(payload,manifest,torch)


class ReplayCapture:
    def __init__(self,torch,device,rank,directory,identity):
        validate_identity(identity)
        self.torch,self.device=torch,torch.device('cuda',device)
        self.directory,self.identity=Path(directory),dict(identity)
        self.config=CaptureConfig();self.lifecycle=CaptureLifecycle(self.config,rank)
        self.buffers={name:torch.empty(desc['shape'],device=self.device,dtype=getattr(torch,desc['dtype']))
                      for name,desc in tensor_descriptors(self.config).items()}
        self.persisted=None

    def record_layer(self,layer,x,ids,routing):
        slot=self.lifecycle.slot
        if slot is None:return
        if self.torch.cuda.is_current_stream_capturing():raise RuntimeError('real trace capture requires eager execution')
        # Shape/type checks inspect CPU metadata only. Routing dtypes are copied
        # into fixed int32/FP32 destinations by CUDA, without a host round trip.
        for value,shape in ((x,(32,4096)),(ids,(32,6)),(routing,(32,6))):
            if tuple(value.shape)!=shape or value.device!=self.device:raise ValueError('capture requires full replicated CUDA B32 inputs')
        if x.dtype!=self.torch.bfloat16 or ids.dtype not in (self.torch.int32,self.torch.int64):
            raise ValueError('capture requires BF16 x and integer global expert IDs')
        if routing.dtype not in (self.torch.float32,self.torch.bfloat16,self.torch.float16):
            raise ValueError('capture requires floating point routing weights')
        if type(layer) is not int or not 0<=layer<43 or self.lifecycle.bitmap[slot][layer]:
            raise ValueError('invalid or duplicate capture layer')
        for name,value in (('x',x),('ids',ids),('routing',routing)):
            self.buffers[name][slot,layer].copy_(value,non_blocking=True)
        self.lifecycle.record_layer(layer)

    def persist(self):
        """Called after owner abort/drain; seal blocks a racing new forward."""
        with self.lifecycle.lock:
            self.lifecycle.seal()
            if self.persisted is not None:return self.persisted
            self.torch.cuda.set_device(self.device)
            self.torch.cuda.synchronize(self.device)
            rank=self.lifecycle.rank
            payload={'schema_version':SCHEMA_VERSION,'rank':rank,'identity':self.identity,
                     'config':self.config.as_dict(),**{name:value.cpu() for name,value in self.buffers.items()}}
            manifest={'schema_version':SCHEMA_VERSION,'kind':KIND,'rank':rank,
                'identity':self.identity,'config':self.config.as_dict(),'coverage':self.lifecycle.snapshot(),
                'tensors':tensor_descriptors(self.config),'tensor_file':f'rank{rank}.pt',
                'tensor_sha256':'0'*64,'tensor_bytes':1,'used_for_formal_performance':False}
            validate_tensor_payload(payload,manifest,self.torch)
            self.directory.mkdir(parents=True,exist_ok=True)
            destination=self.directory/manifest['tensor_file']
            if destination.exists() or (self.directory/f'rank{rank}.json').exists():
                raise ValueError('capture output already exists; use validated reuse or a fresh directory')
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory,prefix=f'.rank{rank}.',delete=False) as stream:
                    temporary=Path(stream.name);self.torch.save(payload,stream)
                    stream.flush();os.fchmod(stream.fileno(),0o664);os.fsync(stream.fileno())
                manifest['tensor_sha256']=sha256_file(temporary)
                manifest['tensor_bytes']=temporary.stat().st_size
                os.replace(temporary,destination);temporary=None
                atomic_json(self.directory/f'rank{rank}.json',manifest)
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)
            self.persisted=manifest
            return manifest
