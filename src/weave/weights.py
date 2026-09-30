"""Load only this rank's original packed expert weights, before native repacking."""
import json
from pathlib import Path


def load_raw_experts(model_path, layer_id, rank, ep_size, hidden_size=4096,
                     intermediate_size=2048, experts=256, device=None):
    import torch
    from safetensors import safe_open
    path = Path(model_path)
    index = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
    if experts % ep_size:
        raise ValueError('expert count must divide EP size')
    local = experts // ep_size
    h, i = hidden_size, intermediate_size
    specs = [('w1', 'weight', (local, i, h//2)), ('w3', 'weight', (local, i, h//2)),
             ('w2', 'weight', (local, h, i//2)), ('w1', 'scale', (local, i, h//32)),
             ('w3', 'scale', (local, i, h//32)), ('w2', 'scale', (local, h, i//32))]
    buffers = [torch.empty(shape, dtype=torch.uint8, device=device) for _, _, shape in specs]
    reads = {}
    for local_id in range(local):
        expert = rank * local + local_id
        for target, (projection, suffix, shape) in enumerate(specs):
            tail = f'layers.{layer_id}.ffn.experts.{expert}.{projection}.{suffix}'
            names = [name for name in (tail, 'model.' + tail) if name in index]
            if len(names) != 1:
                raise ValueError(f'expected one raw checkpoint key for {tail}')
            name = names[0]
            shard = index[name]
            if Path(shard).name != shard:
                raise ValueError('unsafe shard name')
            reads.setdefault(shard, []).append((name, target, local_id, shape[1:], suffix))
    for shard, requests in reads.items():
        with safe_open(path / shard, framework='pt', device='cpu') as handle:
            for name, target, local_id, shape, kind in requests:
                value = handle.get_tensor(name)
                if tuple(value.shape) != tuple(shape):
                    raise ValueError(f'unexpected packed expert shape: {name} {tuple(value.shape)}')
                legal = (torch.int8, torch.uint8) if kind == 'weight' else (torch.float8_e8m0fnu, torch.uint8)
                if value.dtype not in legal:
                    raise ValueError(f'unexpected storage dtype: {name} {value.dtype}')
                buffers[target][local_id].copy_(value.view(torch.uint8))
    return tuple(buffers)
