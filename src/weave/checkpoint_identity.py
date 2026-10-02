"""Streaming content identity for an original checkpoint, without Torch imports."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(path):
    path = Path(path)
    index = json.loads((path/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    if not shards or any(Path(name).name != name or name in ('.', '..') for name in shards):
        raise ValueError('checkpoint shard names must be safe basenames')
    return ['config.json', 'model.safetensors.index.json'] + shards


def checkpoint_stats(path):
    result = {}
    for name in checkpoint_files(path):
        stat = (Path(path)/name).stat()
        result[name] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    return result


def checkpoint_identity(path, *, progress=None):
    before = checkpoint_stats(path)
    hashes = {}
    for number, name in enumerate(before, 1):
        hashes[name] = sha256_file(Path(path)/name)
        if progress:
            progress(number, len(before), name)
    if checkpoint_stats(path) != before:
        raise ValueError('checkpoint changed while hashing')
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'sha256': identity, 'files_sha256': hashes, 'verified_stats': before,
            'scope': 'config_index_and_all_indexed_shard_contents'}
