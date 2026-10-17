"""Pinned, lazy server output timing instrumentation; standard library only."""
import hashlib
import importlib.abc
import importlib.machinery
import math
import os
from pathlib import Path
import sys

BOUNDARY = 'tokenizer_manager_output_handling'
SOURCE = 'sglang_api_request_time_stats'
TARGET = 'sglang.srt.observability.req_time_stats'
NATIVE_SHA256 = 'f4ac4d7c0f01c0fa4b147201453a88471eae5ccc4b2196de87788acfef87ebd9'
TOKENIZER_SHA256 = '0385c00527d2c8ad53021a2170e73a6a3eda5dd94a54c3e86a549c56b2fd4f61'


def source_sha256():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), root/'timing_hooks/sitecustomize.py'):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def timing_metadata(stats, completion_tokens, *, native_source_sha256=NATIVE_SHA256):
    if type(completion_tokens) is not int or completion_tokens < 1:
        raise ValueError('timing requires positive actual completion token count')
    received, first, finished = stats.created_time, stats.first_token_time, stats.finished_time
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (received, first, finished)):
        raise ValueError('missing finite positive native timestamps')
    if received > first or first > finished:
        raise ValueError('native timestamps are not monotonic')
    last = first if completion_tokens == 1 else finished
    return {'request_received_s': received, 'first_token_ready_s': first,
            'last_token_ready_s': last, 'output_tokens': completion_tokens,
            'boundary': BOUNDARY, 'clock': 'perf_counter', 'source': SOURCE,
            'source_sha256': source_sha256(), 'native_source_sha256': native_source_sha256}


def patch_timing_class(cls, native_source_sha256=NATIVE_SHA256):
    if getattr(cls, '_weave_timing_installed', False):
        return
    original = cls.convert_to_output_meta_info

    def convert(self, scheduler_time_stats=None, completion_tokens=0):
        meta = original(self, scheduler_time_stats, completion_tokens)
        try:
            meta['weave_timing'] = timing_metadata(self, completion_tokens,
                native_source_sha256=native_source_sha256)
        except ValueError as error:
            # Empty/aborted outputs are diagnostic only and cannot become perf rows.
            meta['weave_timing'] = {'valid': False, 'error': str(error), 'boundary': BOUNDARY,
                'clock': 'perf_counter', 'source': SOURCE, 'source_sha256': source_sha256(),
                'native_source_sha256': native_source_sha256, 'output_tokens': completion_tokens}
        if os.environ.get('WEAVE_STALL_DIAGNOSTIC') == '1':
            meta['weave_stall_timing'] = stall_timing_metadata(self, scheduler_time_stats)
        return meta

    cls.convert_to_output_meta_info = convert
    cls._weave_timing_installed = True


class TimingLoader(importlib.abc.Loader):
    def __init__(self, loader, origin):
        self.loader, self.origin = loader, Path(origin)

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        tokenizer = self.origin.parent.parent/'managers/tokenizer_manager.py'
        for path, expected in ((self.origin, NATIVE_SHA256), (tokenizer, TOKENIZER_SHA256)):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ImportError(f'Weave timing requires pinned SGLang 0.5.17 source: {path}')
        self.loader.exec_module(module)
        patch_timing_class(module.APIServerReqTimeStats)
        if os.environ.get('WEAVE_STALL_DIAGNOSTIC') == '1':
            patch_scheduler_timing_class(module.SchedulerReqTimeStats)


class TimingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or spec.origin is None:
            raise ImportError('pinned SGLang request timing source missing')
        spec.loader = TimingLoader(spec.loader, spec.origin)
        return spec


def install_timing_hook():
    if TARGET in sys.modules:
        raise RuntimeError('timing hook must be installed before SGLang import')
    if not any(isinstance(finder, TimingFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, TimingFinder())


API_STALL_FIELDS = ('created_time', 'tokenize_finish_time', 'api_server_dispatch_finish_time', 'first_token_time')
SCHEDULER_STALL_FIELDS = ('wait_queue_entry_time', 'forward_entry_time', 'prefill_finished_time')


def stall_timing_metadata(api, scheduler):
    result = {'clock': 'CLOCK_MONOTONIC', 'missing': []}
    for obj, fields in ((api, API_STALL_FIELDS), (scheduler, SCHEDULER_STALL_FIELDS)):
        for field in fields:
            value = getattr(obj, field, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                result[field] = value
            else:
                result['missing'].append(field)
    return result


def patch_scheduler_timing_class(cls):
    if getattr(cls, '_weave_stall_timing_installed', False):
        return
    original = cls.convert_to_output_meta_info
    def convert(self):
        meta = original(self)
        meta['weave_stall_timing'] = stall_timing_metadata(None, self)
        return meta
    cls.convert_to_output_meta_info = convert
    cls._weave_stall_timing_installed = True
