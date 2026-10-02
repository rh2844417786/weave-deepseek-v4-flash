#!/usr/bin/env python3
"""Idle service acceptance; uses one natural GPQA input, not a speed measurement."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weave.service_acceptance import validate_coverage_delta, compare_first_token, NumericalComparisonError

DIRECT = build_opener(ProxyHandler({}))


def http_json(url, data=None, timeout=30):
    body = json.dumps(data).encode() if data is not None else None
    request = Request(url, data=body, headers={'Content-Type': 'application/json'})
    with DIRECT.open(request, timeout=timeout) as response:
        payload = response.read()
        if not payload:
            return None
        return json.loads(payload) if 'json' in response.headers.get('Content-Type', '') else payload.decode()


def coverage(base):
    return [http_json(f'http://127.0.0.1:{base+rank}/coverage', timeout=120) for rank in range(4)]


def wait_ready(url, *, timeout=1800, process=None):
    deadline = time.monotonic()+timeout
    last_error = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'server launcher exited before readiness: {process.returncode}')
        try:
            with DIRECT.open(url+'/health', timeout=3) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f'service not ready in {timeout}s: {last_error}')


def check_service(url, records_path, output, *, weave=False, graph=False, evidence_port=30110, native=None,
                  expected_source=None, expected_kernel=None):
    prepared = json.loads((records_path.parent/'manifest.json').read_text())
    pinned = json.loads((ROOT/'data/gpqa/deepseek_model_manifest.json').read_text())
    if (prepared.get('record_sha256') != hashlib.sha256(records_path.read_bytes()).hexdigest()
            or prepared.get('count') != 448 or prepared.get('reasoning_mode') != 'Think High'
            or prepared.get('max_new_tokens') != 32768 or prepared.get('normal_eos_allowed') is not True
            or prepared.get('model_revision') != pinned['revision']
            or prepared.get('pinned_model_metadata_sha256') != pinned['sha256']):
        raise ValueError('pinned 448-row Think High GPQA manifest required')
    with records_path.open() as stream:
        record = json.loads(stream.readline())
    ids = record['input_token_ids']
    input_sha = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    before = coverage(evidence_port) if weave else None
    if weave and ((expected_source and any(v['source_sha256'] != expected_source for v in before))
                  or (expected_kernel and any(v['kernel'] != expected_kernel for v in before))):
        raise ValueError('service CUDA source/kernel differs from the passed checkpoint gate')
    # Flush cached prefixes so this request must execute a real prefill.
    http_json(url+'/flush_cache')
    request = {'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': 8,
                                                   'ignore_eos': True},
               'return_logprob': True, 'top_logprobs_num': 10, 'stream': False}
    report = {'status': 'running', 'scope': 'one_input_correctness_and_43_layer_execution',
              'input_sha256': input_sha, 'row_index': record['row_index'],
              'request_settings': {k: v for k, v in request.items() if k != 'input_ids'},
              'request': request, 'prepared_manifest': prepared,
              'graph_required': graph, 'performance': 'not_measured'}
    # Store decodable input/output only under ignored work, never in git reports.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    try:
        report['response'] = http_json(url+'/generate', request, timeout=900)
        token_ids = report['response'].get('output_ids')
        if not token_ids or len(token_ids) != 8:
            raise ValueError('acceptance requires eight actual generated token IDs')
        if weave:
            after = coverage(evidence_port)
            report['coverage_before'], report['coverage_after'] = before, after
            report['coverage_check'] = validate_coverage_delta(before, after, require_graph=graph)
        if native:
            baseline = json.loads(native.read_text())
            if baseline['response']['output_ids'] != token_ids:
                raise ValueError('native and Weave generated token IDs differ in the eight-token acceptance sequence')
            report['native_comparison'] = compare_first_token(baseline, report)
            report['native_comparison']['matched_generated_tokens'] = len(token_ids)
        report['status'] = 'passed'
    except Exception as exc:
        if isinstance(exc, NumericalComparisonError):
            report['native_comparison'] = dict(exc.diagnostics, matched_generated_tokens=len(token_ids))
        report['status'], report['error'] = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        output.write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=30001)
    parser.add_argument('--records', type=Path, default=ROOT/'data/gpqa/prepared/deepseek-v4-flash/main.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weave', action='store_true')
    parser.add_argument('--require-graph', action='store_true')
    parser.add_argument('--evidence-port-base', type=int, default=30110)
    parser.add_argument('--native-result', type=Path)
    args = parser.parse_args()
    if args.require_graph and not args.weave:
        parser.error('graph coverage check requires the Weave adapter')
    url = f'http://127.0.0.1:{args.port}'
    wait_ready(url)
    result = check_service(url, args.records, args.output, weave=args.weave,
                           graph=args.require_graph, evidence_port=args.evidence_port_base,
                           native=args.native_result)
    print(json.dumps({'status': result['status'], 'evidence': str(args.output)}))


if __name__ == '__main__':
    main()
