#!/usr/bin/env python3
"""CPU-only private logits comparison; finite differences are always record-only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weave.calibration import atomic_json
from src.weave.c32_serving import ClientError
from src.weave.fixed_prefix_data import digest, load_teacher

ARMS = ('native_a', 'native_b', 'p0_full', 'p0_events', 'p0_buffers')
RANKS = tuple(range(4))
SOURCE_FIELDS = ('fixed_prefix_source_sha256', 'native_sources', 'image_id',
                 'sampler_source_closure_sha256')


def logit_metrics(reference, candidate):
    """Float64 CPU arithmetic; no agreement or numerical acceptance cutoff."""
    import numpy as np
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 2:
        raise ClientError('logits_shape_mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ClientError('nonfinite_original_logits')
    delta = b-a
    absolute = np.abs(delta)
    reference_norm = float(np.linalg.norm(a))
    candidate_norm = float(np.linalg.norm(b))
    difference_norm = float(np.linalg.norm(delta))
    relative = difference_norm/reference_norm if reference_norm else (0. if difference_norm == 0 else None)
    a_top, b_top = a.argmax(axis=1), b.argmax(axis=1)

    def margins(value):
        top2 = np.partition(value, -2, axis=1)[:, -2:]
        margin = top2.max(axis=1)-top2.min(axis=1)
        return {'min': float(margin.min()), 'max': float(margin.max()), 'mean': float(margin.mean())}

    cosine = float(np.vdot(a, b)/(reference_norm*candidate_norm)) if reference_norm and candidate_norm else None
    return {'relative_l2': relative,
            'relative_l2_status': 'defined' if relative is not None else 'undefined_zero_reference_norm',
            'max_abs': float(absolute.max()), 'mean_abs': float(absolute.mean()),
            'cosine_similarity': cosine,
            'top1_matched': int((a_top == b_top).sum()), 'top1_mismatched': int((a_top != b_top).sum()),
            'reference_top1_sha256': digest(a_top.tolist()), 'candidate_top1_sha256': digest(b_top.tolist()),
            'reference_top1_margin': margins(a), 'candidate_top1_margin': margins(b)}


def difference_summary(records):
    ordered = sorted(records, key=lambda item: (item['step'], item['rank']))

    def first(predicate):
        for row in ordered:
            if predicate(row):
                return {key: row[key] for key in ('step', 'rank', 'mode')}
        return None

    return {'first_observed_arithmetic_difference': first(lambda row: row['max_abs'] > 0),
            'first_observed_top1_difference': first(lambda row: row['top1_mismatched'] > 0),
            'observation_order': 'output_position_then_rank_not_cross_rank_wall_clock',
            'top1_matched': sum(row['top1_matched'] for row in records if 'top1_matched' in row),
            'top1_mismatched': sum(row['top1_mismatched'] for row in records)}


def _read_json(path):
    if path.is_symlink() or not path.is_file():
        raise ClientError('required_private_evidence_missing_or_symlink')
    try:
        value = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        raise ClientError('invalid_private_evidence_json') from None
    if not isinstance(value, dict):
        raise ClientError('private_evidence_object_required')
    return value


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_config(config):
    if (config.get('schema_version') != 1 or not _sha(config.get('selection_sha256'))
            or type(config.get('steps')) is not int or config['steps'] < 1
            or type(config.get('vocab_size')) is not int or config['vocab_size'] < 2
            or not isinstance(config.get('requests'), list) or len(config['requests']) != 32):
        raise ClientError('invalid_fixed_prefix_config')
    indices = []
    for slot, row in enumerate(config['requests']):
        if (not isinstance(row, dict) or row.get('row_slot') != slot
                or type(row.get('original_row_index')) is not int
                or not 0 <= row['original_row_index'] < 448
                or type(row.get('input_tokens')) is not int or row['input_tokens'] < 1
                or not _sha(row.get('input_ids_sha256')) or not _sha(row.get('prompt_sha256'))):
            raise ClientError('invalid_fixed_prefix_request_commitment')
        indices.append(row['original_row_index'])
    if len(set(indices)) != 32:
        raise ClientError('duplicate_fixed_prefix_request_commitment')
    return config


def validate_proof(proof, config, teacher, rank):
    """Check actual complete positions and input commitments, independent of logits."""
    steps = config['steps']
    if (proof.get('schema_version') != 1 or proof.get('status') != 'passed'
            or proof.get('numerical_policy') != 'record_only' or proof.get('phase') != 'measure'
            or proof.get('rank') != rank or proof.get('world_size') != len(RANKS)
            or proof.get('selection_sha256') != config['selection_sha256']
            or proof.get('config_sha256') != digest(config)
            or proof.get('teacher_sha256') != teacher['sha256']
            or proof.get('expected_steps') != steps
            or proof.get('completed_steps') != list(range(steps))
            or not isinstance(proof.get('records'), list) or len(proof['records']) != steps):
        raise ClientError('incomplete_or_mismatched_measurement_proof')
    if (not _sha(proof.get('fixed_prefix_source_sha256'))
            or not _sha(proof.get('sampler_source_closure_sha256'))
            or not isinstance(proof.get('image_id'), str) or not proof['image_id']
            or not isinstance(proof.get('native_sources'), dict) or not proof['native_sources']
            or any(not isinstance(key, str) or not _sha(value)
                   for key, value in proof['native_sources'].items())):
        raise ClientError('measurement_source_identity_required')
    if proof['sampler_source_closure_sha256'] != digest(proof['native_sources']):
        raise ClientError('measurement_source_closure_mismatch')
    by_step = {}
    for row in proof['records']:
        if not isinstance(row, dict):
            raise ClientError('invalid_measurement_record')
        step = row.get('step')
        order = row.get('row_slot_order')
        if (type(step) is not int or step not in range(steps) or step in by_step
                or row.get('batch_size') != 32
                or row.get('mode') != ('extend' if step == 0 else 'decode')
                or not isinstance(order, list) or any(type(slot) is not int for slot in order)
                or sorted(order) != list(range(32))
                or row.get('original_logits_finite') is not True or row.get('captured') is not False):
            raise ClientError('invalid_batch_row_or_position_coverage')
        for key in ('logits_tensor_sha256', 'logits_file_sha256', 'input_ids_sha256',
                    'req_pool_indices_sha256', 'out_cache_loc_sha256', 'native_argmax_sha256'):
            if not _sha(row.get(key)):
                raise ClientError('measurement_commitment_required')
        seq_lens = [config['requests'][slot]['input_tokens']+step for slot in order]
        input_commitments = [config['requests'][slot]['input_ids_sha256'] if step == 0 else
                             digest([teacher['output_ids'][slot][step-1]]) for slot in order]
        if (row.get('seq_lens') != seq_lens or row.get('positions') != [n-1 for n in seq_lens]
                or row.get('input_ids_by_row_sha256') != input_commitments):
            raise ClientError('fixed_prefix_input_or_position_mismatch')
        if step and row['input_ids_sha256'] != digest([teacher['output_ids'][slot][step-1] for slot in order]):
            raise ClientError('fixed_prefix_flat_input_mismatch')
        if row.get('sampled_ids_sha256') != digest([teacher['output_ids'][slot][step] for slot in order]):
            raise ClientError('fixed_prefix_sampled_input_commitment_mismatch')
        by_step[step] = row
    return by_step


def load_snapshot(path, record, vocab_size, *, torch_module=None):
    """Only tensor payloads; safe deserialization and CPU placement are mandatory."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            raise ClientError('torch_missing_use_existing_SGLang_image_no_online_install') from None
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ClientError('private_logits_tensor_missing_or_symlink')
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['logits_file_sha256']:
        raise ClientError('private_logits_file_digest_mismatch')
    try:
        tensor = torch_module.load(path, map_location='cpu', weights_only=True)
    except Exception:
        raise ClientError('private_logits_tensor_load_failed') from None
    if (not torch_module.is_tensor(tensor) or tensor.device.type != 'cpu'
            or list(tensor.shape) != [32, vocab_size] or not tensor.is_floating_point()):
        raise ClientError('private_logits_tensor_shape_or_type_mismatch')
    from src.weave.fixed_prefix_runtime import tensor_sha256
    if tensor_sha256(tensor) != record['logits_tensor_sha256']:
        raise ClientError('private_logits_tensor_digest_mismatch')
    value = tensor.detach().double().numpy()
    import numpy as np
    if not np.isfinite(value).all():
        raise ClientError('nonfinite_original_logits')
    if digest(value.argmax(axis=1).tolist()) != record['native_argmax_sha256']:
        raise ClientError('native_argmax_commitment_mismatch')
    return value[np.argsort(record['row_slot_order'])]


def compare_run(root, teacher_path=None):
    root = Path(root)
    teacher_path = Path(teacher_path) if teacher_path is not None else root/'private'/'teacher.json'
    configs = {arm: validate_config(_read_json(root/arm/'private'/'config.json')) for arm in ARMS}
    config = configs['native_a']
    if any(value != config for value in configs.values()):
        raise ClientError('measurement_config_mismatch')
    teacher = load_teacher(teacher_path, selection_sha256=config['selection_sha256'],
                           steps=config['steps'], vocab_size=config['vocab_size'])
    proofs, records = {}, {}
    for arm in ARMS:
        expected_names = {f'rank{rank}-step{step:03d}.pt'
                          for rank in RANKS for step in range(config['steps'])}
        observed = {path.name for path in (root/arm/'private'/'measure').glob('*.pt')}
        if expected_names != observed:
            raise ClientError('exact_measurement_tensor_inventory_required')
        proofs[arm], records[arm] = {}, {}
        for rank in RANKS:
            proof = _read_json(root/arm/'private'/f'measure-rank{rank}.json')
            proofs[arm][rank] = proof
            records[arm][rank] = validate_proof(proof, config, teacher, rank)
    source = {key: proofs['native_a'][0][key] for key in SOURCE_FIELDS}
    if any({key: proof[key] for key in SOURCE_FIELDS} != source
           for arm in ARMS for proof in proofs[arm].values()):
        raise ClientError('measurement_source_identity_mismatch')
    comparisons = []
    # Consume only one corresponding tensor pair at a time. Native repeatability
    # comes first, with each rank retaining its own arithmetic evidence.
    for arm in ARMS[1:]:
        rows = []
        for step in range(config['steps']):
            for rank in RANKS:
                native_record, other_record = records['native_a'][rank][step], records[arm][rank][step]
                name = f'rank{rank}-step{step:03d}.pt'
                reference = load_snapshot(root/'native_a'/'private'/'measure'/name,
                                          native_record, config['vocab_size'])
                candidate = load_snapshot(root/arm/'private'/'measure'/name,
                                          other_record, config['vocab_size'])
                rows.append({'step': step, 'rank': rank, 'mode': native_record['mode'],
                             'output_position_kind': 'prefill_prediction' if step == 0 else 'decode_prediction',
                             'batch_size': 32, 'vocab_size': config['vocab_size'],
                             'row_order_matched':native_record['row_slot_order']==other_record['row_slot_order'],
                             'request_pool_slots_matched':native_record['req_pool_indices_sha256']==other_record['req_pool_indices_sha256'],
                             'cache_write_slots_matched':native_record['out_cache_loc_sha256']==other_record['out_cache_loc_sha256'],
                             'reference_logits_sha256': native_record['logits_tensor_sha256'],
                             'candidate_logits_sha256': other_record['logits_tensor_sha256'],
                             **logit_metrics(reference, candidate)})
        comparisons.append({'reference_arm': 'native_a', 'candidate_arm': arm,
                            'status': 'comparison_complete', 'numerical_policy': 'record_only',
                            **difference_summary(rows), 'records': rows})
    return {'schema_version': 1, 'status': 'comparison_complete',
            'numerical_policy': 'record_only', 'raw_content_exported': False,
            'selection_sha256': config['selection_sha256'], 'teacher_sha256': teacher['sha256'],
            'steps': config['steps'], 'ranks': list(RANKS), 'request_count': 32,
            'vocab_size': config['vocab_size'], 'source_identity': source,
            'comparison_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'metric_dtype': 'float64_cpu', 'comparisons': comparisons,
            'scope': 'original_pre_forcing_logits_at_identical_logical_prefix_and_position',
            'kv_scope': 'logical_positions_controlled_pool_slot_indices_committed_KV_contents_not_asserted_equal'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema_version': 1, 'status': 'preview_not_executed',
                          'arms': list(ARMS), 'ranks': list(RANKS),
                          'numerical_policy': 'record_only', 'raw_content_exported': False}))
        return 0
    try:
        if not args.root.is_absolute() or not args.output.is_absolute() or (
                args.teacher is not None and not args.teacher.is_absolute()):
            raise ClientError('absolute_comparison_paths_required')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        result = compare_run(args.root, args.teacher)
    except Exception as error:
        result = {'schema_version': 1, 'status': 'failed', 'numerical_policy': 'record_only',
                  'raw_content_exported': False,
                  'error_codes': [str(error) if isinstance(error, ClientError) else 'fixed_prefix_comparison_error']}
    atomic_json(args.output, result)
    print(json.dumps({'status': result['status'], 'error_codes': result.get('error_codes', [])}))
    return 0 if result['status'] == 'comparison_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
