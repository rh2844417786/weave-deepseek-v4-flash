"""Private trace lifecycle and identity gates require no Torch or CUDA."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.weave.replay_trace import (CaptureConfig, CaptureLifecycle,
    aggregate_manifests, validate_rank_manifest, validate_capture_directory)


def identity():
    return {'kernel_source_sha256':'a'*64, 'checkpoint_content_sha256':'b'*64,
            'dataset_sha256':'c'*64, 'capture_source_sha256':'d'*64,
            'tokenizer_manifest_sha256':'1'*64,
            'capture_id':'11111111-1111-4111-8111-111111111111',
            'image_id':'sha256:'+'e'*64}


def complete_lifecycle():
    life=CaptureLifecycle(CaptureConfig(),rank=0)
    life.arm()
    for _ in range(128):
        slot=life.begin('decode',32,32)
        if slot is not None:
            for layer in range(43): life.record_layer(layer)
        life.end()
    return life


def manifest(rank=0):
    life=complete_lifecycle()
    life.seal()
    return {'schema_version':1, 'kind':'weave_private_real_decode_trace',
        'rank':rank, 'identity':identity(), 'config':CaptureConfig().as_dict(),
        'coverage':life.snapshot(), 'tensor_file':f'rank{rank}.pt',
        'tensor_sha256':'f'*64, 'tensor_bytes':123,
        'tensors':{'x':{'shape':[2,43,32,4096],'dtype':'bfloat16'},
                   'ids':{'shape':[2,43,32,6],'dtype':'int32'},
                   'routing':{'shape':[2,43,32,6],'dtype':'float32'}},
        'used_for_formal_performance':False}


class ReplayTraceTest(unittest.TestCase):
    def test_tokenizer_manifest_binds_content_presence_and_local_python_without_weight_bytes(self):
        from src.weave.replay_trace import tokenizer_manifest,tokenizer_manifest_sha256
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'tokenizer.json').write_text('{"version":1}')
            (root/'tokenizer_config.json').write_text('{}')
            (root/'generation_config.json').write_text('{}')
            (root/'encoding').mkdir()
            (root/'encoding'/'custom.py').write_text('mapping = 1\n')
            (root/'model.safetensors').write_bytes(b'weight-v1')
            baseline=tokenizer_manifest_sha256(root)
            manifest_value=tokenizer_manifest(root)
            self.assertIsNone(manifest_value['files']['special_tokens_map.json'])
            self.assertIn('encoding/custom.py',manifest_value['files'])
            self.assertNotIn('model.safetensors',manifest_value['files'])
            (root/'model.safetensors').write_bytes(b'weight-v2')
            self.assertEqual(tokenizer_manifest_sha256(root),baseline)
            for name,new in [('tokenizer.json','{"version":2}'),('tokenizer_config.json','{"changed":1}'),
                             ('generation_config.json','{"temperature":0.5}'),('encoding/custom.py','mapping = 2\n')]:
                original=(root/name).read_text();(root/name).write_text(new)
                self.assertNotEqual(tokenizer_manifest_sha256(root),baseline,name)
                (root/name).write_text(original)
            (root/'special_tokens_map.json').write_text('{}')
            self.assertNotEqual(tokenizer_manifest_sha256(root),baseline)

    def test_mixed_capture_uuid_is_rejected_even_with_identical_workload_identity(self):
        values=[manifest(rank) for rank in range(4)]
        aggregate_manifests(values)
        values[2]['identity']['capture_id']='22222222-2222-4222-8222-222222222222'
        with self.assertRaisesRegex(ValueError,'identity'):aggregate_manifests(values)

    def test_reuse_keeps_archive_capture_id_but_rechecks_current_tokenizer_digest(self):
        from src.weave.replay_trace import validate_reused_capture
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);values=[]
            for rank in range(4):
                value=manifest(rank);payload=b'private-placeholder'
                (root/f'rank{rank}.pt').write_bytes(payload)
                value['tensor_bytes']=len(payload);value['tensor_sha256']=hashlib.sha256(payload).hexdigest()
                (root/f'rank{rank}.json').write_text(json.dumps(value));values.append(value)
            summary=aggregate_manifests(values)
            (root/'capture_manifest.json').write_text(json.dumps(summary))
            current={k:v for k,v in identity().items() if k!='capture_id'}
            self.assertEqual(validate_reused_capture(root,current)['identity']['capture_id'],
                             '11111111-1111-4111-8111-111111111111')
            current['tokenizer_manifest_sha256']='2'*64
            with self.assertRaisesRegex(ValueError,'identity'):validate_reused_capture(root,current)

    def test_only_armed_full_b32_decode_forwards_count(self):
        life=CaptureLifecycle(CaptureConfig(),rank=0)
        life.begin('decode',32,32);life.end()
        self.assertEqual(life.snapshot()['eligible_forwards'],0)
        life.arm()
        for phase,rows,valid in [('prefill',32,32),('decode',32,31),('decode',31,31)]:
            self.assertIsNone(life.begin(phase,rows,valid));life.end()
        for _ in range(63):
            self.assertIsNone(life.begin('decode',32,32));life.end()
        self.assertEqual(life.begin('decode',32,32),0)
        for layer in range(43):life.record_layer(layer)
        life.end()
        self.assertEqual(life.snapshot()['sample_sequences'],[64,128])
        self.assertEqual(life.snapshot()['captured_layers'],[43,0])

    def test_missing_duplicate_and_failed_forwards_cannot_persist(self):
        life=CaptureLifecycle(CaptureConfig(),rank=0);life.arm()
        for _ in range(63):life.begin('decode',32,32);life.end()
        life.begin('decode',32,32);life.record_layer(0)
        with self.assertRaisesRegex(ValueError,'duplicate'):life.record_layer(0)
        with self.assertRaisesRegex(ValueError,'incomplete'):life.end()
        with self.assertRaisesRegex(ValueError,'failed'):life.seal()

    def test_complete_capture_seals_only_between_forwards(self):
        life=complete_lifecycle()
        life.begin('decode',32,32)
        self.assertTrue(life.snapshot()['complete'])
        with self.assertRaisesRegex(ValueError,'active'):life.seal()
        life.end();life.seal()
        self.assertTrue(life.snapshot()['complete'])
        with self.assertRaisesRegex(ValueError,'sealed'):life.begin('decode',32,32)
        with self.assertRaisesRegex(ValueError,'already'):life.arm()

    def test_persisted_manifest_requires_sealed_capture(self):
        value=manifest();value['coverage']['sealed']=False
        with self.assertRaisesRegex(ValueError,'coverage'):validate_rank_manifest(value,identity())

    def test_manifest_rejects_missing_layer_and_raw_public_fields(self):
        value=manifest();validate_rank_manifest(value,identity())
        value['coverage']['layer_bitmap'][1][42]=False
        with self.assertRaisesRegex(ValueError,'coverage'):validate_rank_manifest(value,identity())
        value=manifest();value['ids']=[1,2,3]
        with self.assertRaisesRegex(ValueError,'fields'):validate_rank_manifest(value,identity())

    def test_four_rank_identity_and_step_alignment_are_required(self):
        values=[manifest(rank) for rank in range(4)]
        summary=aggregate_manifests(values,identity())
        self.assertEqual(summary['rank_count'],4)
        self.assertEqual(summary['captured_layer_instances'],344)
        values[3]['identity']['dataset_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'identity'):aggregate_manifests(values,identity())

    def test_directory_reuse_checks_bytes_and_filenames_without_loading_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);values=[]
            for rank in range(4):
                value=manifest(rank);payload=b'private-tensor-file-placeholder'
                (root/f'rank{rank}.pt').write_bytes(payload)
                value['tensor_bytes']=len(payload)
                value['tensor_sha256']=hashlib.sha256(payload).hexdigest()
                (root/f'rank{rank}.json').write_text(json.dumps(value));values.append(value)
            summary=aggregate_manifests(values,identity())
            (root/'capture_manifest.json').write_text(json.dumps(summary))
            self.assertEqual(validate_capture_directory(root,identity()),summary)
            (root/'rank2.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash|size'):validate_capture_directory(root,identity())


if __name__=='__main__':unittest.main()
