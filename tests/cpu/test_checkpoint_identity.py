import json
from pathlib import Path
import tempfile
import unittest

from src.weave.checkpoint_identity import checkpoint_identity, checkpoint_stats


class CheckpointIdentityTest(unittest.TestCase):
    def test_same_index_cannot_hide_changed_weights_or_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'config.json').write_text('{"swiglu_limit":10}')
            (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'expert': 'one.safetensors'}}))
            shard = path/'one.safetensors'
            shard.write_bytes(b'original')
            a = checkpoint_identity(path)
            shard.write_bytes(b'different')
            b = checkpoint_identity(path)
            self.assertNotEqual(a['sha256'], b['sha256'])
            self.assertEqual(a['files_sha256']['model.safetensors.index.json'], b['files_sha256']['model.safetensors.index.json'])
            (path/'config.json').write_text('{"swiglu_limit":20}')
            self.assertNotEqual(b['sha256'], checkpoint_identity(path)['sha256'])

    def test_verified_stats_remain_bound_to_the_hashed_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'config.json').write_text('{}')
            (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'one.safetensors'}}))
            shard = path/'one.safetensors'
            shard.write_bytes(b'original')
            before = checkpoint_stats(path)
            identity = checkpoint_identity(path)
            shard.write_bytes(b'changed-after-hash')
            self.assertEqual(identity['verified_stats'], before)
            self.assertNotEqual(identity['verified_stats'], checkpoint_stats(path))

    def test_concurrent_checkpoint_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'config.json').write_text('{}')
            (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'one.safetensors'}}))
            (path/'one.safetensors').write_bytes(b'x')

            def modify(number, total, name):
                if number == total:
                    (path/'config.json').write_text('{"changed": true}')

            with self.assertRaisesRegex(ValueError, 'changed'):
                checkpoint_identity(path, progress=modify)


if __name__ == '__main__':
    unittest.main()
