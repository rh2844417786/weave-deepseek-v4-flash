import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from src.weave.calibration import (CalibrationJournal,TimingStore,atomic_json,
                                  run_diagnostic_case)
from tests.cpu.test_calibration import IDENTITY,point


class CalibrationPermissionsTest(unittest.TestCase):
    def test_rank_artifacts_remain_readable_when_writer_umask_is_restrictive(self):
        # A root container publishes files later read by a different host UID.
        # The group-class read bit also controls the Linux named-user ACL mask.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            previous=os.umask(0o077)
            try:
                store=TimingStore(root/'timings-rank0.json',IDENTITY)
                journal=CalibrationJournal(root,IDENTITY,'permission-case',100.,1200.)
                case={'tokens':1024,'c':131,'K':8,'mode':'mixed'}
                journal.begin(case,0,elapsed_seconds=0.,completed_cases=0)
                result=run_diagnostic_case(store,case,
                    lambda value:point(1024,131,8,'mixed'),lambda:False)
                atomic_json(root/'diagnostic-rank0.json',result)
            finally:
                os.umask(previous)
            for name in ('timings-rank0.json','progress-rank0.json','diagnostic-rank0.json'):
                with self.subTest(name=name):
                    mode=stat.S_IMODE((root/name).stat().st_mode)
                    self.assertTrue(mode&stat.S_IRGRP,'root-written JSON masks the host-reader ACL')
                    self.assertFalse(mode&(stat.S_IXUSR|stat.S_IXGRP|stat.S_IXOTH|stat.S_IWOTH))
                    self.assertIsInstance(json.loads((root/name).read_text()),dict)

    def test_atomic_replacement_restores_reader_permissions_without_changing_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'progress-rank0.json'
            value={'rank':0,'status':'measuring','invocation_id':'current'}
            atomic_json(path,value)
            original=path.read_bytes()
            path.chmod(0o600)
            previous=os.umask(0o077)
            try: atomic_json(path,value)
            finally: os.umask(previous)
            self.assertTrue(path.stat().st_mode&stat.S_IRGRP)
            self.assertEqual(path.read_bytes(),original)
