"""Actual CPU process trees exercise PID-only cancellation and spawn interruption."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.run_model_acceptance import OwnedChild


class AcceptanceChildCleanupTest(unittest.TestCase):
    def make_tree(self, path, *, partial_evidence=False):
        manager = path/'manager.py'
        manager.write_text('import json,signal,sys,time\nfrom pathlib import Path\n'
            'report=Path(sys.argv[1]);ready=Path(sys.argv[2])\n'
            'def stop(sig,frame):\n'
            '    time.sleep(.15)\n'
            + ('    report.write_text("{")\n    time.sleep(.2)\n' if partial_evidence else '')
            + '    report.write_text(json.dumps({"cleanup_status":"absent"}))\n'
            '    sys.exit(0)\n'
            'signal.signal(signal.SIGTERM,stop)\n'
            'ready.write_text("ready")\n'
            'while True: time.sleep(.1)\n')
        leader = path/'leader.py'
        leader.write_text('import subprocess,sys,time\n'
            'subprocess.Popen(sys.argv[1:])\n'
            'time.sleep(3600)\n')
        ready, evidence = path/'ready', path/'lifecycle.json'
        command = [sys.executable, str(leader), sys.executable, str(manager), str(evidence), str(ready)]
        return command, ready, evidence

    def wait_ready(self, ready):
        deadline = time.monotonic()+3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(ready.exists(), 'CPU manager did not start')

    def test_signal_to_acceptance_pid_cleans_descendant_after_leader_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            command, ready, evidence = self.make_tree(Path(tmp))
            child = OwnedChild()
            child.spawn(command, evidence)
            try:
                self.wait_ready(ready)
                with self.assertRaises(InterruptedError):
                    child.signal(signal.SIGTERM, None)
            finally:
                child.stop()
            self.assertEqual(json.loads(evidence.read_text())['cleanup_status'], 'absent')
            self.assertIsNone(child.process)

    def test_partial_lifecycle_json_is_not_treated_as_finished_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            command, ready, evidence = self.make_tree(Path(tmp), partial_evidence=True)
            child = OwnedChild()
            child.spawn(command, evidence)
            try:
                self.wait_ready(ready)
                child.stop()
            finally:
                # Let the short-lived descendant finish even on a failing assertion.
                if child.process is not None:
                    time.sleep(.4)
                    child.stop()
            self.assertEqual(json.loads(evidence.read_text())['cleanup_status'], 'absent')
            self.assertIsNone(child.process)

    def test_signal_during_spawn_is_deferred_until_child_is_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            command, ready, evidence = self.make_tree(Path(tmp))
            child = OwnedChild()
            original = subprocess.Popen

            def interrupted_spawn(*args, **kwargs):
                process = original(*args, **kwargs)
                self.wait_ready(ready)
                child.signal(signal.SIGTERM, None)
                return process

            try:
                with patch('scripts.run_model_acceptance.subprocess.Popen', interrupted_spawn):
                    with self.assertRaises(InterruptedError):
                        child.spawn(command, evidence)
                self.assertIsNotNone(child.process)
            finally:
                child.stop()
            self.assertEqual(json.loads(evidence.read_text())['cleanup_status'], 'absent')


if __name__ == '__main__':
    unittest.main()
