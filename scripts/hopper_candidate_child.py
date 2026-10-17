#!/usr/bin/env python3
"""Execute the two Torch oracle unit gates, then the four-rank GPU candidate."""
import argparse
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.weave.calibration import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    args,rest=parser.parse_known_args()
    import torch  # Required existing image dependency; a skipped oracle is not acceptance.
    suite=unittest.defaultTestLoader.loadTestsFromName('tests.cpu.test_hopper_candidate.TorchOracleTest')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    passed=result.wasSuccessful() and result.testsRun==2 and not result.skipped
    atomic_json(args.output_dir/'oracle-unit-check.json',{'status':'passed' if passed else 'failed',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'torch_version':str(torch.__version__)})
    if not passed:return 1
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        str(ROOT/'tests/gpu/check_hopper_candidate.py'),'--output-dir',str(args.output_dir)]+rest
    os.execv(sys.executable,command)


if __name__=='__main__':sys.exit(main())
