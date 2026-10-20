#!/usr/bin/env python3
"""Native-only 20-minute C32 P0; preserves native Marlin and full CUDA Graph."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_gpqa_c32 import main
if __name__=='__main__':
    sys.exit(main(['--native-p0','--decode-graph','full',*sys.argv[1:]]))
