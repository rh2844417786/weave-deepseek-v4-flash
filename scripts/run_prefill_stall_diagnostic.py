#!/usr/bin/env python3
"""Opt-in instrumented three-group C32 prefill/output-stall diagnostic."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_c32_comparison import main as comparison_main

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    # Required arguments appended last so this convenience entrypoint cannot run a formal suite.
    return comparison_main(argv+['--comparison-suite','hopper','--stall-diagnostic'])
if __name__=='__main__':sys.exit(main())
