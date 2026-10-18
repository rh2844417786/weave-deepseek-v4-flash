#!/usr/bin/env python3
"""Run the prefill suite through the existing owned C32 experiment controller."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_c32_comparison import main as comparison_main

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    # Append last so argparse's final value remains prefill for every spelling.
    return comparison_main(argv+['--comparison-suite','prefill'])

if __name__=='__main__':sys.exit(main())
