"""Opt-in lazy instrumentation shared by native and Weave server launchers."""
import os

if os.environ.get('WEAVE_REQUEST_TIMING') == '1':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.weave.request_timing import install_timing_hook
    install_timing_hook()

if os.environ.get('WEAVE_STALL_DIAGNOSTIC') == '1':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.weave.stall_hooks import install_stall_hook
    install_stall_hook()
    if os.environ.get('WEAVE_REQUEST_TIMING') != '1':
        from src.weave.request_timing import install_timing_hook
        install_timing_hook()

if os.environ.get('WEAVE_GRAPH_C32') == '1':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.weave.graph_hooks import install_graph_hook
    install_graph_hook()

if os.environ.get('WEAVE_FIXED_PREFIX') == '1':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.weave.fixed_prefix_runtime import install_fixed_prefix_hook
    install_fixed_prefix_hook()
