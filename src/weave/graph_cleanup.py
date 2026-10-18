"""Release captured NCCL references before communicator destruction.

NVIDIA documents the dependency at:
https://docs.nvidia.com/dl-cuda-graph/troubleshooting/process-hang.html
The parent process bounds teardown; a Python thread cannot reliably interrupt
blocked CUDA/NCCL C++ calls in this process.
"""
from datetime import timedelta
import time

CLEANUP_PHASES = ('synchronize', 'reset_graphs', 'synchronize_after_reset',
                  'rank_barrier', 'destroy_nccl', 'destroy_gloo')


def cleanup_graphs(graphs, torch, dist, group, publish):
    """Keep tensors/bindings alive while resetting all owned (even partial) graphs.

    Explicit reset also releases graphs still aliased by a dict, closure or loop
    local. Stop on failure: destroying NCCL with a failed live graph is unsafe.
    The parent then terminates its owned container under the cleanup deadline.
    """
    state = {'status': 'running', 'phase': 'synchronize',
             'started_monotonic': time.monotonic(), 'graphs_owned': len(graphs),
             'graphs_reset': 0, 'completed_phases': []}

    def reset():
        for graph in graphs:
            graph.reset()
            state['graphs_reset'] += 1

    operations = (
        torch.cuda.synchronize, reset, torch.cuda.synchronize,
        lambda: dist.monitored_barrier(timeout=timedelta(seconds=30), wait_all_ranks=True),
        lambda: dist.destroy_process_group(group) if group is not None else None,
        lambda: dist.destroy_process_group(),
    )
    for phase, operation in zip(CLEANUP_PHASES, operations):
        state['phase'] = phase
        publish(dict(state, completed_phases=list(state['completed_phases'])))
        try:
            operation()
        except Exception as exc:
            state.update(status='failed', error_type=type(exc).__name__)
            publish(dict(state, completed_phases=list(state['completed_phases'])))
            raise
        state['completed_phases'].append(phase)
    state.update(status='complete', phase='complete', finished_monotonic=time.monotonic())
    publish(dict(state, completed_phases=list(state['completed_phases'])))
    return state
