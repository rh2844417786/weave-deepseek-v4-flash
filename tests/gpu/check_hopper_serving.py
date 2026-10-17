"""Four-H100 smoke of the actual eager Hopper runtime and NCCL slot assembly.

torchrun --nproc-per-node=4 tests/gpu/check_hopper_serving.py --output-dir /work/fresh

Uses synthetic original MXFP4/E8M0 storage, no model download or checkpoint load.
Test-only CPU snapshots are outside any performance measurement. Exact assertions
cover the deterministic assembly of the SAME computed slots, never FP8/BF16
approximation errors. Gloo coordinates local failures before entering NCCL.
"""
import argparse
from datetime import timedelta
import gc
import os
from pathlib import Path
import platform
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weave.hopper_evidence import atomic_report, public_resources

CASES = (('b1_spread', 1), ('b7_empty_rank', 7), ('b17_spread', 17),
         ('b32_spread', 32), ('b7_shrink', 7), ('b1_shrink', 1))
# Exact hand-derived sequential FP32 results. The first/third/fourth lose a
# small addend before cancellation; FP64 or reassociated sums differ.
ORDERED_FIXTURES = (
    ((16777216., 1., -16777216., 0., 0., 0.), 0.),
    ((16777216., -16777216., 1., 0., 0., 0.), 1.),
    ((1., 16777216., -16777216., 0., 0., 0.), 0.),
    ((-16777216., -1., 16777216., 0., 0., 0.), 0.),
    ((1., 2., 4., 8., 16., 32.), 63.),
)


def origin_rows(batch, rank):
    chunk = (batch + 3) // 4
    return list(range(rank * chunk, min((rank + 1) * chunk, batch)))


def route_rows(batch, variant):
    if variant == 1:
        return [[8 + slot for slot in range(6)] for _ in range(batch)]
    return [[(token * 7 + slot * 5 + variant) % 32 for slot in range(6)]
            for token in range(batch)]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class AgreedFailure(RuntimeError):
    pass


def private_exception(stage, exc):
    """Keep compiler/runtime details in the controller-owned private stderr log."""
    if isinstance(exc, AgreedFailure):
        return
    rank = os.environ.get('LOCAL_RANK', 'unknown')
    print(f'[hopper-serving rank={rank} stage={stage}]', file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)


def agreed(label, operation):
    result, failure = None, None
    try:
        result = operation()
    except Exception as exc:
        private_exception(label, exc)
        failure = type(exc).__name__
    states = [None] * 4
    dist.all_gather_object(states, (label, failure))
    if any(stage != label or error is not None for stage, error in states):
        # Exception messages may contain tensors. Reports retain only fixed stage
        # names; torchrun logs remain private under the parent controller.
        raise AgreedFailure(label)
    return result


def synthetic_weights(rank, device):
    shapes = [(8, 128, 64)] * 3 + [(8, 128, 4)] * 3
    raw = []
    for number, shape in enumerate(shapes):
        values = torch.arange(shape[0] * shape[1] * shape[2], device=device, dtype=torch.int32)
        values = ((values + rank * 17 + number * 13) % 256 if number < 3
                  else 119 + (values + rank + number) % 6)
        raw.append(values.to(torch.uint8).reshape(shape))
    return raw


def synthetic_inputs(batch, variant, device):
    generator = torch.Generator().manual_seed(73197 + variant)
    x = (torch.randn((batch, 128), generator=generator) * .125).bfloat16().to(device)
    ids = torch.tensor(route_rows(batch, variant), device=device, dtype=torch.int64)
    weights = [.1, .15, .2, .1, .25, .2] if variant % 2 else [.2, .1, .1, .25, .15, .2]
    routing = torch.tensor([weights] * batch, device=device, dtype=torch.float32)
    return x, ids, routing


def addresses(runtime):
    names = ('slot_outputs', 'group_counts', 'active_experts', 'active_count', 'expert_slots')
    return (tuple(getattr(runtime.ws, name).data_ptr() for name in names),
            runtime._output.data_ptr(),
            tuple((id(view), view.data_ptr()) for view in runtime._slots[1:]),
            tuple((id(view), view.data_ptr()) for view in runtime._outputs[1:]))


def ordered_reference(rank_slots, batch, rank):
    # Each (token, slot) belongs to exactly one expert rank. Check that property
    # separately, so adding three zeros introduces no reduction-order ambiguity.
    active = torch.zeros((batch * 6,), dtype=torch.int32)
    global_slots = torch.zeros((batch * 6, 128), dtype=torch.float32)
    for slots in rank_slots:
        active += (slots != 0).any(dim=1).to(torch.int32)
        global_slots.add_(slots)
    require(bool((active <= 1).all()), 'a routed slot has multiple owners')
    output = torch.zeros((batch, 128), dtype=torch.float32)
    for slot in range(6):
        output.add_(global_slots[slot::6])
    result = output.bfloat16()
    keep = set(origin_rows(batch, rank))
    for row in range(batch):
        if row not in keep:
            result[row].zero_()
    return result


def run_captured(runtime, inputs, bound, label):
    """Observe actual computed slots, then execute the original device SUM.

    A rank that raises before reaching all_reduce joins the same pre-NCCL
    agreement as its peers. Thus rank-local launch/validation failures do not
    strand other ranks inside NCCL. A device-group timeout still bounds faults
    occurring inside NCCL itself.
    """
    original = dist.all_reduce
    entered = False
    snapshots, calls = [], 0
    failure, output = None, None

    def observed(tensor, *args, **kwargs):
        nonlocal entered, calls
        entered = True
        calls += 1
        # A second attempted collective must join the common completion
        # agreement, not start a second pre-NCCL phase while peers finish.
        require(calls == 1, 'runtime issued more than one slot collective')

        def snapshot():
            require(kwargs.get('group') is runtime.device_group, 'wrong device group')
            require(kwargs.get('op') == dist.ReduceOp.SUM, 'slot collective must SUM')
            require(tensor is runtime._slots[inputs[0].shape[0]], 'wrong active slot prefix')
            value = tensor.detach().cpu().clone()
            require(bool(torch.isfinite(value).all()), 'nonfinite computed slots')
            return value

        local = agreed(label + ':pre_collective', snapshot)
        dist.all_gather_object(snapshots, local)
        return original(tensor, *args, **kwargs)

    snapshots.extend([None] * 4)
    dist.all_reduce = observed
    try:
        output = runtime.run(*inputs, bound, 7.)
        torch.cuda.synchronize()
    except Exception as exc:
        private_exception(label + ':runtime', exc)
        failure = type(exc).__name__
    finally:
        dist.all_reduce = original
    if not entered:
        failure = failure or 'MissingCollective'
        try:
            agreed(label + ':pre_collective', lambda: require(False, 'local runtime did not reach SUM'))
        except AgreedFailure:
            pass
    agreed(label + ':runtime_complete', lambda: require(failure is None and calls == 1,
                                                       'runtime/collective failed'))
    return output, snapshots


def ordered_finalizer_fixture(rank, device):
    from src.weave.hopper_finalize import finalize_slots
    slots = torch.empty((32 * 6, 128), dtype=torch.float32, device=device)
    output = torch.empty((32, 128), dtype=torch.bfloat16, device=device)
    before = (slots.data_ptr(), output.data_ptr())
    for batch in (1, 7, 17, 32):
        host = torch.empty((batch * 6, 128), dtype=torch.float32)
        expected = torch.zeros((batch, 128), dtype=torch.bfloat16)
        for row in range(batch):
            for column in range(128):
                values, answer = ORDERED_FIXTURES[(row + column) % len(ORDERED_FIXTURES)]
                for slot, value in enumerate(values):
                    host[row * 6 + slot, column] = value
                if row in origin_rows(batch, rank):
                    expected[row, column] = answer
        slots[:batch * 6].copy_(host)
        output.fill_(float('nan'))
        finalize_slots(slots[:batch * 6], output, batch, 128, 6, rank, 4)
        require(torch.equal(output[:batch].cpu(), expected), 'ordered finalizer fixture mismatch')
        require(before == (slots.data_ptr(), output.data_ptr()), 'finalizer buffers moved')
    return {'status': 'passed', 'case_count': 4, 'batches': [1, 7, 17, 32], 'exact': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    global torch, dist
    import torch
    import torch.distributed as dist
    from src.weave.hopper_candidate import source_sha256
    from src.weave.hopper_decode import HopperDecodeRuntime, assembly_source_sha256
    from src.weave.ops import kernel_source_sha256
    import triton

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', timeout=timedelta(minutes=15))
    report = {'schema_version': 1, 'numerical_policy': 'record_only', 'status': 'failed',
              'rank': rank, 'world_size': dist.get_world_size(), 'cases': [], 'case_count': 0,
              'stop_reason': 'runtime_failure'}
    device_group = None
    fresh = False

    def hashes():
        return {'candidate_source_sha256': source_sha256(),
                'bf16_source_sha256': kernel_source_sha256(),
                'assembly_source_sha256': assembly_source_sha256()}

    def save():
        atomic_report(args.output_dir / f'hopper-serving-rank{rank}.json', report)

    try:
        def inspect():
            require(dist.get_world_size() == 4 and dist.get_rank() == rank, 'single-node torchrun4 required')
            require(not any(args.output_dir.glob('hopper-serving-rank*.json')), 'fresh assembly output required')
            props = torch.cuda.get_device_properties(rank)
            require(torch.cuda.device_count() == 4 and 'H100' in props.name
                    and (props.major, props.minor) == (9, 0), 'four visible H100 GPUs required')
            return {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                    'python': platform.python_version(), 'triton': triton.__version__,
                    'device_name': props.name, 'capability': [props.major, props.minor],
                    'device_count': torch.cuda.device_count(), 'image_id': os.environ.get('WEAVE_IMAGE_ID')}

        report['runtime'] = agreed('inspect', inspect)
        fresh = True
        report.update(agreed('initial_sources', hashes))
        device_group = dist.new_group(ranks=list(range(4)), backend='nccl', timeout=timedelta(minutes=5))
        peer = SimpleNamespace(torch=torch, device=rank, rank=rank, ep=4, h=128, i=128, topk=6)
        runtime = agreed('create_runtime', lambda: HopperDecodeRuntime(peer, device_group, experts=32))
        agreed('loaded_sources', lambda: require(
            runtime.candidate_source_sha256 == report['candidate_source_sha256']
            and runtime.assembly_source_sha256 == report['assembly_source_sha256'],
            'loaded source identity differs from disk'))
        device = f'cuda:{rank}'
        raw = agreed('synthetic_weights', lambda: synthetic_weights(rank, device))
        bound = agreed('bind_weights', lambda: runtime.bind_weights(raw))
        before = addresses(runtime)
        first_slots = None
        flags = {'empty_local_rank': False, 'empty_origin_rank': False}
        for variant, (name, batch) in enumerate(CASES):
            # The final call repeats the first call after dropping all external
            # raw tensor references and exercising larger workspace prefixes.
            input_variant = 0 if variant == len(CASES) - 1 else variant
            inputs = agreed(name + ':inputs', lambda: synthetic_inputs(batch, input_variant, device))
            count_before = runtime.additional_slot_collectives
            output, slots = run_captured(runtime, inputs, bound, name)

            def check():
                expected = ordered_reference(slots, batch, rank)
                actual = output.cpu()
                require(bool(torch.isfinite(actual).all()), 'nonfinite assembled output')
                require(torch.equal(actual, expected), 'actual slot assembly differs')
                require(before == addresses(runtime), 'runtime buffer or cached view moved')
                require(output is runtime._outputs[batch], 'runtime did not return cached output')
                require(runtime.additional_slot_collectives - count_before == 1, 'incorrect collective counter')
                if first_slots is not None and variant == len(CASES) - 1:
                    require(all(torch.equal(a, b) for a, b in zip(first_slots, slots)),
                            'bound weight lifetime or shrinking inputs changed computed slots')
                rows = route_rows(batch, input_variant)
                local_slots = sum(expert // 8 == rank for row in rows for expert in row)
                # Explicitly check nonlocal slots are zero before the collective.
                remote = torch.tensor([expert // 8 != rank for row in rows for expert in row])
                require(not bool(torch.count_nonzero(slots[rank][remote])), 'nonlocal slots are nonzero')
                flags['empty_local_rank'] |= local_slots == 0
                flags['empty_origin_rank'] |= not origin_rows(batch, rank)
                return {'case': name, 'batch': batch, 'additional_slot_collectives': 1,
                        'assembly_exact': True, 'finite': True, 'stable_addresses': True,
                        'binding_alive': True, 'local_slots': local_slots,
                        'origin_rows': len(origin_rows(batch, rank))}

            report['cases'].append(agreed(name + ':check', check))
            if variant == 0:
                first_slots = slots
                del raw
                gc.collect()
        report['ordered_finalizer'] = agreed('ordered_finalizer', lambda: ordered_finalizer_fixture(rank, device))
        all_flags = [None] * 4
        dist.all_gather_object(all_flags, flags)
        agreed('coverage', lambda: require(
            any(item['empty_local_rank'] for item in all_flags)
            and any(item['empty_origin_rank'] for item in all_flags), 'empty-rank coverage missing'))
        report['source_hashes_after'] = agreed('final_sources', hashes)
        agreed('source_unchanged', lambda: require(
            all(report[key] == value for key, value in report['source_hashes_after'].items()),
            'source changed during smoke'))
        report.update(status='passed', stop_reason='none', case_count=len(report['cases']),
                      additional_slot_collectives=runtime.additional_slot_collectives,
                      stable_addresses=True, binding_lifetime=True,
                      empty_local_rank_exercised=True, empty_origin_rank_exercised=True,
                      resources=public_resources(runtime.resources))
        agreed('save', save)
        return 0
    except Exception as exc:
        private_exception('main', exc)
        report.update(status='failed', case_count=len(report['cases']),
                      stop_reason=str(exc) if isinstance(exc, AgreedFailure) else 'runtime_failure')
        if fresh:
            save()
        return 1
    finally:
        if device_group is not None:
            dist.destroy_process_group(device_group)
        dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
