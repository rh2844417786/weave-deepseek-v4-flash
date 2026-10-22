#!/usr/bin/env python3
"""Preview or launch native/functional-Weave EP4 under a GPUQ allocation.

This is a bring-up launcher, not the 1200-second performance experiment runner.
No readiness, generation, or EP-correctness claim follows from process exit.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weave.runtime_contract import load_contract
from src.weave.container_lifecycle import run_container
from src.weave.gpu_allocation import allocated_gpus
from src.weave.execution_profile import add_profile_arguments,profile_options,profile_configuration
from src.weave.replay_trace import validate_capture_id
from src.weave.staged_decode import validate_decode_configuration, validate_prefill_configuration
CONTRACT = load_contract()
DEFAULT_IMAGE = CONTRACT['image']


def _concurrency_env(name: str) -> str:
    """Positive serving-parallelism value from the environment (default 1).

    The formal contract serves batch one. A concurrency experiment raises the
    server-side running-request cap and the decode-graph batch size together so
    the service can actually keep N requests in flight.
    """
    raw = os.environ.get(name, '1')
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer: {raw!r}') from exc
    if value < 1:
        raise ValueError(f'{name} must be positive: {value}')
    return str(value)


def docker_command(args, image: str, name: str, run_name='preview',allocation=None) -> list:
    allocation=allocation or allocated_gpus()
    docker = [
        'docker', 'run', '--rm', '--pull=never', '--name', name,
        '--gpus', allocation.docker_argument, '--network=host', '--shm-size=8g',
        '-v', f'{args.model_path}:/model:ro', '-v', f'{args.work_root}:/work',
        '-v', f'{ROOT}:/workspace:ro', '-w', '/workspace',
        '-e', f'CUDA_VISIBLE_DEVICES={allocation.container_visible}', '-e', 'CUDA_DEVICE_ORDER=PCI_BUS_ID',
        '-e', 'NCCL_IB_DISABLE=1', '-e', 'NCCL_SOCKET_IFNAME=lo',
        '-e', 'GLOO_SOCKET_IFNAME=lo', '-e', 'NCCL_CUMEM_HOST_ENABLE=0',
        '-e', 'TORCH_NCCL_ASYNC_ERROR_HANDLING=1', '-e', 'TORCH_NCCL_BLOCKING_WAIT=1',
        '-e', 'XDG_CACHE_HOME=/work/cache', '-e', 'HF_HOME=/work/cache/huggingface',
        '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
        '-e', 'HF_DATASETS_OFFLINE=1',
        '-e', f'WEAVE_IMAGE_REF={args.image}', '-e', f'WEAVE_IMAGE_ID={image}',
        '-e', f'WEAVE_HOST_HOSTNAME={os.uname().nodename}', '-e', f'WEAVE_HOST_ALLOCATION={allocation.csv}',
    ]
    if getattr(args, 'graph_c32', False):
        docker += ['-e', 'WEAVE_GRAPH_C32=1', '-e', f'WEAVE_GRAPH_EVIDENCE_PORT_BASE={args.graph_evidence_port_base}']
    if getattr(args, 'native_tail', 'off') != 'off':
        docker += ['-e', f'WEAVE_NATIVE_TAIL={args.native_tail}',
                   '-e', f'WEAVE_NATIVE_TAIL_DIRECTORY=/work/{args.native_tail_directory.relative_to(args.work_root)}',
                   '-e', f'WEAVE_NATIVE_TAIL_REPEATS={args.native_tail_repeats}',
                   '-e', f'WEAVE_NATIVE_TAIL_EVIDENCE_DIR=/work/{run_name}/native-tail',
                   '-e', 'WEAVE_MODEL_PATH=/model',
                   '-e', 'TORCH_EXTENSIONS_DIR=/work/cache/weave-native-tail-extensions']
    if getattr(args, 'native_budget', 'off') != 'off':
        mode = 'adaptive' if args.native_budget == 'heuristic' else args.native_budget
        docker += ['-e', f'WEAVE_NATIVE_BUDGET={mode}']
        if mode == 'cost':
            for key in ('native_cost_table_dir', 'native_cost_calibrate_dir'):
                directory = getattr(args, key, None)
                if directory is not None:
                    docker += ['-e', f'WEAVE_{key.upper()}=/work/{directory.relative_to(args.work_root)}']
            docker += ['-e', f'WEAVE_NATIVE_COST_REPEATS={args.native_cost_repeats}']
        else:
            docker += ['-e', f'WEAVE_NATIVE_BUDGET_TARGET_PERCENT={args.native_budget_target_percent}',
                       '-e', f'WEAVE_NATIVE_BUDGET_DIR=/work/{run_name}/native-budget']
    if getattr(args, 'graph_stage_profile', False):
        docker += ['-e', 'WEAVE_GRAPH_STAGE_PROFILE=1',
                   '-e', f'WEAVE_GRAPH_STAGE_STRIDE={args.graph_stage_stride}',
                   '-e', f'WEAVE_GRAPH_STAGE_CAPACITY={args.graph_stage_capacity}']
    if getattr(args, 'native_p0', False):
        docker += ['-e', 'WEAVE_NATIVE_P0=1',
                   '-e', f'WEAVE_NATIVE_P0_STRIDE={args.native_p0_stride}',
                   '-e', f'WEAVE_NATIVE_P0_CAPACITY={args.native_p0_capacity}',
                   '-e', f'WEAVE_NATIVE_P0_PRIVATE_DIR=/work/{run_name}/native-p0-private']
        if getattr(args, 'native_p0_mode', 'full') != 'full' or getattr(args, 'native_p0_no_readback', False):
            docker += ['-e', f'WEAVE_NATIVE_P0_MODE={args.native_p0_mode}',
                       '-e', f'WEAVE_NATIVE_P0_READBACK={int(not args.native_p0_no_readback)}']
    if getattr(args, 'fixed_prefix_dir', None):
        relative = args.fixed_prefix_dir.relative_to(args.work_root)
        docker += ['-e', 'WEAVE_FIXED_PREFIX=1', '-e', f'WEAVE_FIXED_PREFIX_DIR=/work/{relative}']
        if getattr(args, 'fixed_prefix_teacher', None):
            relative = args.fixed_prefix_teacher.relative_to(args.work_root)
            docker += ['-e', f'WEAVE_FIXED_PREFIX_TEACHER=/work/{relative}']
    functional = args.backend == 'weave_functional'
    if args.checkpoint_content_sha256:
        docker += ['-e', f'WEAVE_CHECKPOINT_CONTENT_SHA256={args.checkpoint_content_sha256}']
    if getattr(args,'stall_diagnostic',False):
        stall_candidate = 'native'
        if functional:
            stall_candidate = ('hopper_prefill' if getattr(args, 'prefill_implementation', 'persistent') == 'hopper_fp8'
                               else args.decode_implementation)
        docker += ['-e','WEAVE_STALL_DIAGNOSTIC=1','-e',f'WEAVE_STALL_PORT_BASE={args.stall_port_base}',
                   '-e',f'WEAVE_STALL_CANDIDATE={stall_candidate}']
    if args.request_timing:
        docker += ['-e','WEAVE_REQUEST_TIMING=1','-e','PYTHONPATH=/workspace/timing_hooks:/workspace/src']
    elif functional:
        docker += ['-e','PYTHONPATH=/workspace/src']
    if functional:
        docker += [
            '-e', 'SGLANG_EXTERNAL_MODEL_PACKAGE=weave.sglang_ext.models',
            '-e', 'WEAVE_MODEL_PATH=/model', '-e', 'WEAVE_CAPACITY=2048',
            '-e', f'WEAVE_FIXED_PLAN={args.fixed_plan}', '-e', 'TORCH_EXTENSIONS_DIR=/work/torch_extensions',
            '-e', f'WEAVE_KERNEL={args.kernel}', '-e', f'WEAVE_DECODE_GRAPH={int(args.decode_graph == "full")}',
            '-e', f'WEAVE_EVIDENCE_PORT_BASE={args.evidence_port_base}',
            '-e', f'WEAVE_HOT_CACHE={int(args.hot_cache)}',
            '-e', f'WEAVE_DECODE_IMPLEMENTATION={args.decode_implementation}',
            '-e', f'WEAVE_PREFILL_IMPLEMENTATION={getattr(args,"prefill_implementation","persistent")}',
        ]
        if args.replay_capture_dir:
            relative=args.replay_capture_dir.relative_to(args.work_root)
            docker += ['-e',f'WEAVE_REPLAY_CAPTURE_DIR=/work/{relative}',
                       '-e',f'WEAVE_REPLAY_DATASET_SHA256={args.replay_dataset_sha256}',
                       '-e',f'WEAVE_REPLAY_TOKENIZER_MANIFEST_SHA256={args.replay_tokenizer_manifest_sha256}',
                       '-e',f'WEAVE_REPLAY_CAPTURE_ID={args.replay_capture_id}']
        if args.profile:
            relative = args.profile.relative_to(args.work_root)
            docker += ['-e', f'WEAVE_PROFILE=/work/{relative}']
        if args.execution_profile:
            docker += ['-e','WEAVE_EXECUTION_PROFILE=1',
                       '-e',f'WEAVE_EXECUTION_PROFILE_STRIDE={args.execution_profile_stride}',
                       '-e',f'WEAVE_EXECUTION_PROFILE_CAPACITY={args.execution_profile_capacity}']
            if args.execution_profile_fixed_plan:
                docker += ['-e','WEAVE_FORWARD_PROFILE=1']
    command = docker + [image, args.runtime_python, '/workspace/scripts/check_runtime.py',
        '--stage', 'adapter' if functional else 'native', '--output-dir', f'/work/{run_name}',
        '--exec-command', args.runtime_python, '-m', 'sglang.launch_server',
        '--model-path', '/model', '--trust-remote-code',
        '--served-model-name', 'weave-dsv4-functional' if functional else 'weave-dsv4-native',
        '--host', '127.0.0.1', '--port', str(args.port),
        '--tp-size', '4', '--ep-size', '4',
        '--moe-runner-backend', 'marlin', '--moe-a2a-backend', 'none',
        '--context-length', '131072', '--mem-fraction-static', '0.80',
        '--chunked-prefill-size', '8192', '--max-running-requests', _concurrency_env('WEAVE_MAX_RUNNING_REQUESTS'),
        '--cuda-graph-backend-decode', args.decode_graph, '--cuda-graph-max-bs-decode', _concurrency_env('WEAVE_CUDA_GRAPH_MAX_BS_DECODE'),
        '--watchdog-timeout', str(args.native_tail_calibration_timeout if getattr(args,'native_tail','off')=='calibration' else args.native_cost_calibration_timeout if getattr(args, 'native_cost_calibrate_dir', None) else 1800), '--disable-custom-all-reduce',
        '--disable-shared-experts-fusion', '--reasoning-parser', 'deepseek-v4',
        '--decode-log-interval', '10', '--enable-metrics',
    ]
    if getattr(args, 'graph_c32', False):
        from src.weave.graph_observer import canonical_graph_config
        for flag in ('--cuda-graph-backend-decode', '--cuda-graph-max-bs-decode'):
            index = command.index(flag)
            del command[index:index + 2]
        command += ['--cuda-graph-config', json.dumps(canonical_graph_config(), separators=(',', ':')),
                    '--disable-cuda-graph-padding']
    if args.request_timing:
        command += ['--stream-interval','1','--incremental-streaming-output']
    if getattr(args, 'fixed_prefix_dir', None):
        from src.weave.fixed_prefix_budget import FIXED_PREFILL_TOKENS
        # DSV4 admission rounds each input to 256-token pages. The pinned
        # 6059-token cohort consumes 9472 tokens of this scheduler budget.
        command[command.index('--chunked-prefill-size') + 1] = str(FIXED_PREFILL_TOKENS)
        command += ['--max-prefill-tokens', str(FIXED_PREFILL_TOKENS),
                    '--enable-custom-logit-processor', '--disable-overlap-schedule']
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=Path('/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash'))
    parser.add_argument('--work-root', type=Path, default=Path(os.environ.get('WORK_ROOT', ROOT/CONTRACT['work_directory'])))
    parser.add_argument('--image', default=os.environ.get('IMAGE', DEFAULT_IMAGE))
    parser.add_argument('--runtime-python', default=os.environ.get('RUNTIME_PYTHON', CONTRACT['python_executable']))
    parser.add_argument('--port', type=int, default=30001)
    parser.add_argument('--backend', choices=['native', 'weave_functional'], default='native')
    parser.add_argument('--kernel', choices=['scalar', 'wmma', 'tiled'], default='scalar')
    parser.add_argument('--graph-c32', action='store_true')
    parser.add_argument('--native-budget', choices=('off', 'cost', 'heuristic', 'adaptive', 'full'), default='off',
                        help='cost is measured MoE argmin; heuristic/adaptive retain v1; off is original')
    parser.add_argument('--native-tail', choices=('off','static','calibration','measured'), default='off')
    parser.add_argument('--native-tail-directory',type=Path)
    parser.add_argument('--native-tail-repeats',type=int,default=7)
    parser.add_argument('--native-tail-calibration-timeout',type=int,default=43200)
    parser.add_argument('--native-budget-target-percent', type=int, default=75)
    cost_paths = parser.add_mutually_exclusive_group()
    cost_paths.add_argument('--native-cost-table-dir', type=Path,
                           help='complete measured rank profiles under WORK_ROOT')
    cost_paths.add_argument('--native-cost-calibrate-dir', type=Path,
                           help='fresh calibration output directory under WORK_ROOT')
    parser.add_argument('--native-cost-repeats', type=int, default=7)
    parser.add_argument('--native-cost-calibration-timeout', type=int, default=43200,
                        help='calibration-only worker watchdog in seconds; formal serving stays at 1800')
    parser.add_argument('--graph-stage-profile', action='store_true',
                        help='instrument full C32 graph replay; diagnostic timing only')
    parser.add_argument('--graph-stage-stride', type=int, default=128)
    parser.add_argument('--graph-stage-capacity', type=int, default=64)
    parser.add_argument('--native-p0', action='store_true',
                        help='native Marlin branch and launch diagnosis; instrumented timing only')
    parser.add_argument('--native-p0-stride', type=int, default=128)
    parser.add_argument('--native-p0-capacity', type=int, default=64)
    parser.add_argument('--native-p0-mode', choices=('full', 'events', 'buffers'), default='full',
                        help='retain stage events, grouping buffers, or both for the diagnostic')
    parser.add_argument('--native-p0-no-readback', action='store_true',
                        help='count actual B32 replays without event reads, grouping copies, or P0 trace')
    parser.add_argument('--fixed-prefix-dir', type=Path,
                        help='private fixed-prefix probe directory under WORK_ROOT')
    parser.add_argument('--fixed-prefix-teacher', type=Path,
                        help='private teacher continuation under WORK_ROOT; may be created after launch')
    parser.add_argument('--graph-evidence-port-base', type=int, default=30310)
    parser.add_argument('--decode-graph', choices=['disabled', 'full'], help='functional graph requires explicit full opt-in')
    parser.add_argument('--decode-implementation',choices=['persistent','staged_direct','hopper_fp8','hopper_persistent'],
                        default=os.environ.get('WEAVE_DECODE_IMPLEMENTATION','persistent'),
                        help='routed expert decode implementation; staged_direct/hopper_fp8 require eager tiled hot-cache')
    parser.add_argument('--prefill-implementation',choices=['persistent','hopper_fp8','hopper_persistent'],
                        default=os.environ.get('WEAVE_PREFILL_IMPLEMENTATION','persistent'))
    parser.add_argument('--fixed-plan', default='1,1,0', help='explicit c,K,completed-task steal budget per comm worker')
    parser.add_argument('--evidence-port-base', type=int, default=30110)
    parser.add_argument('--run-name', help='fresh native-ep-* directory name, for acceptance orchestration')
    parser.add_argument('--profile', type=Path, help='measured profile JSON under WORK_ROOT; otherwise functional fixed plan')
    parser.add_argument('--stall-diagnostic',action='store_true')
    parser.add_argument('--stall-port-base',type=int,default=30210)
    parser.add_argument('--request-timing', action='store_true', help='shared native token-ready timing for formal groups')
    parser.add_argument('--checkpoint-content-sha256', help='verified original checkpoint digest for measured tiled profiles')
    parser.add_argument('--hot-cache',action=argparse.BooleanOptionalAction,default=True,
                        help='bind immutable layer weights and cache workspace views before forward')
    parser.add_argument('--replay-capture-dir',type=Path,help='private real B32 trace directory under WORK_ROOT')
    parser.add_argument('--replay-dataset-sha256',help='existing custom9x200 workload SHA256 for capture identity')
    parser.add_argument('--replay-tokenizer-manifest-sha256',help='verified local tokenizer/config/encoding content manifest digest')
    parser.add_argument('--replay-capture-id',help='one canonical UUID4 shared by all ranks of this capture')
    parser.add_argument('--execute', action='store_true', help='launch Docker; default only prints the candidate command')
    add_profile_arguments(parser)
    parser.add_argument('--execution-profile-fixed-plan',action='store_true',
                        help='explicit fixed/eager diagnostic with full-forward sampling; no measured bank')
    args = parser.parse_args()
    if args.native_tail != 'off':
        if (args.backend!='native' or args.native_budget!='off' or not args.graph_c32 or args.decode_graph!='full'
            or not args.request_timing or args.native_p0 or args.graph_stage_profile or args.profile
            or args.execution_profile or args.execution_profile_fixed_plan or args.stall_diagnostic
            or args.replay_capture_dir or args.fixed_prefix_dir or args.fixed_prefix_teacher
            or args.decode_implementation!='persistent' or args.prefill_implementation!='persistent'):
            parser.error('native tail requires native full graph C32 without another method/diagnostic')
        if args.native_tail_directory is None:
            parser.error('native tail directory required; generate with scripts/run_native_tail.py --execute')
        if args.native_tail_repeats<2 or args.native_tail_calibration_timeout<1:
            parser.error('positive tail timeout and at least two measured repeats required')
    elif args.native_tail_directory or args.native_tail_repeats!=7 or args.native_tail_calibration_timeout!=43200:
        parser.error('tail configuration requires an explicit native-tail mode')
    if args.native_budget == 'cost':
        if not (args.native_cost_table_dir or args.native_cost_calibrate_dir):
            parser.error('native cost requires measured tables or calibration output; generate with '
                         'python3 scripts/run_native_cost.py --work-root WORK_ROOT --records PREPARED_MAIN --execute')
        if args.native_budget_target_percent != 75:
            parser.error('native cost has no heuristic target percent')
    elif args.native_cost_table_dir or args.native_cost_calibrate_dir or args.native_cost_repeats != 7:
        parser.error('native cost paths/repeats require --native-budget cost')
    if args.native_cost_repeats < 2:
        parser.error('native cost requires at least 2 repeats')
    if args.native_cost_calibration_timeout < 1:
        parser.error('native cost calibration timeout must be positive')
    if args.native_cost_calibration_timeout != 43200 and not args.native_cost_calibrate_dir:
        parser.error('native cost calibration timeout requires --native-cost-calibrate-dir')
    if not 1 <= args.native_budget_target_percent <= 100:
        parser.error('native budget target percent must be in 1..100')
    if args.native_budget != 'off':
        if (args.backend != 'native' or not args.graph_c32 or args.decode_graph != 'full'
                or not args.request_timing or args.native_p0 or args.graph_stage_profile
                or args.profile or args.execution_profile or args.execution_profile_fixed_plan
                or args.stall_diagnostic or args.replay_capture_dir or args.fixed_prefix_dir
                or args.fixed_prefix_teacher or args.decode_implementation != 'persistent'
                or args.prefill_implementation != 'persistent'):
            parser.error('native budget requires native full graph C32 request timing without diagnostics/profiles/custom implementations')
    elif args.native_budget_target_percent != 75:
        parser.error('native budget target percent requires --native-budget adaptive or full')
    if not 1 <= args.native_p0_stride <= 65536 or not 1 <= args.native_p0_capacity <= 256:
        parser.error('native P0 stride must be in 1..65536 and capacity in 1..256')
    if args.native_p0:
        if (args.backend != 'native' or not args.graph_c32 or args.decode_graph != 'full'
                or not args.request_timing or args.graph_stage_profile
                or args.profile or args.execution_profile or args.execution_profile_fixed_plan
                or args.stall_diagnostic or args.replay_capture_dir
                or args.decode_implementation != 'persistent' or args.prefill_implementation != 'persistent'):
            parser.error('native P0 requires native full graph C32 request timing without other profiles/custom implementations')
        if args.native_p0_mode != 'full' and not args.native_p0_no_readback:
            parser.error('native P0 events/buffers modes require --native-p0-no-readback')
    elif (args.native_p0_stride != 128 or args.native_p0_capacity != 64
          or args.native_p0_mode != 'full' or args.native_p0_no_readback):
        parser.error('native P0 stride/capacity/mode/readback settings require --native-p0')
    if args.fixed_prefix_teacher and not args.fixed_prefix_dir:
        parser.error('fixed-prefix teacher requires --fixed-prefix-dir')
    if args.fixed_prefix_dir:
        if (args.backend != 'native' or not args.graph_c32 or args.decode_graph != 'full'
                or not args.request_timing or args.graph_stage_profile or args.profile
                or args.execution_profile or args.execution_profile_fixed_plan
                or args.stall_diagnostic or args.replay_capture_dir
                or args.decode_implementation != 'persistent' or args.prefill_implementation != 'persistent'
                or (args.native_p0 and not args.native_p0_no_readback)):
            parser.error('fixed-prefix requires native full graph C32 request timing, no other profiles, and P0 readback disabled')
    if not 1 <= args.graph_stage_stride <= 65536 or not 1 <= args.graph_stage_capacity <= 256:
        parser.error('graph stage stride must be in 1..65536 and capacity in 1..256')
    if args.graph_stage_profile and args.decode_implementation == 'hopper_persistent':
        parser.error('paper runtime rejects legacy graph stage profiling')
    if args.graph_stage_profile and not args.graph_c32:
        parser.error('graph stage profiling requires --graph-c32')
    if not args.graph_stage_profile and (args.graph_stage_stride != 128 or args.graph_stage_capacity != 64):
        parser.error('graph stage stride/capacity require --graph-stage-profile')
    if args.graph_c32:
        if (args.decode_graph != 'full' or not args.request_timing
                or _concurrency_env('WEAVE_MAX_RUNNING_REQUESTS') != '32'
                or _concurrency_env('WEAVE_CUDA_GRAPH_MAX_BS_DECODE') != '32'
                or args.stall_diagnostic or (args.profile and args.decode_implementation != 'hopper_persistent') or args.execution_profile
                or args.execution_profile_fixed_plan or args.replay_capture_dir):
            parser.error('graph C32 requires full, request timing, both concurrency settings 32, no diagnostics/profiles/capture')
        if args.backend == 'weave_functional' and (args.decode_implementation not in ('hopper_fp8', 'hopper_persistent') or args.prefill_implementation != args.decode_implementation):
            parser.error('graph C32 Weave requires Hopper decode and prefill')
        ports = [args.port] + list(range(args.evidence_port_base, args.evidence_port_base + 4)) + list(range(args.graph_evidence_port_base, args.graph_evidence_port_base + 4))
        if not 1 <= args.graph_evidence_port_base <= 65532 or len(set(ports)) != 9:
            parser.error('graph evidence ports must be valid and disjoint')
    if args.stall_diagnostic:
        if (args.decode_graph!='disabled' or not args.request_timing or args.profile is not None
                or args.execution_profile or args.execution_profile_fixed_plan or args.replay_capture_dir):
            parser.error('stall diagnostic requires explicit eager, request timing and no profiles/capture')
        ports=[args.port]+list(range(args.evidence_port_base,args.evidence_port_base+4))+list(range(args.stall_port_base,args.stall_port_base+4))
        if not 1<=args.stall_port_base<=65532 or len(set(ports))!=9:
            parser.error('stall ports must be valid and distinct from serving/evidence ports')
    try:
        staged_fixed=tuple(int(value) for value in args.fixed_plan.split(','))
        if args.decode_implementation in ('staged_direct','hopper_fp8','hopper_persistent') and args.backend!='weave_functional':
            raise ValueError(f'{args.decode_implementation} requires weave_functional')
        validate_decode_configuration(decode_implementation=args.decode_implementation,kernel=args.kernel,
            hot_cache=args.hot_cache,allow_decode_graph=args.decode_graph=='full',fixed_plan=staged_fixed,
            profile_path=args.profile,execution_profile=True if args.execution_profile else None,
            replay_capture=bool(args.replay_capture_dir),forward_profile=args.execution_profile_fixed_plan, graph_c32=args.graph_c32 and args.backend=='weave_functional')
        if args.prefill_implementation in ('hopper_fp8','hopper_persistent') and args.backend != 'weave_functional':
            raise ValueError('hopper_fp8 prefill requires weave_functional')
        validate_prefill_configuration(prefill_implementation=args.prefill_implementation,
            decode_implementation=args.decode_implementation,kernel=args.kernel,
            hot_cache=args.hot_cache,allow_decode_graph=args.decode_graph=='full',fixed_plan=staged_fixed,
            profile_path=args.profile,execution_profile=True if args.execution_profile else None,
            replay_capture=bool(args.replay_capture_dir),forward_profile=args.execution_profile_fixed_plan, graph_c32=args.graph_c32 and args.backend=='weave_functional')
    except ValueError as error:parser.error(str(error))
    if args.replay_capture_dir and (args.backend!='weave_functional' or args.kernel!='tiled'
            or args.decode_graph=='full' or args.profile is not None or args.execution_profile
            or args.fixed_plan!='1,1,0' or not args.checkpoint_content_sha256
            or _concurrency_env('WEAVE_MAX_RUNNING_REQUESTS')!='32'
            or not re.fullmatch('[0-9a-f]{64}',args.replay_dataset_sha256 or '')
            or not re.fullmatch('[0-9a-f]{64}',args.replay_tokenizer_manifest_sha256 or '')):
        parser.error('replay capture requires tiled C32 fixed(1,1,0), eager, checkpoint/tokenizer/dataset identity and no profiling')
    if args.replay_capture_dir:
        try:validate_capture_id(args.replay_capture_id)
        except ValueError as error:parser.error(str(error))
    if (args.replay_dataset_sha256 or args.replay_tokenizer_manifest_sha256 or args.replay_capture_id) and not args.replay_capture_dir:
        parser.error('replay identity requires replay capture')
    try: profile_options(args)
    except ValueError as error: parser.error(str(error))
    if args.execution_profile and (args.backend!='weave_functional' or args.kernel!='tiled'):
        parser.error('execution-profile requires weave_functional and tiled kernel')
    if args.execution_profile_fixed_plan and (not args.execution_profile or args.profile is not None
            or args.decode_graph=='full' or not args.checkpoint_content_sha256):
        parser.error('fixed-plan profiling requires execution-profile, eager, checkpoint identity and no profile bank')
    if args.execution_profile and args.execute and args.profile is None and not args.execution_profile_fixed_plan:
        parser.error('execution-profile execution requires a measured profile bank and checkpoint identity')
    if args.checkpoint_content_sha256 and not re.fullmatch(r'[0-9a-f]{64}', args.checkpoint_content_sha256):
        parser.error('checkpoint-content-sha256 must be a full SHA256 digest')
    args.decode_graph = args.decode_graph or ('disabled' if args.backend == 'weave_functional' else 'full')
    if args.run_name and not re.fullmatch(r'native-ep-[A-Za-z0-9_.-]+', args.run_name):
        parser.error('run-name must be a safe native-ep-* basename')
    try:
        c, k, steal = [int(v) for v in args.fixed_plan.split(',')]
        if c < 0 or not 1 <= k <= 8 or steal < 0:
            raise ValueError
    except ValueError:
        parser.error('fixed-plan must contain nonnegative c/steal and K in 1..8')
    if not 1 <= args.evidence_port_base <= 65532 or args.port in range(args.evidence_port_base, args.evidence_port_base+4):
        parser.error('evidence port range must be valid and distinct from serving port')
    try: allocation=allocated_gpus()
    except ValueError as error: parser.error(str(error))
    if not args.model_path.is_absolute() or not args.work_root.is_absolute():
        parser.error('model-path and work-root must be absolute paths')
    args.work_root = args.work_root.resolve()
    if args.native_tail_directory is not None:
        path=args.native_tail_directory
        if not path.is_absolute() or '..' in path.parts:parser.error('native tail directory must be absolute')
        path=path.resolve()
        try:path.relative_to(args.work_root)
        except ValueError:parser.error('native tail directory must be under WORK_ROOT')
        args.native_tail_directory=path
        if args.execute and args.native_tail=='calibration' and any(path.glob('native-tail-rank*.json')):
            parser.error('fresh tail calibration directory required; preserve historical profiles')
        if args.execute and args.native_tail=='measured' and any(not (path/f'native-tail-rank{r}.json').is_file() for r in range(4)):
            parser.error('four measured tail profiles required; run scripts/run_native_tail.py --execute')
        if args.execute and not args.checkpoint_content_sha256:
            parser.error('native tail requires verified original checkpoint identity')
    for key in ('native_cost_table_dir', 'native_cost_calibrate_dir'):
        path = getattr(args, key)
        if path is None:
            continue
        if not path.is_absolute() or '..' in path.parts:
            parser.error('native cost directories must be absolute under WORK_ROOT')
        path = path.resolve()
        try:
            path.relative_to(args.work_root)
        except ValueError:
            parser.error('native cost directories must be under WORK_ROOT')
        setattr(args, key, path)
        if args.execute and key == 'native_cost_calibrate_dir' and any(path.glob('native-cost-rank*.json')):
            parser.error('native cost calibration requires a fresh output directory; preserve existing rank profiles')
        if args.execute and key == 'native_cost_table_dir':
            missing = [rank for rank in range(4) if not (path / f'native-cost-rank{rank}.json').is_file()]
            if missing:
                parser.error('native cost tables missing; generate with python3 scripts/run_native_cost.py '
                             '--work-root WORK_ROOT --records PREPARED_MAIN --execute')
    if args.execute and args.native_budget == 'cost' and not args.checkpoint_content_sha256:
        parser.error('native cost execution requires verified --checkpoint-content-sha256')
    for name in ('fixed_prefix_dir', 'fixed_prefix_teacher'):
        path = getattr(args, name)
        if path is None:
            continue
        if not path.is_absolute() or '..' in path.parts:
            parser.error('fixed-prefix paths must be absolute under WORK_ROOT')
        path = path.resolve()
        try:
            path.relative_to(args.work_root)
        except ValueError:
            parser.error('fixed-prefix paths must be under WORK_ROOT')
        setattr(args, name, path)
    if args.replay_capture_dir:
        if not args.replay_capture_dir.is_absolute() or '..' in args.replay_capture_dir.parts:
            parser.error('replay capture directory must be absolute under WORK_ROOT')
        args.replay_capture_dir=args.replay_capture_dir.resolve()
        try:args.replay_capture_dir.relative_to(args.work_root)
        except ValueError:parser.error('replay capture directory must be under WORK_ROOT')
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    if args.profile:
        if args.backend != 'weave_functional' or not args.profile.is_absolute() or '..' in args.profile.parts:
            parser.error('profile requires weave_functional and an absolute path under WORK_ROOT')
        args.profile = args.profile.resolve()
        try:
            args.profile.relative_to(args.work_root)
        except ValueError:
            parser.error('profile must be under WORK_ROOT')
    if args.profile and args.kernel == 'tiled' and not args.checkpoint_content_sha256:
        parser.error('measured tiled profile requires checkpoint-content-sha256 from the passed core gate')
    name = 'weave-native-ep-' + uuid.uuid4().hex[:12]
    if not args.execute:
        print(json.dumps({'status': 'preview_not_executed', 'docker_argv': docker_command(args, args.image, name,allocation=allocation)}, indent=2))
        return 0
    if not args.model_path.is_dir():
        parser.error(f'model directory missing: {args.model_path}')
    args.work_root.mkdir(parents=True, exist_ok=True)
    if not os.access(args.work_root, os.W_OK | os.X_OK):
        parser.error('WORK_ROOT must be writable and searchable')
    inspection = subprocess.run(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'],
                                capture_output=True, text=True, check=True, timeout=30)
    image_id = inspection.stdout.strip()
    if not image_id.startswith('sha256:'):
        raise ValueError('docker image inspect did not return a content-addressed image ID')
    run_dir = args.work_root / (args.run_name or 'native-ep-' + uuid.uuid4().hex)
    run_dir.mkdir()
    command = docker_command(args, image_id, name, run_dir.name,allocation=allocation)
    report = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'requested_image': args.image, 'resolved_image_id': image_id,
        'host_cuda_visible_devices': allocation.csv,
        'physical_gpus':list(allocation.ids),
        'gpuq_job_id': os.environ.get('GPUQ_JOB_ID'),
        'docker_argv': command, 'generation_status': 'not_verified',
        'backend': args.backend,
        'graph_mode': 'full' if args.graph_c32 else args.decode_graph,
        'graph_c32': args.graph_c32,
        'graph_stage_profile': args.graph_stage_profile,
        'native_p0': args.native_p0,
        'native_tail': args.native_tail,
        'native_tail_configuration': dict(directory=str(args.native_tail_directory),repeats=args.native_tail_repeats,method='native_marlin_peer_tail_v1') if args.native_tail!='off' else None,
        'native_budget': args.native_budget,
        'native_budget_configuration': ({'mode': args.native_budget,
                                         'target_percent': args.native_budget_target_percent,
                                         'ep_size': 4, 'min_percent': 25, 'buckets': 8}
                                        if args.native_budget in ('heuristic', 'adaptive', 'full') else None),
        'native_cost_configuration': ({'table_dir': str(args.native_cost_table_dir) if args.native_cost_table_dir else None,
                                       'calibrate_dir': str(args.native_cost_calibrate_dir) if args.native_cost_calibrate_dir else None,
                                       'repeats': args.native_cost_repeats,
                                       'policy_version': 'measured_moe_surface_v2',
                                       'mode': 'calibration' if args.native_cost_calibrate_dir else 'cost',
                                       'objective': 'measured_full_moe'} if args.native_budget == 'cost' else None),
        'native_p0_configuration': ({'stride': args.native_p0_stride, 'capacity': args.native_p0_capacity}
                                    if args.native_p0 else None),
        'native_p0_intervention': ({'mode': args.native_p0_mode,
                                    'readback': not args.native_p0_no_readback}
                                   if args.native_p0 else None),
        'fixed_prefix': args.fixed_prefix_dir is not None,
        'fixed_prefix_configuration': ({'private_dir': str(args.fixed_prefix_dir),
                                        'teacher_path': str(args.fixed_prefix_teacher)
                                        if args.fixed_prefix_teacher else None}
                                       if args.fixed_prefix_dir else None),
        'graph_stage_configuration': ({'stride': args.graph_stage_stride, 'capacity': args.graph_stage_capacity}
                                      if args.graph_stage_profile else None),
        'hot_cache':args.hot_cache,
        'decode_implementation':args.decode_implementation if args.backend=='weave_functional' else 'native',
        'prefill_implementation':args.prefill_implementation if args.backend=='weave_functional' else 'native',
        'replay_capture_enabled':args.replay_capture_dir is not None,
        'execution_profile_configuration':profile_configuration(args),
        'runtime_contract': CONTRACT, 'runtime_report': str(run_dir/'runtime.json'),
        'ep_correctness_status': 'not_verified', 'performance_status': 'not_run',
        'exit_code': None,
    }
    if args.stall_diagnostic or args.graph_stage_profile or args.native_p0 or args.fixed_prefix_dir:
        report.update(instrumented_diagnostic=True,used_for_formal_performance=False)
    if args.graph_stage_profile or args.native_p0 or args.fixed_prefix_dir:
        report.update(measurement_mode='instrumented_diagnostic', formal_gpqa=False,
                      numerical_policy='record_only')
    if args.native_tail != 'off':
        report.update(method='native_marlin_peer_tail_v1',numerical_policy='record_only',
                      measurement_mode='native_tail_'+args.native_tail,complete_weave_claim=False)
    if args.native_budget == 'cost':
        report.update(method='native_marlin_measured_cta_v2', numerical_policy='record_only',
                      measurement_mode=('native_cost_calibration' if args.native_cost_calibrate_dir
                                        else 'native_cost_performance'), complete_weave_claim=False)
    elif args.native_budget != 'off':
        report.update(method='native_marlin_dynamic_cta_v1', numerical_policy='record_only',
                      measurement_mode='native_dynamic_budget_performance', complete_weave_claim=False)
    report_path = run_dir / 'launch.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{args.backend} EP4 candidate; log: {run_dir / "server.log"}', flush=True)
    with (run_dir / 'server.log').open('w') as log:
        lifecycle = run_container(command, name, run_dir/'container_lifecycle.json', stdout=log,
                                  exit_code_path=run_dir/'exit_code.txt')
    code = lifecycle['exit_code']
    report['exit_code'] = code
    report['termination_signal'] = lifecycle['termination_signal']
    report['container_cleanup'] = lifecycle['cleanup_status']
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
