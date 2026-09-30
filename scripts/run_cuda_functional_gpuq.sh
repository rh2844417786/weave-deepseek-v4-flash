#!/usr/bin/env bash
# Compile and check the actual CUDA core with synthetic small tensors on four
# allocated H100s. This job does not load the full model or measure performance.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?GPUQ must supply its allocation}"
: "${WORK_ROOT:?Set an absolute writable workspace on the server work disk}"
[[ "$CUDA_VISIBLE_DEVICES" == '4,5,6,7' ]] || { echo 'Expected GPU allocation 4,5,6,7' >&2; exit 2; }
[[ "$WORK_ROOT" == /* ]] || { echo 'WORK_ROOT must be absolute' >&2; exit 2; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_ref="${IMAGE:-sangfor-inference-sglang-nvidia-cuda13.0-ubuntu24.04-x86_64:v2.4.0.2}"
image_id="$(docker image inspect "$image_ref" --format '{{.Id}}')"
mkdir -p "$WORK_ROOT"
run_dir="$(mktemp -d "$WORK_ROOT/cuda-functional.XXXXXX")"
run_name="$(basename "$run_dir")"
printf '%s\n' "$image_id" > "$run_dir/image_id.txt"
printf '%s\n' "GPUQ_JOB_ID=${GPUQ_JOB_ID:-unset}" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" > "$run_dir/allocation.txt"
echo "CUDA build and functional log: $run_dir/build-and-test.log"
set +e
docker run --rm --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --network=host --shm-size=8g \
  -v "$repo_root:/workspace:ro" -v "$WORK_ROOT:/work" -w /workspace \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e GLOO_SOCKET_IFNAME=lo \
  -e TORCH_EXTENSIONS_DIR=/work/torch_extensions -e MAX_JOBS=2 \
  "$image_id" /usr/bin/python3 -m torch.distributed.run --standalone --nproc-per-node=4 \
    tests/gpu/check_functional.py --output-dir "/work/$run_name" \
    > "$run_dir/build-and-test.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$run_dir/exit_code.txt"
echo "Exit code: $code; evidence: $run_dir"
exit "$code"
