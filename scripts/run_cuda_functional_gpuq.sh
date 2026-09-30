#!/usr/bin/env bash
# Compile and check the actual CUDA core with synthetic small tensors on four
# allocated H100s. This job does not load the full model or measure performance.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?GPUQ must supply its allocation}"
[[ "$CUDA_VISIBLE_DEVICES" == '4,5,6,7' ]] || { echo 'Expected GPU allocation 4,5,6,7' >&2; exit 2; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$repo_root/work}"
[[ "$WORK_ROOT" == /* ]] || { echo 'WORK_ROOT must be absolute' >&2; exit 2; }
image_ref="${IMAGE:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$repo_root/configs/runtime.json")}"
image_id="$(docker image inspect "$image_ref" --format '{{.Id}}')"
mkdir -p "$WORK_ROOT"
[[ -w "$WORK_ROOT" && -x "$WORK_ROOT" ]] || { echo 'WORK_ROOT must be writable and searchable' >&2; exit 2; }
WORK_ROOT="$(cd "$WORK_ROOT" && pwd)"
run_dir="$(mktemp -d "$WORK_ROOT/cuda-functional.XXXXXX")"
run_name="$(basename "$run_dir")"
printf '%s\n' "$image_id" > "$run_dir/image_id.txt"
printf '%s\n' "GPUQ_JOB_ID=${GPUQ_JOB_ID:-unset}" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" > "$run_dir/allocation.txt"
echo "CUDA build and functional log: $run_dir/build-and-test.log"
set +e
docker run --rm --pull=never --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --network=host --shm-size=8g \
  -v "$repo_root:/workspace:ro" -v "$WORK_ROOT:/work" -w /workspace \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e GLOO_SOCKET_IFNAME=lo \
  -e "WEAVE_IMAGE_REF=$image_ref" -e "WEAVE_IMAGE_ID=$image_id" \
  -e "WEAVE_HOST_HOSTNAME=$(hostname)" -e "WEAVE_HOST_ALLOCATION=$CUDA_VISIBLE_DEVICES" \
  -e TORCH_EXTENSIONS_DIR=/work/torch_extensions -e MAX_JOBS=2 \
  "$image_id" "${RUNTIME_PYTHON:-/usr/bin/python3}" scripts/check_runtime.py \
    --stage core --output-dir "/work/$run_name" --run-core-tests \
    > "$run_dir/build-and-test.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$run_dir/exit_code.txt"
echo "Exit code: $code; evidence: $run_dir"
exit "$code"
