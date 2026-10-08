#!/usr/bin/env bash
# Compile and check the actual CUDA core with synthetic small tensors on four
# allocated H100s. This job does not load the full model or measure performance.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?GPUQ must supply its allocation}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_VISIBLE_DEVICES="$(python3 "$repo_root/scripts/check_gpu_allocation.py")"
export CUDA_VISIBLE_DEVICES
container_visible="$(python3 "$repo_root/scripts/check_gpu_allocation.py" --container-visible)"
WORK_ROOT="${WORK_ROOT:-$repo_root/work}"
[[ "$WORK_ROOT" == /* ]] || { echo 'WORK_ROOT must be absolute' >&2; exit 2; }
image_ref="${IMAGE:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$repo_root/configs/runtime.json")}"
image_id="$(docker image inspect "$image_ref" --format '{{.Id}}')"
extra_mounts=()
extra_tests=()
if [[ "${CHECKPOINT_TESTS:-0}" == 1 ]]; then
  model_path="${MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
  [[ "$model_path" == /* && -f "$model_path/model.safetensors.index.json" ]] || { echo 'Original checkpoint index required' >&2; exit 2; }
  extra_mounts=(-v "$model_path:/model:ro")
  extra_tests=(--run-checkpoint-tests --model-path /model --checkpoint-layers "${CHECKPOINT_LAYERS:-all}")
fi
mkdir -p "$WORK_ROOT"
[[ -w "$WORK_ROOT" && -x "$WORK_ROOT" ]] || { echo 'WORK_ROOT must be writable and searchable' >&2; exit 2; }
WORK_ROOT="$(cd "$WORK_ROOT" && pwd)"
if [[ -n "${CORE_RUN_NAME:-}" ]]; then
  [[ "$CORE_RUN_NAME" =~ ^cuda-functional\.[A-Za-z0-9_-]+$ ]] || { echo 'Unsafe CORE_RUN_NAME' >&2; exit 2; }
  run_dir="$WORK_ROOT/$CORE_RUN_NAME"
  mkdir "$run_dir"
else
  run_dir="$(mktemp -d "$WORK_ROOT/cuda-functional.XXXXXX")"
fi
run_name="$(basename "$run_dir")"
container_name="weave-cuda-functional-${run_name#cuda-functional.}"
printf '%s\n' "$image_id" > "$run_dir/image_id.txt"
printf '%s\n' "GPUQ_JOB_ID=${GPUQ_JOB_ID:-unset}" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" > "$run_dir/allocation.txt"
echo "CUDA build and functional log: $run_dir/build-and-test.log"
set +e
python3 "$repo_root/scripts/manage_container.py" --name "$container_name" \
  --evidence "$run_dir/container_lifecycle.json" --log "$run_dir/build-and-test.log" \
  --exit-code-file "$run_dir/exit_code.txt" -- \
  docker run --rm --name "$container_name" --pull=never --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --network=host --shm-size=8g \
  -v "$repo_root:/workspace:ro" -v "$WORK_ROOT:/work" ${extra_mounts[@]+"${extra_mounts[@]}"} -w /workspace \
  -e "CUDA_VISIBLE_DEVICES=$container_visible" -e GLOO_SOCKET_IFNAME=lo \
  -e "WEAVE_IMAGE_REF=$image_ref" -e "WEAVE_IMAGE_ID=$image_id" \
  -e "WEAVE_HOST_HOSTNAME=$(hostname)" -e "WEAVE_HOST_ALLOCATION=$CUDA_VISIBLE_DEVICES" \
  -e TORCH_EXTENSIONS_DIR=/work/torch_extensions -e MAX_JOBS=2 \
  "$image_id" "${RUNTIME_PYTHON:-/usr/bin/python3}" scripts/check_runtime.py \
    --stage core --output-dir "/work/$run_name" --run-core-tests ${extra_tests[@]+"${extra_tests[@]}"}
code=$?
set -e
printf '%s\n' "$code" > "$run_dir/exit_code.txt"
echo "Exit code: $code; evidence: $run_dir"
exit "$code"
