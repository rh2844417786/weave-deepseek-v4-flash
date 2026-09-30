#!/usr/bin/env bash
# GPUQ-managed launcher for the DeepSeek-V4-Flash read-only preflight.
#
# Two host quirks are load-bearing here:
#   * Docker CLI 29 parses an unquoted comma-separated device list as both a
#     count and DeviceIDs: "device=4,5,6,7" becomes {Count: 7, DeviceIDs: ["4"]}
#     and the daemon rejects the run with "cannot set both Count and DeviceIDs".
#     Keep the inner quotes as part of the argument. Same convention as
#     moe-flex/scripts/server/run_container.sh.
#   * This host has no docker0 bridge, so bridge networking cannot be set up.
#     --network=host avoids it, as moe-flex does.
# Never widen --gpus to "all". This image bakes NVIDIA_VISIBLE_DEVICES=all, so a
# run with no --gpus sees all eight cards; --gpus "device=..." is what scopes the
# container to the allocation.
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?GPUQ must export CUDA_VISIBLE_DEVICES}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/work}"
[[ "$WORK_ROOT" == /* ]] || { echo 'WORK_ROOT must be absolute' >&2; exit 2; }
mkdir -p "$WORK_ROOT"
[[ -w "$WORK_ROOT" && -x "$WORK_ROOT" ]] || { echo 'WORK_ROOT must be writable and searchable' >&2; exit 2; }
WORK_ROOT="$(cd "$WORK_ROOT" && pwd)"
RUN_DIR="$(mktemp -d "$WORK_ROOT/preflight.XXXXXX")"
RUN_NAME="$(basename "$RUN_DIR")"
CONTAINER_NAME="weave-preflight-${RUN_NAME#preflight.}"
OUTPUT_REL="${OUTPUT_REL:-$RUN_NAME/deepseek_v4_preflight.json}"
[[ "$OUTPUT_REL" != /* && "$OUTPUT_REL" != *'..'* ]] || { echo 'OUTPUT_REL must stay under WORK_ROOT' >&2; exit 2; }
IMAGE="${IMAGE:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$REPO_ROOT/configs/runtime.json")}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-/usr/bin/python3}"
RUNTIME_NOTE="${RUNTIME_NOTE:-Current-host image selection: $IMAGE. Package versions in this report are observed at runtime. Historical 0.5.16 preflight evidence belongs to its own recorded image and must not establish availability on this host.}"
IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"

mkdir -p "${WORK_ROOT}/$(dirname "${OUTPUT_REL}")"

echo "gpuq job id:      ${GPUQ_JOB_ID:-unset}"
echo "gpuq project:     ${GPUQ_PROJECT:-unset}"
echo "allocation:       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "image:            ${IMAGE}"
echo "image id:         ${IMAGE_ID}"
echo "runtime python:   ${RUNTIME_PYTHON}"
echo "model path:       ${MODEL_PATH}"
echo "output:           ${WORK_ROOT}/${OUTPUT_REL}"

set +e
python3 "$REPO_ROOT/scripts/manage_container.py" --name "$CONTAINER_NAME" \
  --evidence "$RUN_DIR/container_lifecycle.json" --log "$RUN_DIR/preflight.log" \
  --exit-code-file "$RUN_DIR/exit_code.txt" -- \
docker run --rm --name "$CONTAINER_NAME" --pull=never \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
  --network=host \
  --shm-size=8g \
  -v "${REPO_ROOT}:/workspace:ro" -v "${WORK_ROOT}:/work" \
  -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
  -w /workspace \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e "WEAVE_IMAGE_REF=$IMAGE" -e "WEAVE_IMAGE_ID=$IMAGE_ID" \
  -e "WEAVE_HOST_HOSTNAME=$(hostname)" -e "WEAVE_HOST_ALLOCATION=$CUDA_VISIBLE_DEVICES" \
  "${IMAGE_ID}" \
  "${RUNTIME_PYTHON}" scripts/check_runtime.py --stage core --output-dir "/work/$RUN_NAME" \
    --exec-command "${RUNTIME_PYTHON}" scripts/preflight_deepseek_v4.py \
    --model-path "${MODEL_PATH}" \
    --output "/work/${OUTPUT_REL}" \
    --runtime-image "${IMAGE}" \
    --runtime-image-id "${IMAGE_ID}" \
    --runtime-provenance-note "${RUNTIME_NOTE}"
code=$?
set -e
printf '%s\n' "$code" > "$RUN_DIR/exit_code.txt"
echo "Exit code: $code; evidence: $RUN_DIR"
exit "$code"
