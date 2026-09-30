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
OUTPUT_REL="${OUTPUT_REL:-results/deepseek_v4_preflight.json}"
# The project pins SGLang v0.5.16 at commit fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1.
# Its own /opt/sglang-v4 venv exists only in container c61793e2af9a's writable
# layer and is reproducible from no image. This image is built on that exact
# pinned release, so it supplies the same runtime.
IMAGE="${IMAGE:-sangfor-inference-sglang-nvidia-cuda13.0-ubuntu24.04-x86_64:v2.4.0.2}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-/usr/bin/python3}"
RUNTIME_NOTE="${RUNTIME_NOTE:-Runtime image built on lmsysorg/sglang:v0.5.16 at commit fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1, matching the pinned SGLang v0.5.16 / PyTorch 2.11.0 / Transformers 5.12.1. The project's own /opt/sglang-v4 venv exists only in container c61793e2af9a's writable layer and is reproducible from no image, so this image supplies the same pinned release; the interpreter here is /usr/bin/python3.}"

mkdir -p "${REPO_ROOT}/$(dirname "${OUTPUT_REL}")"

echo "gpuq job id:      ${GPUQ_JOB_ID:-unset}"
echo "gpuq project:     ${GPUQ_PROJECT:-unset}"
echo "allocation:       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "image:            ${IMAGE}"
echo "image id:         $(docker image inspect "${IMAGE}" --format '{{.Id}}')"
echo "runtime python:   ${RUNTIME_PYTHON}"
echo "model path:       ${MODEL_PATH}"
echo "output:           ${OUTPUT_REL}"

exec docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
  --network=host \
  --shm-size=8g \
  -v "${REPO_ROOT}:/workspace" \
  -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
  -w /workspace \
  "${IMAGE}" \
  "${RUNTIME_PYTHON}" scripts/preflight_deepseek_v4.py \
    --model-path "${MODEL_PATH}" \
    --output "${OUTPUT_REL}" \
    --runtime-image "${IMAGE}" \
    --runtime-provenance-note "${RUNTIME_NOTE}"
