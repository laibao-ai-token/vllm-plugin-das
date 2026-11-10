#!/usr/bin/env bash
# CPU source replay only; EP membership and Core routing kernels are doubled.
set -euo pipefail
eplb_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
eplb_core=$(realpath "${VLLM_TARGET_ROOT:-${eplb_repo}/../vllm}")
eplb_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${eplb_repo}/../adapter-das}")
eplb_output=${1:?Provide an output directory}
: "${VLLM_TEST_PYTHON:?Set existing Python 3.12 interpreter}"
mkdir -p "${eplb_output}"
eplb_output=$(realpath "${eplb_output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_ROOT="${eplb_core}"
export PYTHONPATH="${eplb_core}:${eplb_adapter}:${eplb_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${eplb_repo}/tests/vllm033/replay_v41_static_eplb.py" "${eplb_output}/report.json" 2>&1 | tee "${eplb_output}/replay.log"
