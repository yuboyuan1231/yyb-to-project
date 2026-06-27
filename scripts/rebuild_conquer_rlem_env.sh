#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="conquer-rlem"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.9.19 pip=23.3.1 -y
fi

PYTHON_BIN="$(conda run -n "${ENV_NAME}" which python)"
"${PYTHON_BIN}" -m pip install --timeout 3600 --retries 20 \
  torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
"${PYTHON_BIN}" -m pip install --timeout 600 --retries 20 \
  -r "${PROJECT_ROOT}/requirements-conquer-rlem.txt"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/check_migrated_runtime.py" \
  --project_root "${PROJECT_ROOT}"
