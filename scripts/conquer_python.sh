#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="conquer-rlem"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
exec conda run --no-capture-output -n "${ENV_NAME}" python "$@"
