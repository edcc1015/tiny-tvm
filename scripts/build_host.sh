#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required but was not found in PATH." >&2
    exit 1
fi

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build"
cmake --build "${ROOT_DIR}/build"
ctest --test-dir "${ROOT_DIR}/build" --output-on-failure
