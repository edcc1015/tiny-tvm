#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN_FILE="${ROOT_DIR}/cmake/toolchains/arm-linux-gnueabihf.cmake"

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required but was not found in PATH." >&2
    exit 1
fi

if ! command -v arm-linux-gnueabihf-g++ >/dev/null 2>&1; then
    echo "arm-linux-gnueabihf-g++ is required for ARM builds." >&2
    exit 1
fi

if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
    echo "Missing toolchain file: ${TOOLCHAIN_FILE}" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build-arm" -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}"
cmake --build "${ROOT_DIR}/build-arm"
