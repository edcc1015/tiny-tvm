#!/bin/bash
# build.sh - 编译 tiny_tvm 项目

set -euo pipefail

# ===== 配置变量 =====
ROOT_DIR=$(pwd)
BUILD_DIR="$ROOT_DIR/build"
BUILD_TYPE=${1:-Debug}  # 默认 Debug，可传 Release

# ===== 功能函数 =====
function usage() {
    echo "Usage: $0 [Debug|Release] [clean]"
    echo "  Default build type is Debug"
}

function clean_build() {
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    echo "Clean done."
}

# ===== 处理命令行参数 =====
if [[ "${1:-}" == "clean" ]]; then
    clean_build
    exit 0
fi

echo "Building tiny_tvm (type: $BUILD_TYPE)..."

# ===== 创建构建目录 =====
mkdir -p "$BUILD_DIR"

# ===== 配置阶段 =====
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# ===== 编译阶段 =====
cmake --build "$BUILD_DIR" -- -j$(nproc)

# ===== 运行测试（如果有启用） =====
if [ -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
    echo "Running tests..."
    cd "$BUILD_DIR"
    ctest --output-on-failure
    cd "$ROOT_DIR"
fi

echo "Build finished successfully!"