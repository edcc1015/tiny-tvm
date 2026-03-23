#!/usr/bin/env bash
# =============================================================================
# Phase 5 测试脚本：ARM / QEMU 部署
# =============================================================================
# 检测要点：
#   5-1: ARM 交叉编译（build-arm/ 产出 ARM 可执行文件和 .so）
#   5-2: qemu-arm 运行 run_model
#   5-3: Host / ARM 输出对比（误差 < 1e-5）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
BUILD_ARM_DIR="$ROOT_DIR/build-arm"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0

pass() { PASS_COUNT=$((PASS_COUNT + 1)); TOTAL_COUNT=$((TOTAL_COUNT + 1)); echo "[PASS] $1"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); TOTAL_COUNT=$((TOTAL_COUNT + 1)); echo "[FAIL] $1"; [ -n "${2:-}" ] && echo "       Reason: $2"; }
section() { echo ""; echo "======================================================================"; echo "  $1"; echo "======================================================================"; }

# =====================================================================
# 前置检查：ARM 工具链和 QEMU
# =====================================================================
section "前置：检查 ARM 工具链和 QEMU"

ARM_GCC=""
if command -v arm-linux-gnueabihf-g++ > /dev/null 2>&1; then
    ARM_GCC="arm-linux-gnueabihf-g++"
    pass "前置a: arm-linux-gnueabihf-g++ 可用"
else
    fail "前置a: arm-linux-gnueabihf-g++ 不可用" "sudo apt install g++-arm-linux-gnueabihf"
fi

QEMU_ARM=""
if command -v qemu-arm > /dev/null 2>&1; then
    QEMU_ARM="qemu-arm"
    pass "前置b: qemu-arm 可用"
else
    fail "前置b: qemu-arm 不可用" "sudo apt install qemu-user"
fi

# =====================================================================
# 前置：确保 Host 版本已编译
# =====================================================================
section "前置：编译 Host 版本"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug > /dev/null 2>&1 || true
cmake --build "$BUILD_DIR" --parallel > /dev/null 2>&1 || echo "WARNING: Host 编译有错误"

TTVMC="$BUILD_DIR/ttvmc"

# =====================================================================
# 任务 5-1：ARM 交叉编译
# =====================================================================
section "任务 5-1：ARM 交叉编译"

# 检查 toolchain 文件存在
TOOLCHAIN="$ROOT_DIR/cmake/toolchains/arm-linux-gnueabihf.cmake"
if [ -f "$TOOLCHAIN" ]; then
    pass "5-1a: ARM toolchain 文件存在"
else
    fail "5-1a: ARM toolchain 文件不存在" "需要 cmake/toolchains/arm-linux-gnueabihf.cmake"
fi

# 检查 build_arm.sh 存在
BUILD_ARM_SCRIPT="$ROOT_DIR/scripts/build_arm.sh"
if [ -f "$BUILD_ARM_SCRIPT" ]; then
    pass "5-1b: scripts/build_arm.sh 存在"
else
    fail "5-1b: scripts/build_arm.sh 不存在"
fi

# 尝试执行 ARM 编译
if [ -n "$ARM_GCC" ] && [ -f "$TOOLCHAIN" ]; then
    cmake -S "$ROOT_DIR" -B "$BUILD_ARM_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DTINY_TVM_BUILD_TESTS=OFF > /dev/null 2>&1 || true
    cmake --build "$BUILD_ARM_DIR" --parallel > /dev/null 2>&1 || true

    # 检查 ARM 可执行文件产出
    ARM_RUN_MODEL="$BUILD_ARM_DIR/run_model"
    if [ -f "$ARM_RUN_MODEL" ]; then
        # 验证确实是 ARM 二进制
        FILE_TYPE=$(file "$ARM_RUN_MODEL" 2>/dev/null)
        if echo "$FILE_TYPE" | grep -qi "ARM"; then
            pass "5-1c: build-arm/run_model 是 ARM 二进制"
        else
            fail "5-1c: build-arm/run_model 不是 ARM 二进制" "$FILE_TYPE"
        fi
    else
        fail "5-1c: build-arm/run_model 不存在"
    fi

    # 检查 ARM 版 ttvmc（可选）
    ARM_TTVMC="$BUILD_ARM_DIR/ttvmc"
    if [ -f "$ARM_TTVMC" ]; then
        pass "5-1d: build-arm/ttvmc 存在（可选，已产出）"
    else
        pass "5-1d: build-arm/ttvmc 不存在（可选，非必须）"
    fi
else
    fail "5-1c: 无法进行 ARM 编译（缺少工具链或 toolchain 文件）"
    fail "5-1d: 跳过"
fi

# 先在 Host 上编译模型，生成 deploy.c，再用 ARM 编译器编译 deploy.c
MLP_JSON="$ROOT_DIR/examples/json/mlp.json"
OUT_MLP="$TMP_DIR/out_mlp"
mkdir -p "$OUT_MLP"

if [ -f "$TTVMC" ]; then
    "$TTVMC" compile "$MLP_JSON" -o "$OUT_MLP" > /dev/null 2>&1 || true
fi

if [ -n "$ARM_GCC" ] && [ -f "$OUT_MLP/deploy.c" ]; then
    # 用 ARM 编译器编译 deploy.c -> libdeploy.so (ARM)
    ARM_DEPLOY_SO="$OUT_MLP/libdeploy_arm.so"
    if $ARM_GCC -shared -fPIC "$OUT_MLP/deploy.c" -O2 -o "$ARM_DEPLOY_SO" 2>/dev/null; then
        FILE_TYPE_SO=$(file "$ARM_DEPLOY_SO" 2>/dev/null)
        if echo "$FILE_TYPE_SO" | grep -qi "ARM"; then
            pass "5-1e: ARM 版 libdeploy.so 编译成功"
        else
            fail "5-1e: ARM 版 libdeploy.so 不是 ARM 二进制"
        fi
    else
        fail "5-1e: ARM 版 libdeploy.so 编译失败"
    fi
else
    fail "5-1e: 无法编译 ARM 版 libdeploy.so（缺少 ARM 编译器或 deploy.c）"
fi

# =====================================================================
# 任务 5-2：qemu-arm 运行
# =====================================================================
section "任务 5-2：qemu-arm 运行"

# 检查 run_qemu.sh 存在
RUN_QEMU_SCRIPT="$ROOT_DIR/scripts/run_qemu.sh"
if [ -f "$RUN_QEMU_SCRIPT" ]; then
    # 检查不再是纯占位
    if grep -q "not.*implement\|TODO\|placeholder" "$RUN_QEMU_SCRIPT" 2>/dev/null && \
       ! grep -q "qemu-arm" "$RUN_QEMU_SCRIPT" 2>/dev/null; then
        fail "5-2a: run_qemu.sh 仍是占位实现"
    else
        pass "5-2a: run_qemu.sh 包含 qemu-arm 调用逻辑"
    fi
else
    fail "5-2a: scripts/run_qemu.sh 不存在"
fi

ARM_RUN_MODEL="$BUILD_ARM_DIR/run_model"
ARM_DEPLOY_SO="$OUT_MLP/libdeploy_arm.so"

if [ -n "$QEMU_ARM" ] && [ -f "$ARM_RUN_MODEL" ] && [ -f "$ARM_DEPLOY_SO" ]; then
    # 创建输入
    python3 -c "import struct,sys; sys.stdout.buffer.write(struct.pack('4f',1,2,3,4))" > "$TMP_DIR/input.bin"

    # 拷贝 ARM libdeploy.so 到和 run_model 同目录（或使用 LD_LIBRARY_PATH）
    cp "$ARM_DEPLOY_SO" "$OUT_MLP/libdeploy.so.arm" || true

    # 运行 qemu-arm
    QEMU_OUTPUT=$(LD_LIBRARY_PATH="$OUT_MLP" \
        $QEMU_ARM -L /usr/arm-linux-gnueabihf \
        "$ARM_RUN_MODEL" "$OUT_MLP" "$TMP_DIR/input.bin" "$TMP_DIR/output_arm.bin" 2>&1 || true)

    if [ -f "$TMP_DIR/output_arm.bin" ]; then
        ARM_OUT_SIZE=$(stat -c%s "$TMP_DIR/output_arm.bin" 2>/dev/null || echo "0")
        if [ "$ARM_OUT_SIZE" -gt 0 ]; then
            pass "5-2b: qemu-arm 运行 run_model 成功（输出 $ARM_OUT_SIZE bytes）"
        else
            fail "5-2b: qemu-arm 运行产出空文件"
        fi
    else
        fail "5-2b: qemu-arm 运行 run_model 失败" "$QEMU_OUTPUT"
    fi
else
    if [ -z "$QEMU_ARM" ]; then
        fail "5-2b: qemu-arm 不可用"
    elif [ ! -f "$ARM_RUN_MODEL" ]; then
        fail "5-2b: ARM 版 run_model 不存在"
    else
        fail "5-2b: ARM 版 libdeploy.so 不存在"
    fi
fi

# =====================================================================
# 任务 5-3：Host / ARM 输出对比
# =====================================================================
section "任务 5-3：Host / ARM 输出对比"

RUN_MODEL_HOST="$BUILD_DIR/run_model"

# 先在 Host 上运行
if [ -f "$RUN_MODEL_HOST" ] && [ -f "$OUT_MLP/libdeploy.so" ]; then
    python3 -c "import struct,sys; sys.stdout.buffer.write(struct.pack('4f',1,2,3,4))" > "$TMP_DIR/input.bin"
    LD_LIBRARY_PATH="$OUT_MLP" "$RUN_MODEL_HOST" "$OUT_MLP" "$TMP_DIR/input.bin" "$TMP_DIR/output_host.bin" > /dev/null 2>&1 || true
fi

# 比较 Host 和 ARM 输出
if [ -f "$TMP_DIR/output_host.bin" ] && [ -f "$TMP_DIR/output_arm.bin" ]; then
    COMPARE_RESULT=$(python3 << 'PYEOF'
import struct, sys

def read_floats(path):
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 4
    return struct.unpack(f'{n}f', data)

try:
    host = read_floats('/tmp/phase5_host.bin')
    arm = read_floats('/tmp/phase5_arm.bin')
except:
    # 使用环境变量中的路径
    import os
    tmp = os.environ.get('TMP_DIR', '/tmp')
    host = read_floats(f'{tmp}/output_host.bin')
    arm = read_floats(f'{tmp}/output_arm.bin')

if len(host) != len(arm):
    print(f"SIZE_MISMATCH: host={len(host)} arm={len(arm)}")
    sys.exit(1)

max_diff = 0.0
for i, (h, a) in enumerate(zip(host, arm)):
    diff = abs(h - a)
    max_diff = max(max_diff, diff)
    if diff > 1e-5:
        print(f"DIFF_TOO_LARGE: index={i} host={h} arm={a} diff={diff}")
        sys.exit(1)

print(f"COMPARE_OK max_diff={max_diff}")
PYEOF
    )
    export TMP_DIR  # 确保 Python 能读到

    COMPARE_RESULT=$(TMP_DIR="$TMP_DIR" python3 -c "
import struct, sys, os

def read_floats(path):
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 4
    return struct.unpack(f'{n}f', data)

tmp = os.environ.get('TMP_DIR', '/tmp')
host = read_floats(f'{tmp}/output_host.bin')
arm = read_floats(f'{tmp}/output_arm.bin')

if len(host) != len(arm):
    print(f'SIZE_MISMATCH: host={len(host)} arm={len(arm)}')
    sys.exit(1)

max_diff = 0.0
for i, (h, a) in enumerate(zip(host, arm)):
    diff = abs(h - a)
    max_diff = max(max_diff, diff)
    if diff > 1e-5:
        print(f'DIFF_TOO_LARGE: index={i} host={h} arm={a} diff={diff}')
        sys.exit(1)

print(f'COMPARE_OK max_diff={max_diff}')
" 2>/dev/null)

    if echo "$COMPARE_RESULT" | grep -q "COMPARE_OK"; then
        pass "5-3a: Host / ARM MLP 输出一致"
        echo "       $COMPARE_RESULT"
    else
        fail "5-3a: Host / ARM MLP 输出不一致" "$COMPARE_RESULT"
    fi
else
    if [ ! -f "$TMP_DIR/output_host.bin" ]; then
        fail "5-3a: Host 输出不存在"
    elif [ ! -f "$TMP_DIR/output_arm.bin" ]; then
        fail "5-3a: ARM 输出不存在"
    fi
fi

# CNN 对比（如果 CNN 模型存在）
CNN_JSON="$ROOT_DIR/examples/json/cnn.json"
if [ -f "$CNN_JSON" ] && [ -f "$TTVMC" ]; then
    OUT_CNN="$TMP_DIR/out_cnn"
    mkdir -p "$OUT_CNN"
    "$TTVMC" compile "$CNN_JSON" -o "$OUT_CNN" > /dev/null 2>&1 || true

    if [ -f "$OUT_CNN/deploy.c" ] && [ -n "$ARM_GCC" ]; then
        # 编译 ARM 版 CNN libdeploy.so
        $ARM_GCC -shared -fPIC "$OUT_CNN/deploy.c" -O2 -o "$OUT_CNN/libdeploy_arm.so" 2>/dev/null || true

        if [ -f "$OUT_CNN/libdeploy.so" ] && [ -f "$OUT_CNN/libdeploy_arm.so" ]; then
            # 获取 CNN 输入大小
            INPUT_SIZE=$(python3 -c "
import json
with open('$OUT_CNN/graph.json') as f:
    data = json.load(f)
for idx in data.get('graph_inputs', []):
    t = data['tensors'][idx]
    size = 1
    for s in t['shape']: size *= s
    print(size * 4)
    break
" 2>/dev/null)
            if [ "${INPUT_SIZE:-0}" -gt 0 ]; then
                python3 -c "
import struct,sys
n=$INPUT_SIZE//4
sys.stdout.buffer.write(struct.pack(f'{n}f', *[0.5]*n))
" > "$TMP_DIR/cnn_input.bin"

                # Host run
                LD_LIBRARY_PATH="$OUT_CNN" "$RUN_MODEL_HOST" "$OUT_CNN" "$TMP_DIR/cnn_input.bin" "$TMP_DIR/cnn_out_host.bin" > /dev/null 2>&1 || true
                # ARM run
                if [ -n "$QEMU_ARM" ] && [ -f "$ARM_RUN_MODEL" ]; then
                    cp "$OUT_CNN/libdeploy_arm.so" "$OUT_CNN/libdeploy.so.bak" 2>/dev/null || true
                    LD_LIBRARY_PATH="$OUT_CNN" $QEMU_ARM -L /usr/arm-linux-gnueabihf \
                        "$ARM_RUN_MODEL" "$OUT_CNN" "$TMP_DIR/cnn_input.bin" "$TMP_DIR/cnn_out_arm.bin" > /dev/null 2>&1 || true
                fi

                if [ -f "$TMP_DIR/cnn_out_host.bin" ] && [ -f "$TMP_DIR/cnn_out_arm.bin" ]; then
                    pass "5-3b: CNN 模型 Host/ARM 均产出结果"
                else
                    fail "5-3b: CNN 模型 Host 或 ARM 未产出结果"
                fi
            else
                fail "5-3b: 无法确定 CNN 输入大小"
            fi
        else
            fail "5-3b: CNN libdeploy.so 编译失败"
        fi
    else
        fail "5-3b: CNN deploy.c 不存在或 ARM 编译器不可用"
    fi
else
    fail "5-3b: CNN 模型或 ttvmc 不存在"
fi

# =====================================================================
# 汇总
# =====================================================================
section "Phase 5 测试结果汇总"

echo ""
echo "  通过: $PASS_COUNT"
echo "  失败: $FAIL_COUNT"
echo "  总计: $TOTAL_COUNT"
echo ""
echo "  Score: $PASS_COUNT/$TOTAL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "🎉 Phase 5 所有检测项通过！ARM/QEMU 部署完成。"
    exit 0
else
    echo "⚠️  Phase 5 有 $FAIL_COUNT 项未通过，请根据上方 [FAIL] 提示修复。"
    exit 1
fi
