#!/usr/bin/env bash
# =============================================================================
# Phase 2 测试脚本：推理系统增强（Params / Conv / Fusion）
# =============================================================================
# 检测要点：
#   2-1: 参数导出与加载（params.bin 完整流程）
#   2-2: Conv2D 实现（shape infer + codegen + 数值正确）
#   2-3: ConstantFoldPass（常量折叠）
#   2-4: FusePass（Conv + Relu 融合）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0

pass() { PASS_COUNT=$((PASS_COUNT + 1)); TOTAL_COUNT=$((TOTAL_COUNT + 1)); echo "[PASS] $1"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); TOTAL_COUNT=$((TOTAL_COUNT + 1)); echo "[FAIL] $1"; [ -n "${2:-}" ] && echo "       Reason: $2"; }
section() { echo ""; echo "======================================================================"; echo "  $1"; echo "======================================================================"; }

# 前置编译
section "前置：编译项目"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug > /dev/null 2>&1 || true
cmake --build "$BUILD_DIR" --parallel > /dev/null 2>&1 || echo "WARNING: 编译有错误"

TTVMC="$BUILD_DIR/ttvmc"
INCLUDE_DIRS="-I$ROOT_DIR/include"
[ -d "$ROOT_DIR/third_party" ] && INCLUDE_DIRS="$INCLUDE_DIRS -I$ROOT_DIR/third_party"
CORE_SOURCES=$(find "$ROOT_DIR/src" -name "*.cpp" ! -path "*/tools/*" 2>/dev/null | tr '\n' ' ')

# =====================================================================
# 任务 2-1：参数导出与加载
# =====================================================================
section "任务 2-1：参数导出与加载"

# 检查 params.bin 导出逻辑 — 通过编译 MLP 并检查 params.bin 非空
MLP_JSON="$ROOT_DIR/examples/json/mlp.json"
OUT_MLP="$TMP_DIR/out_mlp"
mkdir -p "$OUT_MLP"

if [ -f "$TTVMC" ]; then
    "$TTVMC" compile "$MLP_JSON" -o "$OUT_MLP" > /dev/null 2>&1 || true
fi

if [ -f "$OUT_MLP/params.bin" ]; then
    PSIZE=$(stat -c%s "$OUT_MLP/params.bin" 2>/dev/null || echo "0")
    if [ "$PSIZE" -gt 0 ]; then
        pass "2-1a: params.bin 非空（$PSIZE bytes）"
    else
        fail "2-1a: params.bin 为空"
    fi
else
    fail "2-1a: params.bin 不存在（ttvmc compile 可能未成功）"
fi

# 检查 graph.json 中常量 tensor 有 param_offset
if [ -f "$OUT_MLP/graph.json" ]; then
    PARAM_OFFSET_CHECK=$(python3 -c "
import json, sys
with open('$OUT_MLP/graph.json') as f:
    data = json.load(f)
for t in data.get('tensors', []):
    if t.get('is_constant', False):
        if 'param_offset' in t:
            print('HAS_PARAM_OFFSET')
            sys.exit(0)
print('NO_PARAM_OFFSET')
" 2>/dev/null)
    if [ "$PARAM_OFFSET_CHECK" = "HAS_PARAM_OFFSET" ]; then
        pass "2-1b: graph.json 中常量 tensor 包含 param_offset"
    else
        fail "2-1b: graph.json 中常量 tensor 缺少 param_offset"
    fi
else
    fail "2-1b: graph.json 不存在"
fi

# 验证 params.bin 可被 runtime 正确加载 — 通过 run_model E2E
RUN_MODEL="$BUILD_DIR/run_model"
if [ -f "$RUN_MODEL" ] && [ -f "$OUT_MLP/libdeploy.so" ]; then
    python3 -c "import struct,sys; sys.stdout.buffer.write(struct.pack('4f',1,2,3,4))" > "$TMP_DIR/input.bin"
    LD_LIBRARY_PATH="$OUT_MLP" "$RUN_MODEL" "$OUT_MLP" "$TMP_DIR/input.bin" "$TMP_DIR/output_p1.bin" > /dev/null 2>&1 || true
    if [ -f "$TMP_DIR/output_p1.bin" ] && [ "$(stat -c%s "$TMP_DIR/output_p1.bin" 2>/dev/null)" -gt 0 ]; then
        pass "2-1c: run_model 成功从 params.bin 加载常量并产出结果"
    else
        fail "2-1c: run_model 未能从 params.bin 加载常量执行推理"
    fi
else
    fail "2-1c: run_model 或 libdeploy.so 不存在"
fi

# =====================================================================
# 任务 2-2：Conv2D 实现
# =====================================================================
section "任务 2-2：Conv2D 实现"

# 检查 cnn.json 存在
CNN_JSON="$ROOT_DIR/examples/json/cnn.json"
if [ -f "$CNN_JSON" ]; then
    pass "2-2a: examples/json/cnn.json 存在"
else
    fail "2-2a: examples/json/cnn.json 不存在" "需要提供 CNN 示例模型"
fi

# 检查 InferShapePass 支持 Conv2D
INFER_CPP="$ROOT_DIR/src/pass/graph/infer_shape_pass.cpp"
if [ -f "$INFER_CPP" ] && grep -qi 'conv' "$INFER_CPP"; then
    pass "2-2b: InferShapePass 包含 Conv2D shape 推导逻辑"
else
    fail "2-2b: InferShapePass 缺少 Conv2D shape 推导"
fi

# 检查 codegen 支持 Conv2D
CODEGEN_CPP="$ROOT_DIR/src/codegen/c_codegen.cpp"
if grep -qi 'conv' "$CODEGEN_CPP" 2>/dev/null; then
    pass "2-2c: c_codegen.cpp 包含 Conv2D 代码生成"
else
    fail "2-2c: c_codegen.cpp 缺少 Conv2D 代码生成"
fi

# 编译 CNN 并检查 deploy.c 包含 7 层循环（n, oc, oh, ow, ic, kh, kw）
OUT_CNN="$TMP_DIR/out_cnn"
mkdir -p "$OUT_CNN"

if [ -f "$TTVMC" ] && [ -f "$CNN_JSON" ]; then
    "$TTVMC" compile "$CNN_JSON" -o "$OUT_CNN" > /dev/null 2>&1 || true
    if [ -f "$OUT_CNN/deploy.c" ]; then
        # 计算 deploy.c 中 for 循环的数量
        FOR_COUNT=$(grep -cE 'for\s*\(' "$OUT_CNN/deploy.c" 2>/dev/null || echo "0")
        if [ "$FOR_COUNT" -ge 7 ]; then
            pass "2-2d: deploy.c 包含 >= 7 层 for 循环（Conv2D 完整 7 层循环）"
        else
            fail "2-2d: deploy.c for 循环数量不足（$FOR_COUNT < 7）"
        fi
    else
        fail "2-2d: CNN compile 未生成 deploy.c"
    fi
else
    fail "2-2d: 无法编译 CNN（ttvmc 或 cnn.json 不存在）"
fi

# CNN E2E 数值正确性
if [ -f "$RUN_MODEL" ] && [ -f "$OUT_CNN/libdeploy.so" ]; then
    # 创建 CNN 输入（NCHW，假设 1x1x8x8 = 256 bytes float32）
    # 需要从 graph.json 读取实际输入 shape
    INPUT_SIZE=$(python3 -c "
import json, sys, struct
try:
    with open('$OUT_CNN/graph.json') as f:
        data = json.load(f)
    for idx in data.get('graph_inputs', []):
        t = data['tensors'][idx]
        size = 1
        for s in t['shape']:
            size *= s
        print(size * 4)
        break
except:
    print(0)
" 2>/dev/null)
    if [ "${INPUT_SIZE:-0}" -gt 0 ]; then
        python3 -c "
import struct, sys
n = $INPUT_SIZE // 4
data = struct.pack(f'{n}f', *[1.0]*n)
sys.stdout.buffer.write(data)
" > "$TMP_DIR/cnn_input.bin"
        LD_LIBRARY_PATH="$OUT_CNN" "$RUN_MODEL" "$OUT_CNN" "$TMP_DIR/cnn_input.bin" "$TMP_DIR/cnn_output.bin" > /dev/null 2>&1 || true
        if [ -f "$TMP_DIR/cnn_output.bin" ] && [ "$(stat -c%s "$TMP_DIR/cnn_output.bin" 2>/dev/null)" -gt 0 ]; then
            pass "2-2e: CNN 模型 E2E 运行成功（输出非空）"
        else
            fail "2-2e: CNN E2E 运行失败"
        fi
    else
        fail "2-2e: 无法确定 CNN 输入大小"
    fi
else
    fail "2-2e: run_model 或 CNN libdeploy.so 不存在"
fi

# =====================================================================
# 任务 2-3：ConstantFoldPass
# =====================================================================
section "任务 2-3：ConstantFoldPass"

CF_H="$ROOT_DIR/include/tiny_tvm/pass/graph/constant_fold_pass.h"
CF_CPP="$ROOT_DIR/src/pass/graph/constant_fold_pass.cpp"

if [ -f "$CF_H" ]; then
    pass "2-3a: constant_fold_pass.h 存在"
else
    fail "2-3a: constant_fold_pass.h 不存在"
fi

if [ -f "$CF_CPP" ]; then
    pass "2-3b: constant_fold_pass.cpp 存在"
else
    fail "2-3b: constant_fold_pass.cpp 不存在"
fi

# 编译 checker 验证常量折叠效果
cat > "$TMP_DIR/check_const_fold.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <cstring>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"

#if __has_include("tiny_tvm/pass/graph/constant_fold_pass.h")
#include "tiny_tvm/pass/graph/constant_fold_pass.h"
#define HAS_CF 1
#else
#define HAS_CF 0
#endif

using namespace tiny_tvm::ir;

int main() {
#if !HAS_CF
    std::cerr << "ERROR: ConstantFoldPass not found\n";
    return 1;
#else
    // 构造 Add(const_a, const_b) -> result
    Graph graph;

    Tensor a;
    a.name = "const_a";
    a.shape = {1, 4};
    a.dtype = DType::kFloat32;
    a.is_constant = true;
    a.nbytes = 16;
    a.data.resize(16);
    // 填充 [1.0, 2.0, 3.0, 4.0]
    float vals_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(a.data.data(), vals_a, 16);
    int t0 = graph.add_tensor(std::move(a));

    Tensor b;
    b.name = "const_b";
    b.shape = {1, 4};
    b.dtype = DType::kFloat32;
    b.is_constant = true;
    b.nbytes = 16;
    b.data.resize(16);
    float vals_b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    std::memcpy(b.data.data(), vals_b, 16);
    int t1 = graph.add_tensor(std::move(b));

    Tensor result;
    result.name = "result";
    result.shape = {1, 4};
    result.dtype = DType::kFloat32;
    int t2 = graph.add_tensor(std::move(result));

    Op add_op;
    add_op.kind = "Add";
    add_op.inputs = {t0, t1};
    add_op.outputs = {t2};
    graph.add_op(std::move(add_op));
    graph.graph_inputs() = {};
    graph.graph_outputs().push_back(t2);

    size_t ops_before = graph.op_count();

    tiny_tvm::passes::PassManager pm;
    pm.add(std::make_unique<tiny_tvm::passes::ConstantFoldPass>());
    pm.run(graph);

    // 验证：result tensor 应变为 constant
    auto& res = graph.tensor(t2);
    if (!res.is_constant) {
        std::cerr << "ERROR: result tensor not marked as constant after fold\n";
        return 1;
    }

    // 验证折叠值：[6.0, 8.0, 10.0, 12.0]
    if (res.data.size() >= 16) {
        float out[4];
        std::memcpy(out, res.data.data(), 16);
        float expected[] = {6.0f, 8.0f, 10.0f, 12.0f};
        for (int i = 0; i < 4; i++) {
            if (std::abs(out[i] - expected[i]) > 1e-5f) {
                std::cerr << "ERROR: fold result[" << i << "]=" << out[i]
                          << " expected " << expected[i] << "\n";
                return 1;
            }
        }
    } else {
        std::cerr << "ERROR: result data size too small\n";
        return 1;
    }

    std::cout << "CONST_FOLD_CHECK_OK\n";
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_const_fold.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_const_fold" 2>"$TMP_DIR/cf_compile.log"; then
    CF_OUTPUT=$("$TMP_DIR/check_const_fold" 2>&1)
    if echo "$CF_OUTPUT" | grep -q "CONST_FOLD_CHECK_OK"; then
        pass "2-3c: ConstantFoldPass 正确折叠 Add(const,const)，值 = [6,8,10,12]"
    else
        fail "2-3c: ConstantFoldPass 折叠结果不正确" "$CF_OUTPUT"
    fi
else
    fail "2-3c: ConstantFoldPass checker 编译失败" "$(head -20 "$TMP_DIR/cf_compile.log")"
fi

# =====================================================================
# 任务 2-4：FusePass (Conv + Relu)
# =====================================================================
section "任务 2-4：FusePass (Conv + Relu)"

FUSE_H="$ROOT_DIR/include/tiny_tvm/pass/graph/fuse_pass.h"
FUSE_CPP="$ROOT_DIR/src/pass/graph/fuse_pass.cpp"

if [ -f "$FUSE_H" ]; then
    pass "2-4a: fuse_pass.h 存在"
else
    fail "2-4a: fuse_pass.h 不存在"
fi

if [ -f "$FUSE_CPP" ]; then
    pass "2-4b: fuse_pass.cpp 存在"
else
    fail "2-4b: fuse_pass.cpp 不存在"
fi

# 编译 checker 验证融合效果
cat > "$TMP_DIR/check_fuse.cpp" << 'CHECKER_EOF'
#include <iostream>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"

#if __has_include("tiny_tvm/pass/graph/fuse_pass.h")
#include "tiny_tvm/pass/graph/fuse_pass.h"
#define HAS_FUSE 1
#else
#define HAS_FUSE 0
#endif

using namespace tiny_tvm::ir;

int main() {
#if !HAS_FUSE
    std::cerr << "ERROR: FusePass not found\n";
    return 1;
#else
    // 构造 Conv -> Relu 图
    Graph graph;

    Tensor input;
    input.name = "input";
    input.shape = {1, 1, 8, 8};
    input.dtype = DType::kFloat32;
    int t_in = graph.add_tensor(std::move(input));

    Tensor weight;
    weight.name = "weight";
    weight.shape = {1, 1, 3, 3};
    weight.dtype = DType::kFloat32;
    weight.is_constant = true;
    int t_w = graph.add_tensor(std::move(weight));

    Tensor conv_out;
    conv_out.name = "conv_out";
    conv_out.shape = {1, 1, 6, 6};
    conv_out.dtype = DType::kFloat32;
    int t_conv = graph.add_tensor(std::move(conv_out));

    Tensor relu_out;
    relu_out.name = "relu_out";
    relu_out.shape = {1, 1, 6, 6};
    relu_out.dtype = DType::kFloat32;
    int t_relu = graph.add_tensor(std::move(relu_out));

    Op conv;
    conv.kind = "Conv2D";
    conv.inputs = {t_in, t_w};
    conv.outputs = {t_conv};
    graph.add_op(std::move(conv));

    Op relu;
    relu.kind = "Relu";
    relu.inputs = {t_conv};
    relu.outputs = {t_relu};
    graph.add_op(std::move(relu));

    graph.graph_inputs().push_back(t_in);
    graph.graph_outputs().push_back(t_relu);

    size_t ops_before = graph.op_count();

    tiny_tvm::passes::PassManager pm;
    pm.add(std::make_unique<tiny_tvm::passes::FusePass>());
    pm.run(graph);

    // 验证：应有 ConvRelu op
    bool found_conv_relu = false;
    bool found_standalone_relu = false;
    for (size_t i = 0; i < graph.op_count(); i++) {
        auto& op = graph.op(i);
        if (op.kind == "ConvRelu" || op.kind == "Conv2DRelu") {
            found_conv_relu = true;
        }
        if (op.kind == "Relu") {
            found_standalone_relu = true;
        }
    }

    if (found_conv_relu) {
        std::cout << "FUSE_FOUND_CONVRELU\n";
    } else {
        std::cerr << "ERROR: no ConvRelu op after fusion\n";
        return 1;
    }

    if (!found_standalone_relu) {
        std::cout << "FUSE_NO_STANDALONE_RELU\n";
    } else {
        std::cerr << "ERROR: standalone Relu still exists after fusion\n";
        return 1;
    }

    std::cout << "FUSE_CHECK_OK\n";
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_fuse.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_fuse" 2>"$TMP_DIR/fuse_compile.log"; then
    FUSE_OUTPUT=$("$TMP_DIR/check_fuse" 2>&1)
    if echo "$FUSE_OUTPUT" | grep -q "FUSE_CHECK_OK"; then
        pass "2-4c: FusePass 成功融合 Conv+Relu 为 ConvRelu"
    else
        fail "2-4c: FusePass 融合失败" "$FUSE_OUTPUT"
    fi
    if echo "$FUSE_OUTPUT" | grep -q "FUSE_NO_STANDALONE_RELU"; then
        pass "2-4d: 融合后不再有独立的 Relu op"
    else
        fail "2-4d: 融合后仍存在独立的 Relu op"
    fi
else
    fail "2-4c: FusePass checker 编译失败" "$(head -20 "$TMP_DIR/fuse_compile.log")"
    fail "2-4d: 跳过"
fi

# 检查 codegen 支持 ConvRelu
if grep -qi 'ConvRelu\|conv_relu\|convrelu' "$CODEGEN_CPP" 2>/dev/null; then
    pass "2-4e: c_codegen.cpp 包含 ConvRelu 代码生成分支"
else
    fail "2-4e: c_codegen.cpp 缺少 ConvRelu 代码生成分支"
fi

# =====================================================================
# 汇总
# =====================================================================
section "Phase 2 测试结果汇总"

echo ""
echo "  通过: $PASS_COUNT"
echo "  失败: $FAIL_COUNT"
echo "  总计: $TOTAL_COUNT"
echo ""
echo "  Score: $PASS_COUNT/$TOTAL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "🎉 Phase 2 所有检测项通过！推理系统增强完成。"
    exit 0
else
    echo "⚠️  Phase 2 有 $FAIL_COUNT 项未通过，请根据上方 [FAIL] 提示修复。"
    exit 1
fi
