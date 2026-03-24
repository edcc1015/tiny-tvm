#!/usr/bin/env bash
# =============================================================================
# Phase 6 测试脚本：Type System & Multi-DType
# =============================================================================
# 检测要点：
#   6-1: DType 枚举扩展（kFloat16, kInt8, kUInt8）
#   6-2: dtype_size 正确性
#   6-3: dtype_to_string / string_to_dtype 往返转换
#   6-4: InferShapePass dtype 传播（int8 MatMul -> int32 输出）
#   6-5: TypeCheckPass 存在并可编译
#   6-6: TypeCheckPass 捕获类型不匹配
#   6-7: int8 MLP 端到端（mlp_int8.json）
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
# 任务 6-1：DType 枚举扩展（kFloat16, kInt8, kUInt8）
# =====================================================================
section "任务 6-1：DType 枚举扩展"

cat > "$TMP_DIR/check_dtype_enum.cpp" << 'CHECKER_EOF'
#include <iostream>
#include "tiny_tvm/ir/graph.h"

using namespace tiny_tvm::ir;

int main() {
    // Verify new enum values exist and are distinct
    DType f16 = DType::kFloat16;
    DType i8  = DType::kInt8;
    DType u8  = DType::kUInt8;

    // Also verify existing values still work
    DType f32 = DType::kFloat32;
    DType i32 = DType::kInt32;
    DType unk = DType::kUnknown;

    // Check all six values are distinct
    int vals[] = {
        static_cast<int>(unk), static_cast<int>(f32), static_cast<int>(i32),
        static_cast<int>(f16), static_cast<int>(i8),  static_cast<int>(u8)
    };
    for (int a = 0; a < 6; a++)
        for (int b = a + 1; b < 6; b++)
            if (vals[a] == vals[b]) {
                std::cout << "CHECK_FAIL: DType values not distinct\n";
                return 0;
            }

    std::cout << "CHECK_OK\n";
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_dtype_enum.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_dtype_enum" 2>"$TMP_DIR/dtype_enum_compile.log"; then
    ENUM_OUTPUT=$("$TMP_DIR/check_dtype_enum" 2>&1)
    if echo "$ENUM_OUTPUT" | grep -q "CHECK_OK"; then
        pass "6-1a: DType::kFloat16, kInt8, kUInt8 存在且值互不相同 (3 pts)"
    else
        fail "6-1a: DType 枚举值不正确" "$ENUM_OUTPUT"
    fi
else
    fail "6-1a: DType 枚举扩展编译失败（kFloat16/kInt8/kUInt8 不存在）" "$(head -5 "$TMP_DIR/dtype_enum_compile.log")"
fi

# =====================================================================
# 任务 6-2：dtype_size 正确性
# =====================================================================
section "任务 6-2：dtype_size 正确性"

cat > "$TMP_DIR/check_dtype_size.cpp" << 'CHECKER_EOF'
#include <iostream>
#include "tiny_tvm/ir/graph.h"

using namespace tiny_tvm::ir;

int main() {
    bool ok = true;

    // New types
    if (dtype_size(DType::kFloat16) != 2) {
        std::cout << "CHECK_FAIL: dtype_size(kFloat16) expected 2, got "
                  << dtype_size(DType::kFloat16) << "\n";
        ok = false;
    }
    if (dtype_size(DType::kInt8) != 1) {
        std::cout << "CHECK_FAIL: dtype_size(kInt8) expected 1, got "
                  << dtype_size(DType::kInt8) << "\n";
        ok = false;
    }
    if (dtype_size(DType::kUInt8) != 1) {
        std::cout << "CHECK_FAIL: dtype_size(kUInt8) expected 1, got "
                  << dtype_size(DType::kUInt8) << "\n";
        ok = false;
    }

    // Existing types should still be correct
    if (dtype_size(DType::kFloat32) != 4) {
        std::cout << "CHECK_FAIL: dtype_size(kFloat32) expected 4, got "
                  << dtype_size(DType::kFloat32) << "\n";
        ok = false;
    }
    if (dtype_size(DType::kInt32) != 4) {
        std::cout << "CHECK_FAIL: dtype_size(kInt32) expected 4, got "
                  << dtype_size(DType::kInt32) << "\n";
        ok = false;
    }

    if (ok) {
        std::cout << "CHECK_OK\n";
    }
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_dtype_size.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_dtype_size" 2>"$TMP_DIR/dtype_size_compile.log"; then
    SIZE_OUTPUT=$("$TMP_DIR/check_dtype_size" 2>&1)
    if echo "$SIZE_OUTPUT" | grep -q "CHECK_OK"; then
        pass "6-2a: dtype_size(kFloat16)==2, dtype_size(kInt8)==1, dtype_size(kUInt8)==1 (3 pts)"
    else
        fail "6-2a: dtype_size 返回值不正确" "$SIZE_OUTPUT"
    fi
else
    fail "6-2a: dtype_size checker 编译失败" "$(head -5 "$TMP_DIR/dtype_size_compile.log")"
fi

# =====================================================================
# 任务 6-3：dtype_to_string / string_to_dtype 往返转换
# =====================================================================
section "任务 6-3：dtype_to_string / string_to_dtype"

cat > "$TMP_DIR/check_dtype_string.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <string>
#include "tiny_tvm/ir/graph.h"

using namespace tiny_tvm::ir;

int main() {
    bool ok = true;

    // Test all DType values for round-trip: dtype -> string -> dtype
    DType types[] = {
        DType::kFloat32, DType::kInt32,
        DType::kFloat16, DType::kInt8, DType::kUInt8
    };
    const char* expected_names[] = {
        "float32", "int32",
        "float16", "int8", "uint8"
    };

    for (int i = 0; i < 5; i++) {
        std::string s = dtype_to_string(types[i]);
        if (s != expected_names[i]) {
            std::cout << "CHECK_FAIL: dtype_to_string mismatch for type " << i
                      << ": expected \"" << expected_names[i]
                      << "\", got \"" << s << "\"\n";
            ok = false;
            continue;
        }

        DType roundtrip = string_to_dtype(s);
        if (roundtrip != types[i]) {
            std::cout << "CHECK_FAIL: string_to_dtype(\"" << s
                      << "\") did not round-trip correctly\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "CHECK_OK\n";
    }
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_dtype_string.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_dtype_string" 2>"$TMP_DIR/dtype_string_compile.log"; then
    STR_OUTPUT=$("$TMP_DIR/check_dtype_string" 2>&1)
    if echo "$STR_OUTPUT" | grep -q "CHECK_OK"; then
        pass "6-3a: dtype_to_string / string_to_dtype 全部 5 种类型往返正确 (3 pts)"
    else
        fail "6-3a: dtype_to_string / string_to_dtype 往返转换失败" "$STR_OUTPUT"
    fi
else
    fail "6-3a: dtype_to_string / string_to_dtype checker 编译失败（函数可能不存在）" "$(head -5 "$TMP_DIR/dtype_string_compile.log")"
fi

# =====================================================================
# 任务 6-4：InferShapePass dtype 传播
# =====================================================================
section "任务 6-4：InferShapePass dtype 传播"

cat > "$TMP_DIR/check_dtype_propagation.cpp" << 'CHECKER_EOF'
#include <iostream>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"

#if __has_include("tiny_tvm/pass/graph/infer_shape_pass.h")
#include "tiny_tvm/pass/graph/infer_shape_pass.h"
#define HAS_ISP 1
#else
#define HAS_ISP 0
#endif

using namespace tiny_tvm::ir;

// Helper: build a MatMul graph with specified input dtype
Graph make_matmul_graph(DType input_dtype) {
    Graph g;

    Tensor a; a.name = "A"; a.shape = {2, 4}; a.dtype = input_dtype;
    int ta = g.add_tensor(std::move(a));

    Tensor b; b.name = "B"; b.shape = {4, 8}; b.dtype = input_dtype;
    b.is_constant = true;
    int tb = g.add_tensor(std::move(b));

    Tensor out; out.name = "out"; out.shape = {}; out.dtype = DType::kUnknown;
    int to = g.add_tensor(std::move(out));

    Op matmul;
    matmul.kind = "MatMul";
    matmul.inputs = {ta, tb};
    matmul.outputs = {to};
    g.add_op(std::move(matmul));

    g.graph_inputs().push_back(ta);
    g.graph_outputs().push_back(to);
    return g;
}

int main() {
#if !HAS_ISP
    std::cout << "CHECK_FAIL: InferShapePass header not found\n";
    return 0;
#else
    bool ok = true;

    // Test 1: MatMul(int8, int8) -> output should be int32 (widened accumulation)
    {
        Graph g = make_matmul_graph(DType::kInt8);
        tiny_tvm::passes::PassManager pm;
        pm.add(std::make_unique<tiny_tvm::passes::InferShapePass>());
        pm.run(g);

        DType out_dtype = g.tensor(g.ops()[0].outputs[0]).dtype;
        if (out_dtype != DType::kInt32) {
            std::cout << "CHECK_FAIL: MatMul(int8,int8) output dtype is not int32"
                      << " (got " << static_cast<int>(out_dtype) << ")\n";
            ok = false;
        }
    }

    // Test 2: MatMul(float32, float32) -> output should be float32
    {
        Graph g = make_matmul_graph(DType::kFloat32);
        tiny_tvm::passes::PassManager pm;
        pm.add(std::make_unique<tiny_tvm::passes::InferShapePass>());
        pm.run(g);

        DType out_dtype = g.tensor(g.ops()[0].outputs[0]).dtype;
        if (out_dtype != DType::kFloat32) {
            std::cout << "CHECK_FAIL: MatMul(f32,f32) output dtype is not float32"
                      << " (got " << static_cast<int>(out_dtype) << ")\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "CHECK_OK\n";
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_dtype_propagation.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_dtype_propagation" 2>"$TMP_DIR/dtype_prop_compile.log"; then
    PROP_OUTPUT=$("$TMP_DIR/check_dtype_propagation" 2>&1)
    if echo "$PROP_OUTPUT" | grep -q "CHECK_OK"; then
        pass "6-4a: InferShapePass dtype 传播正确：int8→int32, f32→f32 (3 pts)"
    else
        fail "6-4a: InferShapePass dtype 传播不正确" "$PROP_OUTPUT"
    fi
else
    fail "6-4a: InferShapePass dtype 传播 checker 编译失败" "$(head -5 "$TMP_DIR/dtype_prop_compile.log")"
fi

# =====================================================================
# 任务 6-5：TypeCheckPass 存在并可编译
# =====================================================================
section "任务 6-5：TypeCheckPass 存在并可编译"

# 先检查头文件是否存在
TC_H="$ROOT_DIR/include/tiny_tvm/pass/graph/type_check_pass.h"
TC_CPP="$ROOT_DIR/src/pass/graph/type_check_pass.cpp"

if [ -f "$TC_H" ]; then
    pass "6-5a: type_check_pass.h 存在"
else
    fail "6-5a: type_check_pass.h 不存在"
fi

cat > "$TMP_DIR/check_type_check_pass.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <memory>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"

#if __has_include("tiny_tvm/pass/graph/type_check_pass.h")
#include "tiny_tvm/pass/graph/type_check_pass.h"
#define HAS_TCP 1
#else
#define HAS_TCP 0
#endif

int main() {
#if !HAS_TCP
    std::cout << "CHECK_FAIL: TypeCheckPass header not found\n";
    return 0;
#else
    // Instantiate TypeCheckPass and verify it compiles and can be added to PM
    auto tcp = std::make_unique<tiny_tvm::passes::TypeCheckPass>();

    // Verify it has a name
    std::string n = tcp->name();
    if (n.empty()) {
        std::cout << "CHECK_FAIL: TypeCheckPass::name() returned empty string\n";
        return 0;
    }

    // Verify it can be added to PassManager
    tiny_tvm::passes::PassManager pm;
    pm.add(std::move(tcp));

    // Run on a trivially correct graph (should not throw)
    tiny_tvm::ir::Graph g;
    tiny_tvm::ir::Tensor t;
    t.name = "x"; t.shape = {1, 4}; t.dtype = tiny_tvm::ir::DType::kFloat32;
    int ti = g.add_tensor(std::move(t));
    tiny_tvm::ir::Tensor w;
    w.name = "w"; w.shape = {4, 2}; w.dtype = tiny_tvm::ir::DType::kFloat32;
    w.is_constant = true;
    int wi = g.add_tensor(std::move(w));
    tiny_tvm::ir::Tensor o;
    o.name = "o"; o.shape = {1, 2}; o.dtype = tiny_tvm::ir::DType::kFloat32;
    int oi = g.add_tensor(std::move(o));

    tiny_tvm::ir::Op op;
    op.kind = "MatMul"; op.inputs = {ti, wi}; op.outputs = {oi};
    g.add_op(std::move(op));
    g.graph_inputs().push_back(ti);
    g.graph_outputs().push_back(oi);

    try {
        pm.run(g);
    } catch (...) {
        std::cout << "CHECK_FAIL: TypeCheckPass threw on a valid graph\n";
        return 0;
    }

    std::cout << "CHECK_OK\n";
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_type_check_pass.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_type_check_pass" 2>"$TMP_DIR/tcp_compile.log"; then
    TCP_OUTPUT=$("$TMP_DIR/check_type_check_pass" 2>&1)
    if echo "$TCP_OUTPUT" | grep -q "CHECK_OK"; then
        pass "6-5b: TypeCheckPass 可实例化、注册到 PassManager 并运行 (2 pts)"
    else
        fail "6-5b: TypeCheckPass 运行不正确" "$TCP_OUTPUT"
    fi
else
    fail "6-5b: TypeCheckPass checker 编译失败" "$(head -5 "$TMP_DIR/tcp_compile.log")"
fi

# =====================================================================
# 任务 6-6：TypeCheckPass 捕获类型不匹配
# =====================================================================
section "任务 6-6：TypeCheckPass 捕获类型不匹配"

cat > "$TMP_DIR/check_type_mismatch.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <memory>
#include <stdexcept>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"

#if __has_include("tiny_tvm/pass/graph/type_check_pass.h")
#include "tiny_tvm/pass/graph/type_check_pass.h"
#define HAS_TCP 1
#else
#define HAS_TCP 0
#endif

using namespace tiny_tvm::ir;

int main() {
#if !HAS_TCP
    std::cout << "CHECK_FAIL: TypeCheckPass header not found\n";
    return 0;
#else
    // Build a graph with mismatched dtypes: MatMul(float32, int8)
    Graph g;

    Tensor a; a.name = "A"; a.shape = {2, 4}; a.dtype = DType::kFloat32;
    int ta = g.add_tensor(std::move(a));

    Tensor b; b.name = "B"; b.shape = {4, 8}; b.dtype = DType::kInt8;
    b.is_constant = true;
    int tb = g.add_tensor(std::move(b));

    Tensor out; out.name = "out"; out.shape = {2, 8}; out.dtype = DType::kFloat32;
    int to = g.add_tensor(std::move(out));

    Op matmul;
    matmul.kind = "MatMul";
    matmul.inputs = {ta, tb};
    matmul.outputs = {to};
    g.add_op(std::move(matmul));
    g.graph_inputs().push_back(ta);
    g.graph_outputs().push_back(to);

    // TypeCheckPass should throw or report an error for mismatched input dtypes
    tiny_tvm::passes::PassManager pm;
    pm.add(std::make_unique<tiny_tvm::passes::TypeCheckPass>());

    bool caught = false;
    try {
        pm.run(g);
    } catch (const std::exception& e) {
        caught = true;
        std::cout << "Exception caught: " << e.what() << "\n";
    } catch (...) {
        caught = true;
        std::cout << "Unknown exception caught\n";
    }

    if (caught) {
        std::cout << "CHECK_OK\n";
    } else {
        std::cout << "CHECK_FAIL: TypeCheckPass did not throw on mismatched dtypes "
                  << "(float32 vs int8 MatMul inputs)\n";
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_type_mismatch.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_type_mismatch" 2>"$TMP_DIR/tm_compile.log"; then
    TM_OUTPUT=$("$TMP_DIR/check_type_mismatch" 2>&1)
    if echo "$TM_OUTPUT" | grep -q "CHECK_OK"; then
        pass "6-6a: TypeCheckPass 对 float32/int8 不匹配正确抛出异常 (3 pts)"
    else
        fail "6-6a: TypeCheckPass 未捕获类型不匹配" "$TM_OUTPUT"
    fi
else
    fail "6-6a: TypeCheckPass mismatch checker 编译失败" "$(head -5 "$TMP_DIR/tm_compile.log")"
fi

# =====================================================================
# 任务 6-7：int8 MLP 端到端
# =====================================================================
section "任务 6-7：int8 MLP 端到端"

MLP_INT8_JSON="$ROOT_DIR/examples/json/mlp_int8.json"

if [ -f "$MLP_INT8_JSON" ]; then
    pass "6-7a: examples/json/mlp_int8.json 存在"

    # Validate JSON structure has int8 dtype references
    if python3 -c "
import json, sys
with open('$MLP_INT8_JSON') as f:
    data = json.load(f)
tensors = data.get('tensors', [])
has_int8 = any('int8' in str(t.get('dtype', '')) for t in tensors)
if has_int8:
    print('HAS_INT8')
else:
    print('NO_INT8')
" 2>/dev/null | grep -q "HAS_INT8"; then
        pass "6-7b: mlp_int8.json 包含 int8 dtype 的 tensor"
    else
        fail "6-7b: mlp_int8.json 不包含 int8 dtype 的 tensor"
    fi

    # If ttvmc exists, attempt to compile the int8 model
    if [ -f "$TTVMC" ]; then
        OUT_INT8="$TMP_DIR/out_int8"
        mkdir -p "$OUT_INT8"
        if "$TTVMC" compile "$MLP_INT8_JSON" -o "$OUT_INT8" > /dev/null 2>&1; then
            if [ -f "$OUT_INT8/graph.json" ]; then
                # Verify output contains int8/int32 dtype references
                if python3 -c "
import json
with open('$OUT_INT8/graph.json') as f:
    data = json.load(f)
tensors = data.get('tensors', [])
dtypes = set(t.get('dtype', '') for t in tensors)
# Expect at least int8 or int32 in the compiled graph
has_narrow = any(d in dtypes for d in ['int8', 'uint8', 'int32'])
print('INT8_COMPILE_OK' if has_narrow else 'NO_NARROW_TYPES')
" 2>/dev/null | grep -q "INT8_COMPILE_OK"; then
                    pass "6-7c: ttvmc 成功编译 int8 MLP 模型 (3 pts)"
                else
                    fail "6-7c: 编译后 graph.json 不包含窄类型 tensor"
                fi
            else
                fail "6-7c: ttvmc compile 未生成 graph.json"
            fi
        else
            fail "6-7c: ttvmc compile mlp_int8.json 失败"
        fi
    else
        fail "6-7c: ttvmc 不存在，跳过 int8 E2E 编译"
    fi
else
    fail "6-7a: examples/json/mlp_int8.json 不存在"
    fail "6-7b: 跳过（mlp_int8.json 不存在）"
    fail "6-7c: 跳过（mlp_int8.json 不存在）"
fi

# =====================================================================
# 汇总
# =====================================================================
section "Phase 6 测试结果汇总"

echo ""
echo "========== Phase 6 Test Results =========="
echo "Score: $PASS_COUNT / $TOTAL_COUNT"
echo "Pass: $PASS_COUNT  Fail: $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "🎉 Phase 6 所有检测项通过！Type System & Multi-DType 完成。"
    exit 0
else
    echo "⚠️  Phase 6 有 $FAIL_COUNT 项未通过，请根据上方 [FAIL] 提示修复。"
    exit 1
fi
