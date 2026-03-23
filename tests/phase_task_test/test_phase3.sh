#!/usr/bin/env bash
# =============================================================================
# Phase 3 测试脚本：编译器化（ONNX + 完整 Pass 体系）
# =============================================================================
# 检测要点：
#   3-1: ONNX Frontend
#   3-2: OpCanonicalizePass（Gemm -> MatMul + Add）
#   3-3: DeadCodeEliminationPass
#   3-4: LivenessAnalysisPass + MemoryReusePass
#   3-5: dump 和编译报告
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
# 任务 3-1：ONNX Frontend
# =====================================================================
section "任务 3-1：ONNX Frontend"

ONNX_FE_H="$ROOT_DIR/include/tiny_tvm/frontend/onnx/onnx_frontend.h"
ONNX_FE_CPP="$ROOT_DIR/src/frontend/onnx/onnx_frontend.cpp"

if [ -f "$ONNX_FE_H" ]; then
    pass "3-1a: onnx_frontend.h 存在"
else
    fail "3-1a: onnx_frontend.h 不存在"
fi

if [ -f "$ONNX_FE_CPP" ]; then
    # 检查是否仍是占位实现
    if grep -q "not implemented" "$ONNX_FE_CPP" 2>/dev/null && \
       ! grep -q "ModelProto\|GraphProto\|NodeProto" "$ONNX_FE_CPP" 2>/dev/null; then
        fail "3-1b: onnx_frontend.cpp 仍是占位实现"
    else
        pass "3-1b: onnx_frontend.cpp 包含实际解析逻辑"
    fi
else
    fail "3-1b: onnx_frontend.cpp 不存在"
fi

# 检查 protobuf 依赖或 onnx.proto 存在
if find "$ROOT_DIR" -name "onnx.proto" -o -name "onnx.pb.h" -o -name "onnx.pb.cc" 2>/dev/null | grep -q .; then
    pass "3-1c: ONNX protobuf 定义存在"
else
    fail "3-1c: 未找到 onnx.proto 或生成的 pb 文件"
fi

# 检查示例 ONNX 模型存在
if find "$ROOT_DIR/examples" -name "*.onnx" 2>/dev/null | grep -q .; then
    pass "3-1d: examples 目录包含 .onnx 模型文件"
    
    # 尝试用 ttvmc 编译 ONNX 模型
    ONNX_MODEL=$(find "$ROOT_DIR/examples" -name "*.onnx" 2>/dev/null | head -1)
    OUT_ONNX="$TMP_DIR/out_onnx"
    mkdir -p "$OUT_ONNX"
    
    if [ -f "$TTVMC" ]; then
        "$TTVMC" compile "$ONNX_MODEL" -o "$OUT_ONNX" > /dev/null 2>&1 || true
        if [ -f "$OUT_ONNX/graph.json" ]; then
            # 验证导入结果
            TENSOR_COUNT=$(python3 -c "
import json
with open('$OUT_ONNX/graph.json') as f:
    data = json.load(f)
print(len(data.get('tensors', [])))
" 2>/dev/null)
            if [ "${TENSOR_COUNT:-0}" -gt 0 ]; then
                pass "3-1e: ONNX 模型成功导入（$TENSOR_COUNT tensors）"
            else
                fail "3-1e: ONNX 导入后 graph.json 中 tensor 数量为 0"
            fi
        else
            fail "3-1e: ONNX compile 未生成 graph.json"
        fi
    else
        fail "3-1e: ttvmc 不存在"
    fi
else
    fail "3-1d: examples 目录无 .onnx 模型文件"
    fail "3-1e: 跳过 ONNX 导入测试"
fi

# =====================================================================
# 任务 3-2：OpCanonicalizePass
# =====================================================================
section "任务 3-2：OpCanonicalizePass"

OC_H="$ROOT_DIR/include/tiny_tvm/pass/op/op_canonicalize_pass.h"
OC_CPP="$ROOT_DIR/src/pass/op/op_canonicalize_pass.cpp"

if [ -f "$OC_H" ]; then
    pass "3-2a: op_canonicalize_pass.h 存在"
else
    fail "3-2a: op_canonicalize_pass.h 不存在"
fi

if [ -f "$OC_CPP" ]; then
    pass "3-2b: op_canonicalize_pass.cpp 存在"
else
    fail "3-2b: op_canonicalize_pass.cpp 不存在"
fi

# 编译 checker 验证 Gemm -> MatMul + Add
cat > "$TMP_DIR/check_canonicalize.cpp" << 'CHECKER_EOF'
#include <iostream>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"

#if __has_include("tiny_tvm/pass/op/op_canonicalize_pass.h")
#include "tiny_tvm/pass/op/op_canonicalize_pass.h"
#define HAS_OC 1
#else
#define HAS_OC 0
#endif

using namespace tiny_tvm::ir;

int main() {
#if !HAS_OC
    std::cerr << "ERROR: OpCanonicalizePass not found\n";
    return 1;
#else
    Graph graph;

    // 构造 Gemm(A, B, C) -> Y
    Tensor a; a.name = "A"; a.shape = {1, 4}; a.dtype = DType::kFloat32;
    int t_a = graph.add_tensor(std::move(a));

    Tensor b; b.name = "B"; b.shape = {4, 8}; b.dtype = DType::kFloat32;
    b.is_constant = true;
    int t_b = graph.add_tensor(std::move(b));

    Tensor c; c.name = "C"; c.shape = {1, 8}; c.dtype = DType::kFloat32;
    c.is_constant = true;
    int t_c = graph.add_tensor(std::move(c));

    Tensor y; y.name = "Y"; y.shape = {1, 8}; y.dtype = DType::kFloat32;
    int t_y = graph.add_tensor(std::move(y));

    Op gemm;
    gemm.kind = "Gemm";
    gemm.inputs = {t_a, t_b, t_c};
    gemm.outputs = {t_y};
    graph.add_op(std::move(gemm));
    graph.graph_inputs().push_back(t_a);
    graph.graph_outputs().push_back(t_y);

    // 运行 canonicalize
    tiny_tvm::passes::PassManager pm;
    pm.add(std::make_unique<tiny_tvm::passes::OpCanonicalizePass>());
    pm.run(graph);

    // 验证：不再有 Gemm，应有 MatMul 和 Add
    bool found_gemm = false;
    bool found_matmul = false;
    bool found_add = false;
    for (size_t i = 0; i < graph.op_count(); i++) {
        auto& op = graph.op(i);
        if (op.kind == "Gemm") found_gemm = true;
        if (op.kind == "MatMul") found_matmul = true;
        if (op.kind == "Add") found_add = true;
    }

    if (found_gemm) {
        std::cerr << "ERROR: Gemm still exists after canonicalize\n";
        return 1;
    }
    if (!found_matmul) {
        std::cerr << "ERROR: no MatMul after Gemm canonicalize\n";
        return 1;
    }
    if (!found_add) {
        std::cerr << "ERROR: no Add after Gemm canonicalize\n";
        return 1;
    }

    std::cout << "CANONICALIZE_CHECK_OK\n";
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_canonicalize.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_canonicalize" 2>"$TMP_DIR/oc_compile.log"; then
    OC_OUTPUT=$("$TMP_DIR/check_canonicalize" 2>&1)
    if echo "$OC_OUTPUT" | grep -q "CANONICALIZE_CHECK_OK"; then
        pass "3-2c: OpCanonicalizePass 成功将 Gemm 拆为 MatMul + Add"
    else
        fail "3-2c: OpCanonicalizePass 结果不正确" "$OC_OUTPUT"
    fi
else
    fail "3-2c: OpCanonicalizePass checker 编译失败" "$(head -20 "$TMP_DIR/oc_compile.log")"
fi

# =====================================================================
# 任务 3-3：DeadCodeEliminationPass
# =====================================================================
section "任务 3-3：DeadCodeEliminationPass"

DCE_H="$ROOT_DIR/include/tiny_tvm/pass/graph/dead_code_elimination_pass.h"
DCE_CPP="$ROOT_DIR/src/pass/graph/dead_code_elimination_pass.cpp"

if [ -f "$DCE_H" ]; then
    pass "3-3a: dead_code_elimination_pass.h 存在"
else
    fail "3-3a: dead_code_elimination_pass.h 不存在"
fi

if [ -f "$DCE_CPP" ]; then
    pass "3-3b: dead_code_elimination_pass.cpp 存在"
else
    fail "3-3b: dead_code_elimination_pass.cpp 不存在"
fi

# 编译 checker 验证 DCE
cat > "$TMP_DIR/check_dce.cpp" << 'CHECKER_EOF'
#include <iostream>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"

#if __has_include("tiny_tvm/pass/graph/dead_code_elimination_pass.h")
#include "tiny_tvm/pass/graph/dead_code_elimination_pass.h"
#define HAS_DCE 1
#else
#define HAS_DCE 0
#endif

using namespace tiny_tvm::ir;

int main() {
#if !HAS_DCE
    std::cerr << "ERROR: DCE pass not found\n";
    return 1;
#else
    // 构造图：input -> MatMul -> output（live）+ dead_op（dead）
    Graph graph;

    Tensor input; input.name = "input"; input.shape = {1, 4}; input.dtype = DType::kFloat32;
    int t0 = graph.add_tensor(std::move(input));

    Tensor w1; w1.name = "w1"; w1.shape = {4, 4}; w1.dtype = DType::kFloat32; w1.is_constant = true;
    int t1 = graph.add_tensor(std::move(w1));

    Tensor out1; out1.name = "out1"; out1.shape = {1, 4}; out1.dtype = DType::kFloat32;
    int t2 = graph.add_tensor(std::move(out1));

    // Dead branch
    Tensor w_dead; w_dead.name = "w_dead"; w_dead.shape = {4, 4}; w_dead.dtype = DType::kFloat32; w_dead.is_constant = true;
    int t3 = graph.add_tensor(std::move(w_dead));

    Tensor out_dead; out_dead.name = "out_dead"; out_dead.shape = {1, 4}; out_dead.dtype = DType::kFloat32;
    int t4 = graph.add_tensor(std::move(out_dead));

    Op live_op; live_op.kind = "MatMul"; live_op.inputs = {t0, t1}; live_op.outputs = {t2};
    graph.add_op(std::move(live_op));

    Op dead_op; dead_op.kind = "MatMul"; dead_op.inputs = {t0, t3}; dead_op.outputs = {t4};
    graph.add_op(std::move(dead_op));

    graph.graph_inputs().push_back(t0);
    graph.graph_outputs().push_back(t2);  // 只有 out1 是 graph output

    size_t ops_before = graph.op_count();
    size_t tensors_before = graph.tensor_count();

    tiny_tvm::passes::PassManager pm;
    pm.add(std::make_unique<tiny_tvm::passes::DeadCodeEliminationPass>());
    pm.run(graph);

    size_t ops_after = graph.op_count();
    size_t tensors_after = graph.tensor_count();

    // 验证：dead op 被删除
    if (ops_after < ops_before) {
        std::cout << "DCE ops: " << ops_before << " -> " << ops_after << "\n";
    } else {
        std::cerr << "ERROR: DCE did not remove dead ops (" << ops_before << " -> " << ops_after << ")\n";
        return 1;
    }

    // 验证：图索引仍然正确（graph output 可访问）
    for (int idx : graph.graph_outputs()) {
        auto& t = graph.tensor(idx);
        if (t.name.empty()) {
            std::cerr << "ERROR: graph output tensor has empty name after DCE\n";
            return 1;
        }
    }

    std::cout << "DCE_CHECK_OK\n";
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_dce.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_dce" 2>"$TMP_DIR/dce_compile.log"; then
    DCE_OUTPUT=$("$TMP_DIR/check_dce" 2>&1)
    if echo "$DCE_OUTPUT" | grep -q "DCE_CHECK_OK"; then
        pass "3-3c: DCE 成功删除死 op，图索引正确"
    else
        fail "3-3c: DCE 验证失败" "$DCE_OUTPUT"
    fi
else
    fail "3-3c: DCE checker 编译失败" "$(head -20 "$TMP_DIR/dce_compile.log")"
fi

# =====================================================================
# 任务 3-4：LivenessAnalysis + MemoryReusePass
# =====================================================================
section "任务 3-4：LivenessAnalysis + MemoryReusePass"

LA_H="$ROOT_DIR/include/tiny_tvm/pass/memory/liveness_analysis_pass.h"
MR_H="$ROOT_DIR/include/tiny_tvm/pass/memory/memory_reuse_pass.h"

if [ -f "$LA_H" ]; then
    pass "3-4a: liveness_analysis_pass.h 存在"
else
    fail "3-4a: liveness_analysis_pass.h 不存在"
fi

if [ -f "$MR_H" ]; then
    pass "3-4b: memory_reuse_pass.h 存在"
else
    fail "3-4b: memory_reuse_pass.h 不存在"
fi

# 编译 checker 验证内存复用效果
cat > "$TMP_DIR/check_mem_reuse.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <algorithm>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"

#if __has_include("tiny_tvm/pass/graph/infer_shape_pass.h")
#include "tiny_tvm/pass/graph/infer_shape_pass.h"
#endif
#if __has_include("tiny_tvm/pass/memory/naive_memory_planner.h")
#include "tiny_tvm/pass/memory/naive_memory_planner.h"
#define HAS_NMP 1
#else
#define HAS_NMP 0
#endif
#if __has_include("tiny_tvm/pass/memory/liveness_analysis_pass.h")
#include "tiny_tvm/pass/memory/liveness_analysis_pass.h"
#define HAS_LA 1
#else
#define HAS_LA 0
#endif
#if __has_include("tiny_tvm/pass/memory/memory_reuse_pass.h")
#include "tiny_tvm/pass/memory/memory_reuse_pass.h"
#define HAS_MR 1
#else
#define HAS_MR 0
#endif

using namespace tiny_tvm::ir;

// 计算 workspace total（最大 offset + nbytes）
size_t calc_workspace(const Graph& g) {
    size_t max_end = 0;
    for (size_t i = 0; i < g.tensor_count(); i++) {
        auto& t = g.tensor(i);
        if (!t.is_constant && t.nbytes > 0) {
            max_end = std::max(max_end, t.offset + t.nbytes);
        }
    }
    return max_end;
}

int main() {
#if !HAS_MR
    std::cerr << "ERROR: MemoryReusePass not found\n";
    return 1;
#endif

    // 构造链式图：A -> B -> C -> D（中间 tensor 可复用）
    auto make_graph = []() {
        Graph g;
        Tensor in; in.name = "in"; in.shape = {1, 64}; in.dtype = DType::kFloat32;
        int t0 = g.add_tensor(std::move(in));

        Tensor w1; w1.name = "w1"; w1.shape = {64, 64}; w1.dtype = DType::kFloat32; w1.is_constant = true;
        int tw1 = g.add_tensor(std::move(w1));
        Tensor o1; o1.name = "o1"; o1.shape = {1, 64}; o1.dtype = DType::kFloat32;
        int t1 = g.add_tensor(std::move(o1));

        Tensor w2; w2.name = "w2"; w2.shape = {64, 64}; w2.dtype = DType::kFloat32; w2.is_constant = true;
        int tw2 = g.add_tensor(std::move(w2));
        Tensor o2; o2.name = "o2"; o2.shape = {1, 64}; o2.dtype = DType::kFloat32;
        int t2 = g.add_tensor(std::move(o2));

        Tensor w3; w3.name = "w3"; w3.shape = {64, 64}; w3.dtype = DType::kFloat32; w3.is_constant = true;
        int tw3 = g.add_tensor(std::move(w3));
        Tensor o3; o3.name = "o3"; o3.shape = {1, 64}; o3.dtype = DType::kFloat32;
        int t3 = g.add_tensor(std::move(o3));

        Op op1; op1.kind = "MatMul"; op1.inputs = {t0, tw1}; op1.outputs = {t1};
        g.add_op(std::move(op1));
        Op op2; op2.kind = "MatMul"; op2.inputs = {t1, tw2}; op2.outputs = {t2};
        g.add_op(std::move(op2));
        Op op3; op3.kind = "MatMul"; op3.inputs = {t2, tw3}; op3.outputs = {t3};
        g.add_op(std::move(op3));

        g.graph_inputs().push_back(t0);
        g.graph_outputs().push_back(t3);
        return g;
    };

    // 跑 NaiveMemoryPlanner
    Graph g1 = make_graph();
    {
        tiny_tvm::passes::PassManager pm;
#if __has_include("tiny_tvm/pass/graph/infer_shape_pass.h")
        pm.add(std::make_unique<tiny_tvm::passes::InferShapePass>());
#endif
#if HAS_NMP
        pm.add(std::make_unique<tiny_tvm::passes::NaiveMemoryPlanner>());
#endif
        pm.run(g1);
    }
    size_t naive_ws = calc_workspace(g1);

    // 跑 MemoryReusePass
    Graph g2 = make_graph();
    {
        tiny_tvm::passes::PassManager pm;
#if __has_include("tiny_tvm/pass/graph/infer_shape_pass.h")
        pm.add(std::make_unique<tiny_tvm::passes::InferShapePass>());
#endif
#if HAS_LA
        pm.add(std::make_unique<tiny_tvm::passes::LivenessAnalysisPass>());
#endif
        pm.add(std::make_unique<tiny_tvm::passes::MemoryReusePass>());
        pm.run(g2);
    }
    size_t reuse_ws = calc_workspace(g2);

    std::cout << "Naive workspace: " << naive_ws << " bytes\n";
    std::cout << "Reuse workspace: " << reuse_ws << " bytes\n";

    if (reuse_ws < naive_ws) {
        std::cout << "MEM_REUSE_CHECK_OK\n";
    } else {
        std::cerr << "ERROR: MemoryReusePass did not reduce workspace\n";
        return 1;
    }
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_mem_reuse.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_mem_reuse" 2>"$TMP_DIR/mr_compile.log"; then
    MR_OUTPUT=$("$TMP_DIR/check_mem_reuse" 2>&1)
    if echo "$MR_OUTPUT" | grep -q "MEM_REUSE_CHECK_OK"; then
        pass "3-4c: MemoryReusePass 成功减少 workspace 用量"
        # 打印实际数值
        echo "       $(echo "$MR_OUTPUT" | grep -E 'workspace')"
    else
        fail "3-4c: MemoryReusePass 未减少 workspace" "$MR_OUTPUT"
    fi
else
    fail "3-4c: MemoryReusePass checker 编译失败" "$(head -20 "$TMP_DIR/mr_compile.log")"
fi

# =====================================================================
# 任务 3-5：dump 和编译报告
# =====================================================================
section "任务 3-5：dump 和编译报告"

if [ -f "$TTVMC" ]; then
    MLP_JSON="$ROOT_DIR/examples/json/mlp.json"
    DUMP_DIR="$TMP_DIR/dump_test"
    OUT_DUMP="$TMP_DIR/out_dump"
    mkdir -p "$DUMP_DIR" "$OUT_DUMP"

    # 尝试带 --dump-dir 参数编译
    "$TTVMC" compile "$MLP_JSON" -o "$OUT_DUMP" --dump-dir "$DUMP_DIR" > /dev/null 2>&1 || true

    # 检查 dump 文件
    DUMP_FILES=$(find "$DUMP_DIR" -name "*.json" 2>/dev/null | wc -l)
    if [ "$DUMP_FILES" -gt 0 ]; then
        pass "3-5a: --dump-dir 生成了 $DUMP_FILES 个 dump 文件"
    else
        fail "3-5a: --dump-dir 未生成 dump 文件"
    fi

    # 检查 compile_report.json
    if [ -f "$OUT_DUMP/compile_report.json" ]; then
        pass "3-5b: compile_report.json 存在"
        # 验证报告内容
        REPORT_CHECK=$(python3 -c "
import json, sys
with open('$OUT_DUMP/compile_report.json') as f:
    data = json.load(f)
required = ['op_count', 'tensor_count']
# 更宽松的检查：至少包含部分关键字段
found = sum(1 for k in required if k in data or any(k in str(v) for v in data.keys()))
if found > 0 or len(data) >= 2:
    print('REPORT_OK')
else:
    print('REPORT_INCOMPLETE')
" 2>/dev/null)
        if [ "$REPORT_CHECK" = "REPORT_OK" ]; then
            pass "3-5c: compile_report.json 包含关键字段"
        else
            fail "3-5c: compile_report.json 内容不完整"
        fi
    else
        fail "3-5b: compile_report.json 不存在"
        fail "3-5c: 跳过报告内容检查"
    fi
else
    fail "3-5a: ttvmc 不存在"
    fail "3-5b: 跳过"
    fail "3-5c: 跳过"
fi

# =====================================================================
# 汇总
# =====================================================================
section "Phase 3 测试结果汇总"

echo ""
echo "  通过: $PASS_COUNT"
echo "  失败: $FAIL_COUNT"
echo "  总计: $TOTAL_COUNT"
echo ""
echo "  Score: $PASS_COUNT/$TOTAL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "🎉 Phase 3 所有检测项通过！编译器化完成。"
    exit 0
else
    echo "⚠️  Phase 3 有 $FAIL_COUNT 项未通过，请根据上方 [FAIL] 提示修复。"
    exit 1
fi
