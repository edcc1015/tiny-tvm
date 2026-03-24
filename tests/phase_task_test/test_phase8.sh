#!/usr/bin/env bash
# =============================================================================
# Phase 8 测试脚本：TensorIR / LoopIR
# =============================================================================
# 检测要点：
#   8-1: LoopIR 数据结构（LoopVar, Block, LoopProgram）
#   8-2: LowerToLoopIRPass（Graph → LoopIR 降级）
#   8-3: Schedule 原语（split, reorder, fuse_loops）
#   8-4: LoopIR Codegen（LoopProgram → C 代码）
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

INCLUDE_DIRS="-I$ROOT_DIR/include"
[ -d "$ROOT_DIR/third_party" ] && INCLUDE_DIRS="$INCLUDE_DIRS -I$ROOT_DIR/third_party"
CORE_SOURCES=$(find "$ROOT_DIR/src" -name "*.cpp" ! -path "*/tools/*" 2>/dev/null | tr '\n' ' ')

# =====================================================================
# 任务 8-1：LoopIR 数据结构
# =====================================================================
section "任务 8-1：LoopIR 数据结构（LoopVar / Block / LoopProgram）"

LOOP_IR_H="$ROOT_DIR/include/tiny_tvm/ir/loop_ir.h"

if [ -f "$LOOP_IR_H" ]; then
    pass "8-1a: loop_ir.h 存在"
else
    fail "8-1a: loop_ir.h 不存在"
fi

# --- Check 1: LoopVar struct exists (2 pts) ---
cat > "$TMP_DIR/checker_loopvar.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <string>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

int main() {
#if !HAS_LOOP_IR
    std::cout << "CHECK_FAIL: loop_ir.h not found" << std::endl;
    return 0;
#else
    tiny_tvm::ir::LoopVar lv;
    lv.name = "i";
    lv.extent = 16;
    if (lv.name == "i" && lv.extent == 16) {
        std::cout << "CHECK_OK" << std::endl;
    } else {
        std::cout << "CHECK_FAIL: LoopVar fields incorrect" << std::endl;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_loopvar.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_loopvar" 2>"$TMP_DIR/loopvar_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_loopvar" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-1b: LoopVar struct 存在且具备 name/extent 字段 (2 pts)"
    else
        fail "8-1b: LoopVar struct 检查未通过" "$OUTPUT"
    fi
else
    fail "8-1b: LoopVar checker 编译失败 (2 pts)" "$(head -20 "$TMP_DIR/loopvar_compile.log")"
fi

# --- Check 2: Block struct exists (3 pts) ---
cat > "$TMP_DIR/checker_block.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <string>
#include <vector>
#include <functional>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

int main() {
#if !HAS_LOOP_IR
    std::cout << "CHECK_FAIL: loop_ir.h not found" << std::endl;
    return 0;
#else
    tiny_tvm::ir::Block block;

    // Check loop_vars field
    block.loop_vars.push_back({"i", 4});
    block.loop_vars.push_back({"j", 8});

    // Check loop_order field
    block.loop_order = {0, 1};

    // Check compute_body field (should be a string or callable description)
    block.compute_body = "C[i][j] += A[i][k] * B[k][j]";

    // Check tiles field
    block.tiles = {2, 4};

    // Check unroll_innermost field
    block.unroll_innermost = false;

    bool ok = true;
    if (block.loop_vars.size() != 2) ok = false;
    if (block.loop_order.size() != 2) ok = false;
    if (block.compute_body.empty()) ok = false;

    if (ok) {
        std::cout << "CHECK_OK" << std::endl;
    } else {
        std::cout << "CHECK_FAIL: Block struct missing expected fields" << std::endl;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_block.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_block" 2>"$TMP_DIR/block_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_block" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-1c: Block struct 存在且具备 loop_vars/loop_order/compute_body/tiles/unroll_innermost 字段 (3 pts)"
    else
        fail "8-1c: Block struct 检查未通过" "$OUTPUT"
    fi
else
    fail "8-1c: Block checker 编译失败 (3 pts)" "$(head -20 "$TMP_DIR/block_compile.log")"
fi

# --- Check 3: LoopProgram struct exists (2 pts) ---
cat > "$TMP_DIR/checker_loopprogram.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <vector>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

#include "tiny_tvm/ir/graph.h"

int main() {
#if !HAS_LOOP_IR
    std::cout << "CHECK_FAIL: loop_ir.h not found" << std::endl;
    return 0;
#else
    tiny_tvm::ir::LoopProgram prog;

    // Check blocks vector
    tiny_tvm::ir::Block b;
    b.loop_vars.push_back({"i", 4});
    b.loop_order = {0};
    b.compute_body = "body";
    prog.blocks.push_back(b);

    // Check source_graph pointer
    tiny_tvm::ir::Graph g;
    prog.source_graph = &g;

    if (prog.blocks.size() == 1 && prog.source_graph != nullptr) {
        std::cout << "CHECK_OK" << std::endl;
    } else {
        std::cout << "CHECK_FAIL: LoopProgram missing blocks or source_graph" << std::endl;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_loopprogram.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_loopprogram" 2>"$TMP_DIR/loopprogram_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_loopprogram" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-1d: LoopProgram struct 存在且具备 blocks/source_graph 字段 (2 pts)"
    else
        fail "8-1d: LoopProgram struct 检查未通过" "$OUTPUT"
    fi
else
    fail "8-1d: LoopProgram checker 编译失败 (2 pts)" "$(head -20 "$TMP_DIR/loopprogram_compile.log")"
fi

# =====================================================================
# 任务 8-2：LowerToLoopIRPass
# =====================================================================
section "任务 8-2：LowerToLoopIRPass（Graph → LoopIR）"

LOWER_H="$ROOT_DIR/include/tiny_tvm/pass/lower/lower_to_loop_ir_pass.h"

if [ -f "$LOWER_H" ]; then
    pass "8-2a: lower_to_loop_ir_pass.h 存在"
else
    fail "8-2a: lower_to_loop_ir_pass.h 不存在"
fi

# --- Check 4: LowerToLoopIRPass exists and has lower() (2 pts) ---
cat > "$TMP_DIR/checker_lower_pass.cpp" << 'CHECKER_EOF'
#include <iostream>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

#if __has_include("tiny_tvm/pass/lower/lower_to_loop_ir_pass.h")
#include "tiny_tvm/pass/lower/lower_to_loop_ir_pass.h"
#define HAS_LOWER 1
#else
#define HAS_LOWER 0
#endif

#include "tiny_tvm/ir/graph.h"

int main() {
#if !HAS_LOOP_IR || !HAS_LOWER
    std::cout << "CHECK_FAIL: loop_ir.h or lower_to_loop_ir_pass.h not found" << std::endl;
    return 0;
#else
    tiny_tvm::ir::Graph graph;
    tiny_tvm::passes::LowerToLoopIRPass pass;
    // Verify lower() method exists and is callable
    tiny_tvm::ir::LoopProgram prog = pass.lower(graph);
    std::cout << "CHECK_OK" << std::endl;
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_lower_pass.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_lower_pass" 2>"$TMP_DIR/lower_pass_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_lower_pass" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-2b: LowerToLoopIRPass 存在且具备 lower() 方法 (2 pts)"
    else
        fail "8-2b: LowerToLoopIRPass 检查未通过" "$OUTPUT"
    fi
else
    fail "8-2b: LowerToLoopIRPass checker 编译失败 (2 pts)" "$(head -20 "$TMP_DIR/lower_pass_compile.log")"
fi

# --- Check 5: MatMul lowering (3 pts) ---
cat > "$TMP_DIR/checker_matmul_lower.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <vector>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

#if __has_include("tiny_tvm/pass/lower/lower_to_loop_ir_pass.h")
#include "tiny_tvm/pass/lower/lower_to_loop_ir_pass.h"
#define HAS_LOWER 1
#else
#define HAS_LOWER 0
#endif

#include "tiny_tvm/ir/graph.h"

using namespace tiny_tvm::ir;

int main() {
#if !HAS_LOOP_IR || !HAS_LOWER
    std::cout << "CHECK_FAIL: required headers not found" << std::endl;
    return 0;
#else
    // Build MatMul graph: [4,8] x [8,16] -> [4,16]
    Graph graph;

    Tensor a; a.name = "A"; a.shape = {4, 8}; a.dtype = DType::kFloat32;
    a.nbytes = 4 * 8 * 4;
    int t0 = graph.add_tensor(std::move(a));

    Tensor b; b.name = "B"; b.shape = {8, 16}; b.dtype = DType::kFloat32;
    b.is_constant = true; b.nbytes = 8 * 16 * 4;
    int t1 = graph.add_tensor(std::move(b));

    Tensor c; c.name = "C"; c.shape = {4, 16}; c.dtype = DType::kFloat32;
    c.nbytes = 4 * 16 * 4;
    int t2 = graph.add_tensor(std::move(c));

    Op matmul;
    matmul.kind = "MatMul";
    matmul.inputs = {t0, t1};
    matmul.outputs = {t2};
    graph.add_op(std::move(matmul));
    graph.graph_inputs().push_back(t0);
    graph.graph_outputs().push_back(t2);

    tiny_tvm::passes::LowerToLoopIRPass pass;
    LoopProgram prog = pass.lower(graph);

    if (prog.blocks.empty()) {
        std::cout << "CHECK_FAIL: no blocks produced from lowering" << std::endl;
        return 0;
    }

    const Block& blk = prog.blocks[0];

    // MatMul [4,8]x[8,16] should produce 3 loop vars: i(4), j(16), k(8)
    if (blk.loop_vars.size() != 3) {
        std::cout << "CHECK_FAIL: expected 3 loop_vars, got " << blk.loop_vars.size() << std::endl;
        return 0;
    }

    // Check extents (order: i=4, j=16, k=8)
    std::vector<int64_t> expected_extents = {4, 16, 8};
    bool extents_ok = true;
    for (size_t idx = 0; idx < 3; idx++) {
        if (blk.loop_vars[idx].extent != expected_extents[idx]) {
            extents_ok = false;
            std::cout << "CHECK_FAIL: loop_var[" << idx << "] extent="
                      << blk.loop_vars[idx].extent
                      << " expected=" << expected_extents[idx] << std::endl;
        }
    }

    // Check loop_order is {0,1,2}
    bool order_ok = (blk.loop_order.size() == 3 &&
                     blk.loop_order[0] == 0 &&
                     blk.loop_order[1] == 1 &&
                     blk.loop_order[2] == 2);

    if (extents_ok && order_ok) {
        std::cout << "CHECK_OK" << std::endl;
    } else if (!order_ok) {
        std::cout << "CHECK_FAIL: loop_order not {0,1,2}" << std::endl;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_matmul_lower.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_matmul_lower" 2>"$TMP_DIR/matmul_lower_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_matmul_lower" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-2c: MatMul [4,8]×[8,16] 降级产生 3 个 loop_vars (extents 4,16,8), loop_order={0,1,2} (3 pts)"
    else
        fail "8-2c: MatMul 降级检查未通过 (3 pts)" "$OUTPUT"
    fi
else
    fail "8-2c: MatMul lowering checker 编译失败 (3 pts)" "$(head -20 "$TMP_DIR/matmul_lower_compile.log")"
fi

# =====================================================================
# 任务 8-3：Schedule 原语（split / reorder / fuse_loops）
# =====================================================================
section "任务 8-3：Schedule 原语（split / reorder / fuse_loops）"

SCHED_PRIM_H="$ROOT_DIR/include/tiny_tvm/pass/schedule/schedule_primitives.h"

if [ -f "$SCHED_PRIM_H" ]; then
    pass "8-3a: schedule_primitives.h 存在"
else
    fail "8-3a: schedule_primitives.h 不存在"
fi

# --- Check 6: split primitive (3 pts) ---
cat > "$TMP_DIR/checker_split.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <vector>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

#if __has_include("tiny_tvm/pass/lower/lower_to_loop_ir_pass.h")
#include "tiny_tvm/pass/lower/lower_to_loop_ir_pass.h"
#define HAS_LOWER 1
#else
#define HAS_LOWER 0
#endif

#if __has_include("tiny_tvm/pass/schedule/schedule_primitives.h")
#include "tiny_tvm/pass/schedule/schedule_primitives.h"
#define HAS_SCHED_PRIM 1
#else
#define HAS_SCHED_PRIM 0
#endif

#include "tiny_tvm/ir/graph.h"

using namespace tiny_tvm::ir;

int main() {
#if !HAS_LOOP_IR || !HAS_LOWER || !HAS_SCHED_PRIM
    std::cout << "CHECK_FAIL: required headers not found" << std::endl;
    return 0;
#else
    // Build MatMul graph: [4,8] x [8,16] -> [4,16]
    Graph graph;
    Tensor a; a.name = "A"; a.shape = {4, 8}; a.dtype = DType::kFloat32; a.nbytes = 4*8*4;
    int t0 = graph.add_tensor(std::move(a));
    Tensor b; b.name = "B"; b.shape = {8, 16}; b.dtype = DType::kFloat32;
    b.is_constant = true; b.nbytes = 8*16*4;
    int t1 = graph.add_tensor(std::move(b));
    Tensor c; c.name = "C"; c.shape = {4, 16}; c.dtype = DType::kFloat32; c.nbytes = 4*16*4;
    int t2 = graph.add_tensor(std::move(c));
    Op matmul; matmul.kind = "MatMul"; matmul.inputs = {t0, t1}; matmul.outputs = {t2};
    graph.add_op(std::move(matmul));
    graph.graph_inputs().push_back(t0);
    graph.graph_outputs().push_back(t2);

    tiny_tvm::passes::LowerToLoopIRPass lower;
    LoopProgram prog = lower.lower(graph);

    if (prog.blocks.empty()) {
        std::cout << "CHECK_FAIL: no blocks after lowering" << std::endl;
        return 0;
    }

    Block& blk = prog.blocks[0];
    size_t orig_vars = blk.loop_vars.size();   // should be 3
    size_t orig_order = blk.loop_order.size();  // should be 3

    // split axis 0 (i, extent=4) with factor 2 → i_outer(2), i_inner(2)
    tiny_tvm::passes::split(blk, 0, 2);

    size_t new_vars = blk.loop_vars.size();
    size_t new_order = blk.loop_order.size();

    // After split: loop_vars should grow (original var replaced by two new vars)
    // Expect new_vars == orig_vars + 1 (one var split into two)
    if (new_vars > orig_vars && new_order > orig_order) {
        std::cout << "CHECK_OK" << std::endl;
    } else {
        std::cout << "CHECK_FAIL: split did not increase loop_vars (was "
                  << orig_vars << ", now " << new_vars << ") or loop_order (was "
                  << orig_order << ", now " << new_order << ")" << std::endl;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_split.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_split" 2>"$TMP_DIR/split_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_split" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-3b: split(block, 0, 2) 增加了 loop_vars 和 loop_order 的长度 (3 pts)"
    else
        fail "8-3b: split 原语检查未通过 (3 pts)" "$OUTPUT"
    fi
else
    fail "8-3b: split checker 编译失败 (3 pts)" "$(head -20 "$TMP_DIR/split_compile.log")"
fi

# --- Check 7: reorder primitive (2 pts) ---
cat > "$TMP_DIR/checker_reorder.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <vector>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

#if __has_include("tiny_tvm/pass/lower/lower_to_loop_ir_pass.h")
#include "tiny_tvm/pass/lower/lower_to_loop_ir_pass.h"
#define HAS_LOWER 1
#else
#define HAS_LOWER 0
#endif

#if __has_include("tiny_tvm/pass/schedule/schedule_primitives.h")
#include "tiny_tvm/pass/schedule/schedule_primitives.h"
#define HAS_SCHED_PRIM 1
#else
#define HAS_SCHED_PRIM 0
#endif

#include "tiny_tvm/ir/graph.h"

using namespace tiny_tvm::ir;

int main() {
#if !HAS_LOOP_IR || !HAS_LOWER || !HAS_SCHED_PRIM
    std::cout << "CHECK_FAIL: required headers not found" << std::endl;
    return 0;
#else
    // Build MatMul graph: [4,8] x [8,16] -> [4,16]
    Graph graph;
    Tensor a; a.name = "A"; a.shape = {4, 8}; a.dtype = DType::kFloat32; a.nbytes = 4*8*4;
    int t0 = graph.add_tensor(std::move(a));
    Tensor b; b.name = "B"; b.shape = {8, 16}; b.dtype = DType::kFloat32;
    b.is_constant = true; b.nbytes = 8*16*4;
    int t1 = graph.add_tensor(std::move(b));
    Tensor c; c.name = "C"; c.shape = {4, 16}; c.dtype = DType::kFloat32; c.nbytes = 4*16*4;
    int t2 = graph.add_tensor(std::move(c));
    Op matmul; matmul.kind = "MatMul"; matmul.inputs = {t0, t1}; matmul.outputs = {t2};
    graph.add_op(std::move(matmul));
    graph.graph_inputs().push_back(t0);
    graph.graph_outputs().push_back(t2);

    tiny_tvm::passes::LowerToLoopIRPass lower;
    LoopProgram prog = lower.lower(graph);

    if (prog.blocks.empty()) {
        std::cout << "CHECK_FAIL: no blocks after lowering" << std::endl;
        return 0;
    }

    Block& blk = prog.blocks[0];
    // Original order: {0, 1, 2}
    std::vector<int> orig_order = blk.loop_order;

    // Reorder to {2, 0, 1} (k, i, j)
    std::vector<int> new_order = {2, 0, 1};
    tiny_tvm::passes::reorder(blk, new_order);

    if (blk.loop_order == new_order) {
        std::cout << "CHECK_OK" << std::endl;
    } else {
        std::cout << "CHECK_FAIL: reorder did not update loop_order to {2,0,1}, got {";
        for (size_t i = 0; i < blk.loop_order.size(); i++) {
            if (i) std::cout << ",";
            std::cout << blk.loop_order[i];
        }
        std::cout << "}" << std::endl;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_reorder.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_reorder" 2>"$TMP_DIR/reorder_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_reorder" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-3c: reorder(block, {2,0,1}) 正确更新 loop_order (2 pts)"
    else
        fail "8-3c: reorder 原语检查未通过 (2 pts)" "$OUTPUT"
    fi
else
    fail "8-3c: reorder checker 编译失败 (2 pts)" "$(head -20 "$TMP_DIR/reorder_compile.log")"
fi

# --- Check 8: fuse_loops primitive (2 pts) ---
cat > "$TMP_DIR/checker_fuse.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <vector>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

#if __has_include("tiny_tvm/pass/lower/lower_to_loop_ir_pass.h")
#include "tiny_tvm/pass/lower/lower_to_loop_ir_pass.h"
#define HAS_LOWER 1
#else
#define HAS_LOWER 0
#endif

#if __has_include("tiny_tvm/pass/schedule/schedule_primitives.h")
#include "tiny_tvm/pass/schedule/schedule_primitives.h"
#define HAS_SCHED_PRIM 1
#else
#define HAS_SCHED_PRIM 0
#endif

#include "tiny_tvm/ir/graph.h"

using namespace tiny_tvm::ir;

int main() {
#if !HAS_LOOP_IR || !HAS_LOWER || !HAS_SCHED_PRIM
    std::cout << "CHECK_FAIL: required headers not found" << std::endl;
    return 0;
#else
    // Build MatMul graph: [4,8] x [8,16] -> [4,16]
    Graph graph;
    Tensor a; a.name = "A"; a.shape = {4, 8}; a.dtype = DType::kFloat32; a.nbytes = 4*8*4;
    int t0 = graph.add_tensor(std::move(a));
    Tensor b; b.name = "B"; b.shape = {8, 16}; b.dtype = DType::kFloat32;
    b.is_constant = true; b.nbytes = 8*16*4;
    int t1 = graph.add_tensor(std::move(b));
    Tensor c; c.name = "C"; c.shape = {4, 16}; c.dtype = DType::kFloat32; c.nbytes = 4*16*4;
    int t2 = graph.add_tensor(std::move(c));
    Op matmul; matmul.kind = "MatMul"; matmul.inputs = {t0, t1}; matmul.outputs = {t2};
    graph.add_op(std::move(matmul));
    graph.graph_inputs().push_back(t0);
    graph.graph_outputs().push_back(t2);

    tiny_tvm::passes::LowerToLoopIRPass lower;
    LoopProgram prog = lower.lower(graph);

    if (prog.blocks.empty()) {
        std::cout << "CHECK_FAIL: no blocks after lowering" << std::endl;
        return 0;
    }

    Block& blk = prog.blocks[0];
    size_t orig_vars = blk.loop_vars.size();  // should be 3

    // Fuse loop_vars 0 and 1 (i: extent=4 and j: extent=16)
    // Expected: merged extent = 4 * 16 = 64, loop_vars.size decreases by 1
    int64_t e0 = blk.loop_vars[0].extent;
    int64_t e1 = blk.loop_vars[1].extent;
    int64_t expected_fused = e0 * e1;

    tiny_tvm::passes::fuse_loops(blk, 0, 1);

    size_t new_vars = blk.loop_vars.size();

    // After fuse: one fewer loop var
    if (new_vars == orig_vars - 1) {
        // Check the fused var has correct extent
        int64_t fused_extent = blk.loop_vars[0].extent;
        if (fused_extent == expected_fused) {
            std::cout << "CHECK_OK" << std::endl;
        } else {
            std::cout << "CHECK_FAIL: fused extent=" << fused_extent
                      << " expected=" << expected_fused << std::endl;
        }
    } else {
        std::cout << "CHECK_FAIL: loop_vars.size after fuse=" << new_vars
                  << " expected=" << (orig_vars - 1) << std::endl;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_fuse.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_fuse" 2>"$TMP_DIR/fuse_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_fuse" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-3d: fuse_loops(block, 0, 1) 合并两个循环变量 (extent=v1*v2) (2 pts)"
    else
        fail "8-3d: fuse_loops 原语检查未通过 (2 pts)" "$OUTPUT"
    fi
else
    fail "8-3d: fuse_loops checker 编译失败 (2 pts)" "$(head -20 "$TMP_DIR/fuse_compile.log")"
fi

# =====================================================================
# 任务 8-4：LoopIR Codegen
# =====================================================================
section "任务 8-4：LoopIR Codegen（LoopProgram → C 代码）"

CODEGEN_H="$ROOT_DIR/include/tiny_tvm/codegen/loop_ir_codegen.h"

if [ -f "$CODEGEN_H" ]; then
    pass "8-4a: loop_ir_codegen.h 存在"
else
    fail "8-4a: loop_ir_codegen.h 不存在"
fi

# --- Check 9: emit_c_from_loop_program function exists (2 pts) ---
cat > "$TMP_DIR/checker_codegen_exists.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <string>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

#if __has_include("tiny_tvm/codegen/loop_ir_codegen.h")
#include "tiny_tvm/codegen/loop_ir_codegen.h"
#define HAS_CODEGEN 1
#else
#define HAS_CODEGEN 0
#endif

int main() {
#if !HAS_LOOP_IR || !HAS_CODEGEN
    std::cout << "CHECK_FAIL: loop_ir.h or loop_ir_codegen.h not found" << std::endl;
    return 0;
#else
    tiny_tvm::ir::LoopProgram prog;
    // Check that emit_c_from_loop_program is callable
    std::string code = tiny_tvm::codegen::emit_c_from_loop_program(prog);
    std::cout << "CHECK_OK" << std::endl;
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_codegen_exists.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_codegen_exists" 2>"$TMP_DIR/codegen_exists_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_codegen_exists" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-4b: emit_c_from_loop_program 函数存在且可调用 (2 pts)"
    else
        fail "8-4b: emit_c_from_loop_program 检查未通过" "$OUTPUT"
    fi
else
    fail "8-4b: emit_c_from_loop_program checker 编译失败 (2 pts)" "$(head -20 "$TMP_DIR/codegen_exists_compile.log")"
fi

# --- Check 10: Codegen generates valid C with for loops (2 pts) ---
cat > "$TMP_DIR/checker_codegen_for.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <string>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

#if __has_include("tiny_tvm/pass/lower/lower_to_loop_ir_pass.h")
#include "tiny_tvm/pass/lower/lower_to_loop_ir_pass.h"
#define HAS_LOWER 1
#else
#define HAS_LOWER 0
#endif

#if __has_include("tiny_tvm/codegen/loop_ir_codegen.h")
#include "tiny_tvm/codegen/loop_ir_codegen.h"
#define HAS_CODEGEN 1
#else
#define HAS_CODEGEN 0
#endif

#include "tiny_tvm/ir/graph.h"

using namespace tiny_tvm::ir;

int main() {
#if !HAS_LOOP_IR || !HAS_LOWER || !HAS_CODEGEN
    std::cout << "CHECK_FAIL: required headers not found" << std::endl;
    return 0;
#else
    // Build MatMul graph
    Graph graph;
    Tensor a; a.name = "A"; a.shape = {4, 8}; a.dtype = DType::kFloat32; a.nbytes = 4*8*4;
    int t0 = graph.add_tensor(std::move(a));
    Tensor b; b.name = "B"; b.shape = {8, 16}; b.dtype = DType::kFloat32;
    b.is_constant = true; b.nbytes = 8*16*4;
    int t1 = graph.add_tensor(std::move(b));
    Tensor c; c.name = "C"; c.shape = {4, 16}; c.dtype = DType::kFloat32; c.nbytes = 4*16*4;
    int t2 = graph.add_tensor(std::move(c));
    Op matmul; matmul.kind = "MatMul"; matmul.inputs = {t0, t1}; matmul.outputs = {t2};
    graph.add_op(std::move(matmul));
    graph.graph_inputs().push_back(t0);
    graph.graph_outputs().push_back(t2);

    tiny_tvm::passes::LowerToLoopIRPass lower;
    LoopProgram prog = lower.lower(graph);
    prog.source_graph = &graph;

    std::string code = tiny_tvm::codegen::emit_c_from_loop_program(prog);

    // Check output contains "for" loops
    if (code.find("for") != std::string::npos) {
        std::cout << "CHECK_OK" << std::endl;
    } else {
        std::cout << "CHECK_FAIL: codegen output does not contain 'for' loops" << std::endl;
        std::cout << "Output was: " << code.substr(0, 200) << std::endl;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_codegen_for.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_codegen_for" 2>"$TMP_DIR/codegen_for_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_codegen_for" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-4c: LoopIR codegen 生成的代码包含 for 循环 (2 pts)"
    else
        fail "8-4c: LoopIR codegen for 循环检查未通过 (2 pts)" "$OUTPUT"
    fi
else
    fail "8-4c: LoopIR codegen for-loop checker 编译失败 (2 pts)" "$(head -20 "$TMP_DIR/codegen_for_compile.log")"
fi

# --- Check 11: Codegen after split produces different code (2 pts) ---
cat > "$TMP_DIR/checker_codegen_split_diff.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <string>

#if __has_include("tiny_tvm/ir/loop_ir.h")
#include "tiny_tvm/ir/loop_ir.h"
#define HAS_LOOP_IR 1
#else
#define HAS_LOOP_IR 0
#endif

#if __has_include("tiny_tvm/pass/lower/lower_to_loop_ir_pass.h")
#include "tiny_tvm/pass/lower/lower_to_loop_ir_pass.h"
#define HAS_LOWER 1
#else
#define HAS_LOWER 0
#endif

#if __has_include("tiny_tvm/pass/schedule/schedule_primitives.h")
#include "tiny_tvm/pass/schedule/schedule_primitives.h"
#define HAS_SCHED_PRIM 1
#else
#define HAS_SCHED_PRIM 0
#endif

#if __has_include("tiny_tvm/codegen/loop_ir_codegen.h")
#include "tiny_tvm/codegen/loop_ir_codegen.h"
#define HAS_CODEGEN 1
#else
#define HAS_CODEGEN 0
#endif

#include "tiny_tvm/ir/graph.h"

using namespace tiny_tvm::ir;

Graph make_matmul_graph() {
    Graph graph;
    Tensor a; a.name = "A"; a.shape = {4, 8}; a.dtype = DType::kFloat32; a.nbytes = 4*8*4;
    int t0 = graph.add_tensor(std::move(a));
    Tensor b; b.name = "B"; b.shape = {8, 16}; b.dtype = DType::kFloat32;
    b.is_constant = true; b.nbytes = 8*16*4;
    int t1 = graph.add_tensor(std::move(b));
    Tensor c; c.name = "C"; c.shape = {4, 16}; c.dtype = DType::kFloat32; c.nbytes = 4*16*4;
    int t2 = graph.add_tensor(std::move(c));
    Op matmul; matmul.kind = "MatMul"; matmul.inputs = {t0, t1}; matmul.outputs = {t2};
    graph.add_op(std::move(matmul));
    graph.graph_inputs().push_back(t0);
    graph.graph_outputs().push_back(t2);
    return graph;
}

int main() {
#if !HAS_LOOP_IR || !HAS_LOWER || !HAS_SCHED_PRIM || !HAS_CODEGEN
    std::cout << "CHECK_FAIL: required headers not found" << std::endl;
    return 0;
#else
    // Generate code BEFORE split
    Graph graph1 = make_matmul_graph();
    tiny_tvm::passes::LowerToLoopIRPass lower1;
    LoopProgram prog1 = lower1.lower(graph1);
    prog1.source_graph = &graph1;
    std::string code_before = tiny_tvm::codegen::emit_c_from_loop_program(prog1);

    // Generate code AFTER split
    Graph graph2 = make_matmul_graph();
    tiny_tvm::passes::LowerToLoopIRPass lower2;
    LoopProgram prog2 = lower2.lower(graph2);
    prog2.source_graph = &graph2;

    if (!prog2.blocks.empty()) {
        tiny_tvm::passes::split(prog2.blocks[0], 0, 2);
    }

    std::string code_after = tiny_tvm::codegen::emit_c_from_loop_program(prog2);

    if (code_before != code_after) {
        std::cout << "CHECK_OK" << std::endl;
    } else {
        std::cout << "CHECK_FAIL: codegen output is identical before and after split" << std::endl;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/checker_codegen_split_diff.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/checker_codegen_split_diff" 2>"$TMP_DIR/codegen_split_compile.log"; then
    OUTPUT=$("$TMP_DIR/checker_codegen_split_diff" 2>&1)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "8-4d: split 后 codegen 输出与 split 前不同 (2 pts)"
    else
        fail "8-4d: split 前后 codegen 输出相同 (2 pts)" "$OUTPUT"
    fi
else
    fail "8-4d: codegen split-diff checker 编译失败 (2 pts)" "$(head -20 "$TMP_DIR/codegen_split_compile.log")"
fi

# =====================================================================
# 汇总
# =====================================================================

echo ""
echo "========== Phase 8 Test Results =========="
echo "Score: $PASS_COUNT / $TOTAL_COUNT"
echo "Pass: $PASS_COUNT  Fail: $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "🎉 Phase 8 所有检测项通过！TensorIR / LoopIR 完成。"
    exit 0
else
    echo "⚠️  Phase 8 有 $FAIL_COUNT 项未通过，请根据上方 [FAIL] 提示修复。"
    exit 1
fi
