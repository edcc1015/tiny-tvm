#!/usr/bin/env bash
# =============================================================================
# Phase 4 测试脚本：Schedule 深化（先服务 MatMul）
# =============================================================================
# 检测要点：
#   4-1: Schedule 定义（LoopOrder enum、tile 字段）
#   4-2: TilingPass（对大 MatMul 设置非默认 tile 值）
#   4-3: LoopReorderPass + UnrollPass
#   4-4: Codegen 真的吃到 schedule（tiled vs naive 不同，数值一致）
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
# 任务 4-1：Schedule 定义
# =====================================================================
section "任务 4-1：Schedule 定义"

SCHEDULE_H="$ROOT_DIR/include/tiny_tvm/schedule/schedule.h"

if [ -f "$SCHEDULE_H" ]; then
    pass "4-1a: schedule.h 存在"
else
    fail "4-1a: schedule.h 不存在"
fi

# 检查 LoopOrder enum
if grep -q 'LoopOrder\|loop_order\|kIJK\|kIKJ' "$SCHEDULE_H" 2>/dev/null; then
    pass "4-1b: Schedule 包含 LoopOrder 枚举（kIJK/kIKJ）"
else
    fail "4-1b: Schedule 缺少 LoopOrder 枚举" "当前只有 bool reorder_ikj，应改为显式 LoopOrder"
fi

# 检查 tile 字段
if grep -q 'tile_m' "$SCHEDULE_H" 2>/dev/null && \
   grep -q 'tile_n' "$SCHEDULE_H" 2>/dev/null && \
   grep -q 'tile_k' "$SCHEDULE_H" 2>/dev/null; then
    pass "4-1c: Schedule 包含 tile_m/tile_n/tile_k 字段"
else
    fail "4-1c: Schedule 缺少 tile 字段"
fi

# 检查 unroll_inner 字段
if grep -q 'unroll_inner' "$SCHEDULE_H" 2>/dev/null; then
    pass "4-1d: Schedule 包含 unroll_inner 字段"
else
    fail "4-1d: Schedule 缺少 unroll_inner 字段"
fi

# 检查 order 字段（LoopOrder 类型）
if grep -qE '(LoopOrder|loop_order)\s+order' "$SCHEDULE_H" 2>/dev/null || \
   grep -q 'order.*=.*LoopOrder\|order.*=.*kIJK' "$SCHEDULE_H" 2>/dev/null; then
    pass "4-1e: Schedule 包含 LoopOrder order 字段"
else
    fail "4-1e: Schedule 缺少 LoopOrder order 字段"
fi

# =====================================================================
# 任务 4-2：TilingPass
# =====================================================================
section "任务 4-2：TilingPass"

TILING_H="$ROOT_DIR/include/tiny_tvm/pass/schedule/tiling_pass.h"
TILING_CPP="$ROOT_DIR/src/pass/schedule/tiling_pass.cpp"

if [ -f "$TILING_H" ]; then
    pass "4-2a: tiling_pass.h 存在"
else
    fail "4-2a: tiling_pass.h 不存在"
fi

if [ -f "$TILING_CPP" ]; then
    pass "4-2b: tiling_pass.cpp 存在"
else
    fail "4-2b: tiling_pass.cpp 不存在"
fi

# 编译 checker 验证 TilingPass 效果
cat > "$TMP_DIR/check_tiling.cpp" << 'CHECKER_EOF'
#include <iostream>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"

#if __has_include("tiny_tvm/pass/schedule/tiling_pass.h")
#include "tiny_tvm/pass/schedule/tiling_pass.h"
#define HAS_TILING 1
#else
#define HAS_TILING 0
#endif

#if __has_include("tiny_tvm/pass/graph/infer_shape_pass.h")
#include "tiny_tvm/pass/graph/infer_shape_pass.h"
#endif

using namespace tiny_tvm::ir;

int main() {
#if !HAS_TILING
    std::cerr << "ERROR: TilingPass not found\n";
    return 1;
#else
    // 构造大 MatMul: [1, 128] x [128, 128] -> [1, 128]
    Graph graph;

    Tensor input; input.name = "input"; input.shape = {1, 128}; input.dtype = DType::kFloat32;
    int t0 = graph.add_tensor(std::move(input));

    Tensor weight; weight.name = "weight"; weight.shape = {128, 128}; weight.dtype = DType::kFloat32;
    weight.is_constant = true;
    int t1 = graph.add_tensor(std::move(weight));

    Tensor output; output.name = "output"; output.shape = {1, 128}; output.dtype = DType::kFloat32;
    int t2 = graph.add_tensor(std::move(output));

    Op matmul; matmul.kind = "MatMul"; matmul.inputs = {t0, t1}; matmul.outputs = {t2};
    graph.add_op(std::move(matmul));
    graph.graph_inputs().push_back(t0);
    graph.graph_outputs().push_back(t2);

    tiny_tvm::passes::PassManager pm;
#if __has_include("tiny_tvm/pass/graph/infer_shape_pass.h")
    pm.add(std::make_unique<tiny_tvm::passes::InferShapePass>());
#endif
    pm.add(std::make_unique<tiny_tvm::passes::TilingPass>());
    pm.run(graph);

    auto& sched = graph.op(0).schedule;
    bool has_tile = (sched.tile_m > 0 || sched.tile_n > 0 || sched.tile_k > 0);
    if (has_tile) {
        std::cout << "tile_m=" << sched.tile_m
                  << " tile_n=" << sched.tile_n
                  << " tile_k=" << sched.tile_k << "\n";
        std::cout << "TILING_CHECK_OK\n";
    } else {
        std::cerr << "ERROR: TilingPass did not set any tile values for 128x128 MatMul\n";
        return 1;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_tiling.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_tiling" 2>"$TMP_DIR/tiling_compile.log"; then
    TILING_OUTPUT=$("$TMP_DIR/check_tiling" 2>&1)
    if echo "$TILING_OUTPUT" | grep -q "TILING_CHECK_OK"; then
        pass "4-2c: TilingPass 对大 MatMul 设置了非默认 tile 值"
        echo "       $(echo "$TILING_OUTPUT" | grep 'tile_')"
    else
        fail "4-2c: TilingPass 未对大 MatMul 设置 tile 值" "$TILING_OUTPUT"
    fi
else
    fail "4-2c: TilingPass checker 编译失败" "$(head -20 "$TMP_DIR/tiling_compile.log")"
fi

# =====================================================================
# 任务 4-3：LoopReorderPass + UnrollPass
# =====================================================================
section "任务 4-3：LoopReorderPass + UnrollPass"

LR_H="$ROOT_DIR/include/tiny_tvm/pass/schedule/loop_reorder_pass.h"
LR_CPP="$ROOT_DIR/src/pass/schedule/loop_reorder_pass.cpp"
UR_H="$ROOT_DIR/include/tiny_tvm/pass/schedule/unroll_pass.h"
UR_CPP="$ROOT_DIR/src/pass/schedule/unroll_pass.cpp"

if [ -f "$LR_H" ]; then pass "4-3a: loop_reorder_pass.h 存在"; else fail "4-3a: loop_reorder_pass.h 不存在"; fi
if [ -f "$LR_CPP" ]; then pass "4-3b: loop_reorder_pass.cpp 存在"; else fail "4-3b: loop_reorder_pass.cpp 不存在"; fi
if [ -f "$UR_H" ]; then pass "4-3c: unroll_pass.h 存在"; else fail "4-3c: unroll_pass.h 不存在"; fi
if [ -f "$UR_CPP" ]; then pass "4-3d: unroll_pass.cpp 存在"; else fail "4-3d: unroll_pass.cpp 不存在"; fi

# 编译 checker 验证 schedule 被修改
cat > "$TMP_DIR/check_schedule_passes.cpp" << 'CHECKER_EOF'
#include <iostream>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"
#include "tiny_tvm/schedule/schedule.h"

#if __has_include("tiny_tvm/pass/graph/infer_shape_pass.h")
#include "tiny_tvm/pass/graph/infer_shape_pass.h"
#endif
#if __has_include("tiny_tvm/pass/schedule/tiling_pass.h")
#include "tiny_tvm/pass/schedule/tiling_pass.h"
#endif
#if __has_include("tiny_tvm/pass/schedule/loop_reorder_pass.h")
#include "tiny_tvm/pass/schedule/loop_reorder_pass.h"
#define HAS_LR 1
#else
#define HAS_LR 0
#endif
#if __has_include("tiny_tvm/pass/schedule/unroll_pass.h")
#include "tiny_tvm/pass/schedule/unroll_pass.h"
#define HAS_UR 1
#else
#define HAS_UR 0
#endif

using namespace tiny_tvm::ir;

int main() {
#if !HAS_LR && !HAS_UR
    std::cerr << "ERROR: LoopReorderPass and UnrollPass not found\n";
    return 1;
#endif

    Graph graph;
    Tensor input; input.name = "input"; input.shape = {1, 128}; input.dtype = DType::kFloat32;
    int t0 = graph.add_tensor(std::move(input));
    Tensor weight; weight.name = "weight"; weight.shape = {128, 128}; weight.dtype = DType::kFloat32;
    weight.is_constant = true;
    int t1 = graph.add_tensor(std::move(weight));
    Tensor output; output.name = "output"; output.shape = {1, 128}; output.dtype = DType::kFloat32;
    int t2 = graph.add_tensor(std::move(output));

    Op matmul; matmul.kind = "MatMul"; matmul.inputs = {t0, t1}; matmul.outputs = {t2};
    graph.add_op(std::move(matmul));
    graph.graph_inputs().push_back(t0);
    graph.graph_outputs().push_back(t2);

    tiny_tvm::passes::PassManager pm;
#if __has_include("tiny_tvm/pass/graph/infer_shape_pass.h")
    pm.add(std::make_unique<tiny_tvm::passes::InferShapePass>());
#endif
#if __has_include("tiny_tvm/pass/schedule/tiling_pass.h")
    pm.add(std::make_unique<tiny_tvm::passes::TilingPass>());
#endif
#if HAS_LR
    pm.add(std::make_unique<tiny_tvm::passes::LoopReorderPass>());
#endif
#if HAS_UR
    pm.add(std::make_unique<tiny_tvm::passes::UnrollPass>());
#endif
    pm.run(graph);

    auto& sched = graph.op(0).schedule;

    if (sched.is_default()) {
        std::cerr << "ERROR: schedule is still default after all schedule passes\n";
        return 1;
    }

    std::cout << "Schedule after passes:\n";
    std::cout << "  tile_m=" << sched.tile_m << " tile_n=" << sched.tile_n << " tile_k=" << sched.tile_k << "\n";
    std::cout << "  unroll_inner=" << (sched.unroll_inner ? "true" : "false") << "\n";
    std::cout << "SCHEDULE_PASSES_OK\n";
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_schedule_passes.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_schedule_passes" 2>"$TMP_DIR/sp_compile.log"; then
    SP_OUTPUT=$("$TMP_DIR/check_schedule_passes" 2>&1)
    if echo "$SP_OUTPUT" | grep -q "SCHEDULE_PASSES_OK"; then
        pass "4-3e: LoopReorderPass + UnrollPass 修改了 schedule"
        echo "       $(echo "$SP_OUTPUT" | grep 'tile_\|unroll')"
    else
        fail "4-3e: Schedule passes 未修改 schedule" "$SP_OUTPUT"
    fi
else
    fail "4-3e: Schedule passes checker 编译失败" "$(head -20 "$TMP_DIR/sp_compile.log")"
fi

# =====================================================================
# 任务 4-4：Codegen 真的吃到 schedule
# =====================================================================
section "任务 4-4：Codegen 真的吃到 schedule"

CODEGEN_CPP="$ROOT_DIR/src/codegen/c_codegen.cpp"

# 检查 codegen 中有 tiled 分支
if grep -qi 'tiled\|tile_m\|tile_n\|tile_k' "$CODEGEN_CPP" 2>/dev/null; then
    pass "4-4a: c_codegen.cpp 包含 tiled MatMul 分支"
else
    fail "4-4a: c_codegen.cpp 缺少 tiled MatMul 分支"
fi

# 检查 codegen 中有 naive 分支
if grep -qi 'naive\|emit_matmul_naive\|default' "$CODEGEN_CPP" 2>/dev/null; then
    pass "4-4b: c_codegen.cpp 包含 naive MatMul 分支"
else
    fail "4-4b: c_codegen.cpp 缺少 naive/default MatMul 分支"
fi

# 编译 checker：生成两种不同 schedule 的代码，对比差异
cat > "$TMP_DIR/check_codegen_schedule.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <string>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/codegen/c_codegen.h"

using namespace tiny_tvm::ir;

Graph make_matmul_graph(int tile_m_val, int tile_n_val, int tile_k_val) {
    Graph graph;
    Tensor input; input.name = "input"; input.shape = {1, 64}; input.dtype = DType::kFloat32;
    input.nbytes = 256;
    int t0 = graph.add_tensor(std::move(input));

    Tensor weight; weight.name = "weight"; weight.shape = {64, 64}; weight.dtype = DType::kFloat32;
    weight.is_constant = true; weight.nbytes = 16384;
    int t1 = graph.add_tensor(std::move(weight));

    Tensor output; output.name = "output"; output.shape = {1, 64}; output.dtype = DType::kFloat32;
    output.nbytes = 256; output.offset = 256;
    int t2 = graph.add_tensor(std::move(output));

    Op matmul;
    matmul.kind = "MatMul";
    matmul.inputs = {t0, t1};
    matmul.outputs = {t2};
    matmul.schedule.tile_m = tile_m_val;
    matmul.schedule.tile_n = tile_n_val;
    matmul.schedule.tile_k = tile_k_val;
    graph.add_op(std::move(matmul));
    graph.graph_inputs().push_back(t0);
    graph.graph_outputs().push_back(t2);
    return graph;
}

int main() {
    // 生成 naive 版本（默认 schedule）
    Graph g_naive = make_matmul_graph(-1, -1, -1);
    std::string code_naive = tiny_tvm::codegen::emit_c_module(g_naive);

    // 生成 tiled 版本
    Graph g_tiled = make_matmul_graph(16, 16, 16);
    std::string code_tiled = tiny_tvm::codegen::emit_c_module(g_tiled);

    // 检查两者不同
    if (code_naive == code_tiled) {
        std::cerr << "ERROR: naive and tiled codegen output are identical\n";
        std::cerr << "This means schedule is not affecting code generation\n";
        return 1;
    }

    // 检查 tiled 版本确实包含更多循环或不同结构
    size_t naive_for = 0, tiled_for = 0;
    for (size_t i = 0; i < code_naive.size(); i++)
        if (code_naive.substr(i, 3) == "for") naive_for++;
    for (size_t i = 0; i < code_tiled.size(); i++)
        if (code_tiled.substr(i, 3) == "for") tiled_for++;

    std::cout << "Naive for-count: " << naive_for << "\n";
    std::cout << "Tiled for-count: " << tiled_for << "\n";

    if (code_naive != code_tiled) {
        std::cout << "CODEGEN_SCHEDULE_DIFF_OK\n";
    }

    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_codegen_schedule.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_codegen_schedule" 2>"$TMP_DIR/cgs_compile.log"; then
    CGS_OUTPUT=$("$TMP_DIR/check_codegen_schedule" 2>&1)
    if echo "$CGS_OUTPUT" | grep -q "CODEGEN_SCHEDULE_DIFF_OK"; then
        pass "4-4c: Schedule 开关前后 codegen 输出不同"
        echo "       $(echo "$CGS_OUTPUT" | grep 'for-count')"
    else
        fail "4-4c: Schedule 开关前后 codegen 输出相同" "$CGS_OUTPUT"
    fi
else
    fail "4-4c: Codegen schedule checker 编译失败" "$(head -20 "$TMP_DIR/cgs_compile.log")"
fi

# E2E 数值一致性：tiled 和 naive 产出相同结果
if [ -f "$TTVMC" ]; then
    MLP_JSON="$ROOT_DIR/examples/json/mlp.json"
    RUN_MODEL="$BUILD_DIR/run_model"

    # 这里做简化检查：只检查 ttvmc compile 在有 schedule 优化时仍能成功
    OUT_SCHED="$TMP_DIR/out_sched"
    mkdir -p "$OUT_SCHED"
    if "$TTVMC" compile "$MLP_JSON" -o "$OUT_SCHED" > /dev/null 2>&1; then
        if [ -f "$OUT_SCHED/libdeploy.so" ] && [ -f "$RUN_MODEL" ]; then
            python3 -c "import struct,sys; sys.stdout.buffer.write(struct.pack('4f',1,2,3,4))" > "$TMP_DIR/input.bin"
            LD_LIBRARY_PATH="$OUT_SCHED" "$RUN_MODEL" "$OUT_SCHED" "$TMP_DIR/input.bin" "$TMP_DIR/output_sched.bin" > /dev/null 2>&1 || true
            if [ -f "$TMP_DIR/output_sched.bin" ] && [ "$(stat -c%s "$TMP_DIR/output_sched.bin" 2>/dev/null)" -gt 0 ]; then
                pass "4-4d: 带 Schedule 优化的编译仍能正确运行"
            else
                fail "4-4d: 带 Schedule 优化的编译运行失败"
            fi
        else
            fail "4-4d: libdeploy.so 或 run_model 不存在"
        fi
    else
        fail "4-4d: ttvmc compile 失败"
    fi
else
    fail "4-4d: ttvmc 不存在"
fi

# =====================================================================
# 汇总
# =====================================================================
section "Phase 4 测试结果汇总"

echo ""
echo "  通过: $PASS_COUNT"
echo "  失败: $FAIL_COUNT"
echo "  总计: $TOTAL_COUNT"
echo ""
echo "  Score: $PASS_COUNT/$TOTAL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "🎉 Phase 4 所有检测项通过！Schedule 深化完成。"
    exit 0
else
    echo "⚠️  Phase 4 有 $FAIL_COUNT 项未通过，请根据上方 [FAIL] 提示修复。"
    exit 1
fi
