#!/usr/bin/env bash
# =============================================================================
# Phase 0 测试脚本：工程骨架完善
# =============================================================================
# 检测要点：
#   0-1: IR 定义完整性（Tensor 字段、Graph 接口、辅助函数）
#   0-2: Pass 基础设施（PassManager、NoOpPass、多 pass 连续执行）
#   0-3: CLI 骨架（ttvmc smoke/help/version）
#   0-4: 测试与构建入口（build_host.sh、graph_smoke_test）
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

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo "[PASS] $1"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo "[FAIL] $1"
    if [ -n "${2:-}" ]; then
        echo "       Reason: $2"
    fi
}

section() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "======================================================================"
}

# ---------------------------------------------------------------------------
# 先确保项目能编译
# ---------------------------------------------------------------------------
section "前置：编译项目"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found, building..."
    cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug > /dev/null 2>&1 || true
fi

cmake --build "$BUILD_DIR" --parallel > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "WARNING: 项目编译失败，部分测试可能无法执行"
fi

# =====================================================================
# 任务 0-1：IR 定义完整性
# =====================================================================
section "任务 0-1：IR 定义完整性"

GRAPH_H="$ROOT_DIR/include/tiny_tvm/ir/graph.h"

# --- 检查 Tensor 结构体字段 ---
# param_offset 字段
if grep -q 'param_offset' "$GRAPH_H" 2>/dev/null; then
    pass "0-1a: Tensor 包含 param_offset 字段"
else
    fail "0-1a: Tensor 缺少 param_offset 字段" "常量张量需要 param_offset 来记录在 params.bin 中的偏移"
fi

# data 字段 (std::vector<uint8_t>)
if grep -q 'data' "$GRAPH_H" 2>/dev/null && grep -q 'uint8_t' "$GRAPH_H" 2>/dev/null; then
    pass "0-1b: Tensor 包含 data 字段（uint8_t vector）"
else
    fail "0-1b: Tensor 缺少 data 字段" "常量张量需要 data 字段在编译期保存内容"
fi

# --- 检查辅助函数 ---
# dtype_size
if grep -q 'dtype_size' "$GRAPH_H" 2>/dev/null; then
    pass "0-1c: dtype_size() 辅助函数存在"
else
    fail "0-1c: dtype_size() 辅助函数不存在"
fi

# num_elements
if grep -q 'num_elements' "$GRAPH_H" 2>/dev/null; then
    pass "0-1d: num_elements() 辅助函数存在"
else
    fail "0-1d: num_elements() 辅助函数不存在" "需要 num_elements 来计算张量元素数量"
fi

# align_up
if grep -q 'align_up' "$GRAPH_H" 2>/dev/null; then
    pass "0-1e: align_up() 辅助函数存在"
else
    fail "0-1e: align_up() 辅助函数不存在" "内存对齐需要 align_up 辅助函数"
fi

# --- 检查 Graph 接口 ---
# add_tensor
if grep -q 'add_tensor' "$GRAPH_H" 2>/dev/null; then
    pass "0-1f: Graph::add_tensor() 接口存在"
else
    fail "0-1f: Graph::add_tensor() 接口不存在"
fi

# add_op
if grep -q 'add_op' "$GRAPH_H" 2>/dev/null; then
    pass "0-1g: Graph::add_op() 接口存在"
else
    fail "0-1g: Graph::add_op() 接口不存在"
fi

# summary
if grep -q 'summary' "$GRAPH_H" 2>/dev/null; then
    pass "0-1h: Graph::summary() 接口存在"
else
    fail "0-1h: Graph::summary() 接口不存在"
fi

# --- 编译一个小程序来验证 IR 功能正确性 ---
cat > "$TMP_DIR/check_ir.cpp" << 'CHECKER_EOF'
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include "tiny_tvm/ir/graph.h"

using namespace tiny_tvm::ir;

int main() {
    int errors = 0;

    // Test 1: dtype_size
    if (dtype_size(DType::kFloat32) != 4) {
        std::cerr << "ERROR: dtype_size(kFloat32) != 4\n";
        errors++;
    }
    if (dtype_size(DType::kInt32) != 4) {
        std::cerr << "ERROR: dtype_size(kInt32) != 4\n";
        errors++;
    }
    if (dtype_size(DType::kUnknown) != 0) {
        std::cerr << "ERROR: dtype_size(kUnknown) != 0\n";
        errors++;
    }

    // Test 2: Graph construction
    Graph graph;
    Tensor t1;
    t1.name = "input";
    t1.shape = {1, 4};
    t1.dtype = DType::kFloat32;
    int id1 = graph.add_tensor(std::move(t1));

    Tensor t2;
    t2.name = "weight";
    t2.shape = {4, 8};
    t2.dtype = DType::kFloat32;
    t2.is_constant = true;
    int id2 = graph.add_tensor(std::move(t2));

    Tensor t3;
    t3.name = "output";
    t3.shape = {1, 8};
    t3.dtype = DType::kFloat32;
    int id3 = graph.add_tensor(std::move(t3));

    if (graph.tensor_count() != 3) {
        std::cerr << "ERROR: tensor_count() != 3\n";
        errors++;
    }

    // Test 3: Op construction
    Op op1;
    op1.kind = "MatMul";
    op1.inputs = {id1, id2};
    op1.outputs = {id3};
    graph.add_op(std::move(op1));

    if (graph.op_count() != 1) {
        std::cerr << "ERROR: op_count() != 1\n";
        errors++;
    }

    // Test 4: graph_inputs / graph_outputs
    graph.graph_inputs().push_back(id1);
    graph.graph_outputs().push_back(id3);

    if (graph.graph_inputs().size() != 1) {
        std::cerr << "ERROR: graph_inputs size != 1\n";
        errors++;
    }
    if (graph.graph_outputs().size() != 1) {
        std::cerr << "ERROR: graph_outputs size != 1\n";
        errors++;
    }

    // Test 5: summary
    std::string s = graph.summary();
    if (s.empty()) {
        std::cerr << "ERROR: summary() is empty\n";
        errors++;
    }

    // Test 6: tensor accessor
    const auto& tensor = graph.tensor(id1);
    if (tensor.name != "input") {
        std::cerr << "ERROR: tensor(0).name != 'input'\n";
        errors++;
    }

    if (errors > 0) {
        std::cerr << "IR check: " << errors << " errors\n";
        return 1;
    }
    std::cout << "IR_CHECK_OK\n";
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 -I"$ROOT_DIR/include" "$TMP_DIR/check_ir.cpp" \
     "$ROOT_DIR/src/ir/graph.cpp" -o "$TMP_DIR/check_ir" 2>"$TMP_DIR/ir_compile.log"; then
    IR_OUTPUT=$("$TMP_DIR/check_ir" 2>&1)
    if echo "$IR_OUTPUT" | grep -q "IR_CHECK_OK"; then
        pass "0-1i: IR 功能验证通过（Graph/Tensor/Op 构造、访问、summary）"
    else
        fail "0-1i: IR 功能验证失败" "$IR_OUTPUT"
    fi
else
    fail "0-1i: IR checker 编译失败" "$(cat "$TMP_DIR/ir_compile.log")"
fi

# =====================================================================
# 任务 0-2：Pass 基础设施
# =====================================================================
section "任务 0-2：Pass 基础设施"

PASS_H="$ROOT_DIR/include/tiny_tvm/pass/pass.h"
PM_H="$ROOT_DIR/include/tiny_tvm/pass/pass_manager.h"

# Pass 基类
if grep -q 'class Pass' "$PASS_H" 2>/dev/null; then
    pass "0-2a: Pass 基类存在"
else
    fail "0-2a: Pass 基类不存在"
fi

# name() 虚函数
if grep -q 'virtual.*name()' "$PASS_H" 2>/dev/null; then
    pass "0-2b: Pass::name() 虚函数存在"
else
    fail "0-2b: Pass::name() 虚函数不存在"
fi

# run(Graph&) 虚函数
if grep -q 'virtual.*run.*Graph' "$PASS_H" 2>/dev/null; then
    pass "0-2c: Pass::run(Graph&) 虚函数存在"
else
    fail "0-2c: Pass::run(Graph&) 虚函数不存在"
fi

# NoOpPass
if grep -q 'NoOpPass' "$PASS_H" 2>/dev/null; then
    pass "0-2d: NoOpPass 实现存在"
else
    fail "0-2d: NoOpPass 实现不存在"
fi

# PassManager::add
if grep -q 'add' "$PM_H" 2>/dev/null; then
    pass "0-2e: PassManager::add() 接口存在"
else
    fail "0-2e: PassManager::add() 接口不存在"
fi

# PassManager::run
if grep -q 'run' "$PM_H" 2>/dev/null; then
    pass "0-2f: PassManager::run() 接口存在"
else
    fail "0-2f: PassManager::run() 接口不存在"
fi

# --- 编译 checker 验证 PassManager 多 pass 执行 ---
cat > "$TMP_DIR/check_pass.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass.h"
#include "tiny_tvm/pass/pass_manager.h"

using namespace tiny_tvm;

// 自定义测试 Pass：给 graph 中第一个 op 的 kind 追加标记
class MarkPass : public passes::Pass {
public:
    explicit MarkPass(std::string tag) : tag_(std::move(tag)) {}
    std::string name() const override { return "MarkPass_" + tag_; }
    void run(ir::Graph& graph) override {
        if (graph.op_count() > 0) {
            graph.op(0).kind += tag_;
        }
    }
private:
    std::string tag_;
};

int main() {
    int errors = 0;

    // Build a graph
    ir::Graph graph;
    ir::Tensor t;
    t.name = "x";
    t.shape = {1};
    t.dtype = ir::DType::kFloat32;
    int tid = graph.add_tensor(std::move(t));

    ir::Op op;
    op.kind = "Base";
    op.inputs = {tid};
    op.outputs = {tid};
    graph.add_op(std::move(op));

    // Test multiple passes execute in order
    passes::PassManager pm;
    pm.add(std::make_unique<passes::NoOpPass>());
    pm.add(std::make_unique<MarkPass>("_A"));
    pm.add(std::make_unique<MarkPass>("_B"));
    pm.run(graph);

    std::string kind = graph.op(0).kind;
    if (kind != "Base_A_B") {
        std::cerr << "ERROR: multi-pass execution order wrong, got: " << kind << "\n";
        errors++;
    }

    // Test pass_names
    auto names = pm.pass_names();
    if (names.size() != 3) {
        std::cerr << "ERROR: pass_names().size() != 3, got: " << names.size() << "\n";
        errors++;
    }

    if (errors > 0) {
        std::cerr << "Pass check: " << errors << " errors\n";
        return 1;
    }
    std::cout << "PASS_CHECK_OK\n";
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 -I"$ROOT_DIR/include" "$TMP_DIR/check_pass.cpp" \
     "$ROOT_DIR/src/ir/graph.cpp" "$ROOT_DIR/src/pass/pass_manager.cpp" \
     -o "$TMP_DIR/check_pass" 2>"$TMP_DIR/pass_compile.log"; then
    PASS_OUTPUT=$("$TMP_DIR/check_pass" 2>&1)
    if echo "$PASS_OUTPUT" | grep -q "PASS_CHECK_OK"; then
        pass "0-2g: PassManager 多 pass 连续执行验证通过"
    else
        fail "0-2g: PassManager 多 pass 连续执行验证失败" "$PASS_OUTPUT"
    fi
else
    fail "0-2g: Pass checker 编译失败" "$(cat "$TMP_DIR/pass_compile.log")"
fi

# =====================================================================
# 任务 0-3：CLI 骨架
# =====================================================================
section "任务 0-3：CLI 骨架"

TTVMC="$BUILD_DIR/ttvmc"

if [ ! -f "$TTVMC" ]; then
    fail "0-3a: ttvmc 可执行文件不存在" "请先执行构建"
else
    # help 命令
    HELP_OUT=$("$TTVMC" help 2>&1 || true)
    if echo "$HELP_OUT" | grep -qi -e "usage" -e "help" -e "smoke" -e "compile"; then
        pass "0-3a: ttvmc help 输出包含 usage 信息"
    else
        fail "0-3a: ttvmc help 输出缺少 usage 信息"
    fi

    # version 命令
    VER_OUT=$("$TTVMC" version 2>&1 || true)
    if echo "$VER_OUT" | grep -qE '[0-9]+\.[0-9]+'; then
        pass "0-3b: ttvmc version 输出版本号"
    else
        fail "0-3b: ttvmc version 未输出版本号" "got: $VER_OUT"
    fi

    # smoke 命令
    SMOKE_OUT=$("$TTVMC" smoke 2>&1 || true)

    # 检查 smoke 输出包含 graph summary
    if echo "$SMOKE_OUT" | grep -qi -e "tensor" -e "graph" -e "op"; then
        pass "0-3c: ttvmc smoke 输出包含 graph summary 信息"
    else
        fail "0-3c: ttvmc smoke 未输出 graph summary" "got: $SMOKE_OUT"
    fi

    # 检查 smoke 输出包含 runtime 信息
    if echo "$SMOKE_OUT" | grep -qi -e "runtime" -e "workspace"; then
        pass "0-3d: ttvmc smoke 输出包含 runtime 信息"
    else
        fail "0-3d: ttvmc smoke 未输出 runtime 信息"
    fi

    # 检查 smoke 输出包含 C 代码骨架
    if echo "$SMOKE_OUT" | grep -q "tiny_tvm_run"; then
        pass "0-3e: ttvmc smoke 输出包含 C 代码骨架（tiny_tvm_run）"
    else
        fail "0-3e: ttvmc smoke 未输出 C 代码骨架"
    fi

    # compile 命令应有占位或实现
    COMPILE_OUT=$("$TTVMC" compile 2>&1 || true)
    COMPILE_RC=$?
    # compile 要么成功执行，要么给出有意义的提示
    if echo "$COMPILE_OUT" | grep -qi -e "compile" -e "usage" -e "model" -e "not implemented"; then
        pass "0-3f: ttvmc compile 有占位或实现（给出有意义提示）"
    else
        fail "0-3f: ttvmc compile 无任何有意义输出"
    fi
fi

# =====================================================================
# 任务 0-4：测试与构建入口
# =====================================================================
section "任务 0-4：测试与构建入口"

# build_host.sh 存在
BUILD_SCRIPT="$ROOT_DIR/scripts/build_host.sh"
if [ -f "$BUILD_SCRIPT" ]; then
    pass "0-4a: scripts/build_host.sh 存在"
else
    fail "0-4a: scripts/build_host.sh 不存在"
fi

# build_host.sh 中有 cmake 依赖检查
if [ -f "$BUILD_SCRIPT" ] && grep -q "cmake" "$BUILD_SCRIPT" 2>/dev/null; then
    pass "0-4b: build_host.sh 包含 cmake 调用"
else
    fail "0-4b: build_host.sh 不包含 cmake 调用"
fi

# graph_smoke_test 存在且可执行
SMOKE_TEST="$BUILD_DIR/graph_smoke_test"
if [ -f "$SMOKE_TEST" ]; then
    pass "0-4c: graph_smoke_test 可执行文件存在"
else
    fail "0-4c: graph_smoke_test 可执行文件不存在"
fi

# graph_smoke_test 运行通过
if [ -f "$SMOKE_TEST" ]; then
    if "$SMOKE_TEST" > /dev/null 2>&1; then
        pass "0-4d: graph_smoke_test 运行通过"
    else
        fail "0-4d: graph_smoke_test 运行失败"
    fi
else
    fail "0-4d: graph_smoke_test 无法运行（不存在）"
fi

# CMakeLists.txt 中包含 tiny_tvm_core
if grep -q 'tiny_tvm_core' "$ROOT_DIR/CMakeLists.txt" 2>/dev/null; then
    pass "0-4e: CMakeLists.txt 定义 tiny_tvm_core 库"
else
    fail "0-4e: CMakeLists.txt 未定义 tiny_tvm_core 库"
fi

# CMakeLists.txt 中包含 ttvmc
if grep -q 'ttvmc' "$ROOT_DIR/CMakeLists.txt" 2>/dev/null; then
    pass "0-4f: CMakeLists.txt 定义 ttvmc 可执行文件"
else
    fail "0-4f: CMakeLists.txt 未定义 ttvmc 可执行文件"
fi

# CMakeLists.txt 中包含 graph_smoke_test
if grep -q 'graph_smoke_test' "$ROOT_DIR/CMakeLists.txt" 2>/dev/null; then
    pass "0-4g: CMakeLists.txt 注册 graph_smoke_test 测试"
else
    fail "0-4g: CMakeLists.txt 未注册 graph_smoke_test 测试"
fi

# ctest 通过
if cd "$BUILD_DIR" && ctest --output-on-failure --no-tests=error > /dev/null 2>&1; then
    pass "0-4h: ctest 所有测试通过"
else
    fail "0-4h: ctest 部分测试失败"
fi

# =====================================================================
# 汇总
# =====================================================================
section "Phase 0 测试结果汇总"

echo ""
echo "  通过: $PASS_COUNT"
echo "  失败: $FAIL_COUNT"
echo "  总计: $TOTAL_COUNT"
echo ""
echo "  Score: $PASS_COUNT/$TOTAL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "🎉 Phase 0 所有检测项通过！工程骨架完善。"
    exit 0
else
    echo "⚠️  Phase 0 有 $FAIL_COUNT 项未通过，请根据上方 [FAIL] 提示修复。"
    exit 1
fi
