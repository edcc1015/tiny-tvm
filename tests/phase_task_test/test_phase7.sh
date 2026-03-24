#!/usr/bin/env bash
# =============================================================================
# Phase 7 测试脚本：Graph Transforms & Operator Expansion
# =============================================================================
# 检测要点（~20 points total）：
#   7-1: 新算子 OpKind（kMaxPool2D, kGlobalAvgPool2D, kSoftmax, kBatchNorm） [4 pts]
#   7-2: MaxPool2D 形状推导                                                    [2 pts]
#   7-3: GlobalAvgPool2D 形状推导                                              [2 pts]
#   7-4: Softmax 形状推导                                                      [1 pt]
#   7-5: BatchNorm 形状推导                                                    [1 pt]
#   7-6: BNFoldPass 存在性                                                     [2 pts]
#   7-7: BNFoldPass 折叠 Conv+BN                                              [3 pts]
#   7-8: FusePass 模式匹配扩展                                                [3 pts]
#   7-9: 新算子 CNN 端到端测试                                                 [2 pts]
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0; FAIL=0; TOTAL=0
pass() { ((++PASS)); ((++TOTAL)); echo "[PASS] $1"; }
fail() { ((++FAIL)); ((++TOTAL)); echo "[FAIL] $1"; }
section() { echo ""; echo "======================================================================"; echo "  $1"; echo "======================================================================"; }

CORE_SOURCES=$(find "$ROOT_DIR/src" -name "*.cpp" ! -path "*/tools/*" 2>/dev/null | tr '\n' ' ')
INCLUDE_DIRS="-I$ROOT_DIR/include"
[ -d "$ROOT_DIR/third_party" ] && INCLUDE_DIRS="$INCLUDE_DIRS -I$ROOT_DIR/third_party"

# Pre-build
section "前置：编译项目"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug > /dev/null 2>&1 || true
cmake --build "$BUILD_DIR" --parallel > /dev/null 2>&1 || echo "WARNING: build errors"

TTVMC="$BUILD_DIR/ttvmc"

# =====================================================================
# 7-1: OpKind enum — new operators (4 pts, 1 each)
# =====================================================================
section "7-1: OpKind enum — 新算子枚举值"

for OP_ENTRY in "kMaxPool2D:7-1a" "kGlobalAvgPool2D:7-1b" "kSoftmax:7-1c" "kBatchNorm:7-1d"; do
    OP_KIND="${OP_ENTRY%%:*}"
    TEST_ID="${OP_ENTRY##*:}"

    cat > "$TMP_DIR/check_${OP_KIND}.cpp" << EOF
#include <iostream>
#include "tiny_tvm/ir/graph.h"
using namespace tiny_tvm::ir;
int main() {
    OpKind k = OpKind::${OP_KIND};
    (void)k;
    std::cout << "CHECK_OK" << std::endl;
    return 0;
}
EOF

    if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_${OP_KIND}.cpp" $CORE_SOURCES \
         -o "$TMP_DIR/check_${OP_KIND}" 2>/dev/null; then
        OUTPUT=$("$TMP_DIR/check_${OP_KIND}" 2>&1 || true)
        if echo "$OUTPUT" | grep -q "CHECK_OK"; then
            pass "$TEST_ID: OpKind::${OP_KIND} exists"
        else
            fail "$TEST_ID: OpKind::${OP_KIND} runtime error"
        fi
    else
        fail "$TEST_ID: OpKind::${OP_KIND} does not compile"
    fi
done

# =====================================================================
# 7-2: MaxPool2D shape inference (2 pts)
# Input [1,1,8,8], kernel=3, stride=2, pad=1 → [1,1,4,4]
# Formula: (H + 2*pad - KH) / stride + 1 = (8+2-3)/2+1 = 4
# =====================================================================
section "7-2: MaxPool2D 形状推导"

cat > "$TMP_DIR/check_maxpool_shape.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"
#include "tiny_tvm/pass/graph/infer_shape_pass.h"

using namespace tiny_tvm::ir;

int main() {
    try {
        Graph graph;

        Tensor input;
        input.name = "input"; input.shape = {1, 1, 8, 8}; input.dtype = DType::kFloat32;
        int t0 = graph.add_tensor(std::move(input));

        Tensor output;
        output.name = "pool_out"; output.dtype = DType::kFloat32;
        int t1 = graph.add_tensor(std::move(output));

        Op pool;
        pool.kind = "MaxPool2D";
        pool.inputs = {t0};
        pool.outputs = {t1};
        pool.attrs["kernel_size"] = std::vector<int64_t>{3, 3};
        pool.attrs["stride"]      = std::vector<int64_t>{2, 2};
        pool.attrs["padding"]     = std::vector<int64_t>{1, 1};
        graph.add_op(std::move(pool));
        graph.graph_inputs().push_back(t0);
        graph.graph_outputs().push_back(t1);

        tiny_tvm::passes::PassManager pm;
        pm.add(std::make_unique<tiny_tvm::passes::InferShapePass>());
        pm.run(graph);

        auto& s = graph.tensor(t1).shape;
        std::cout << "SHAPE:";
        for (size_t i = 0; i < s.size(); i++) std::cout << (i ? "," : "") << s[i];
        std::cout << std::endl;

        if (s.size() == 4 && s[0] == 1 && s[1] == 1 && s[2] == 4 && s[3] == 4) {
            std::cout << "CHECK_OK" << std::endl;
        } else {
            std::cout << "CHECK_FAIL: expected 1,1,4,4" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "CHECK_FAIL: " << e.what() << std::endl;
    }
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_maxpool_shape.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_maxpool_shape" 2>/dev/null; then
    pass "7-2a: MaxPool2D + InferShapePass compiles"
    OUTPUT=$("$TMP_DIR/check_maxpool_shape" 2>&1 || true)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "7-2b: MaxPool2D shape = [1,1,4,4] (input [1,1,8,8], k=3, s=2, p=1)"
    else
        fail "7-2b: MaxPool2D shape incorrect — $(echo "$OUTPUT" | grep -E 'SHAPE:|CHECK_FAIL' | head -1)"
    fi
else
    fail "7-2a: MaxPool2D + InferShapePass does not compile"
    fail "7-2b: MaxPool2D shape (skipped — compile failed)"
fi

# =====================================================================
# 7-3: GlobalAvgPool2D shape inference (2 pts)
# Input [1,3,8,8] → Output [1,3,1,1]
# =====================================================================
section "7-3: GlobalAvgPool2D 形状推导"

cat > "$TMP_DIR/check_gap_shape.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"
#include "tiny_tvm/pass/graph/infer_shape_pass.h"

using namespace tiny_tvm::ir;

int main() {
    try {
        Graph graph;

        Tensor input;
        input.name = "input"; input.shape = {1, 3, 8, 8}; input.dtype = DType::kFloat32;
        int t0 = graph.add_tensor(std::move(input));

        Tensor output;
        output.name = "gap_out"; output.dtype = DType::kFloat32;
        int t1 = graph.add_tensor(std::move(output));

        Op gap;
        gap.kind = "GlobalAvgPool2D";
        gap.inputs = {t0};
        gap.outputs = {t1};
        graph.add_op(std::move(gap));
        graph.graph_inputs().push_back(t0);
        graph.graph_outputs().push_back(t1);

        tiny_tvm::passes::PassManager pm;
        pm.add(std::make_unique<tiny_tvm::passes::InferShapePass>());
        pm.run(graph);

        auto& s = graph.tensor(t1).shape;
        std::cout << "SHAPE:";
        for (size_t i = 0; i < s.size(); i++) std::cout << (i ? "," : "") << s[i];
        std::cout << std::endl;

        if (s.size() == 4 && s[0] == 1 && s[1] == 3 && s[2] == 1 && s[3] == 1) {
            std::cout << "CHECK_OK" << std::endl;
        } else {
            std::cout << "CHECK_FAIL: expected 1,3,1,1" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "CHECK_FAIL: " << e.what() << std::endl;
    }
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_gap_shape.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_gap_shape" 2>/dev/null; then
    pass "7-3a: GlobalAvgPool2D + InferShapePass compiles"
    OUTPUT=$("$TMP_DIR/check_gap_shape" 2>&1 || true)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "7-3b: GlobalAvgPool2D shape = [1,3,1,1] (input [1,3,8,8])"
    else
        fail "7-3b: GlobalAvgPool2D shape incorrect — $(echo "$OUTPUT" | grep -E 'SHAPE:|CHECK_FAIL' | head -1)"
    fi
else
    fail "7-3a: GlobalAvgPool2D + InferShapePass does not compile"
    fail "7-3b: GlobalAvgPool2D shape (skipped — compile failed)"
fi

# =====================================================================
# 7-4: Softmax shape inference (1 pt)
# Input [1,10] → Output [1,10]  (shape preserved)
# =====================================================================
section "7-4: Softmax 形状推导"

cat > "$TMP_DIR/check_softmax_shape.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"
#include "tiny_tvm/pass/graph/infer_shape_pass.h"

using namespace tiny_tvm::ir;

int main() {
    try {
        Graph graph;

        Tensor input;
        input.name = "input"; input.shape = {1, 10}; input.dtype = DType::kFloat32;
        int t0 = graph.add_tensor(std::move(input));

        Tensor output;
        output.name = "sm_out"; output.dtype = DType::kFloat32;
        int t1 = graph.add_tensor(std::move(output));

        Op sm;
        sm.kind = "Softmax";
        sm.inputs = {t0};
        sm.outputs = {t1};
        sm.attrs["axis"] = int64_t{-1};
        graph.add_op(std::move(sm));
        graph.graph_inputs().push_back(t0);
        graph.graph_outputs().push_back(t1);

        tiny_tvm::passes::PassManager pm;
        pm.add(std::make_unique<tiny_tvm::passes::InferShapePass>());
        pm.run(graph);

        auto& s = graph.tensor(t1).shape;
        std::cout << "SHAPE:";
        for (size_t i = 0; i < s.size(); i++) std::cout << (i ? "," : "") << s[i];
        std::cout << std::endl;

        if (s.size() == 2 && s[0] == 1 && s[1] == 10) {
            std::cout << "CHECK_OK" << std::endl;
        } else {
            std::cout << "CHECK_FAIL: expected 1,10" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "CHECK_FAIL: " << e.what() << std::endl;
    }
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_softmax_shape.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_softmax_shape" 2>/dev/null; then
    OUTPUT=$("$TMP_DIR/check_softmax_shape" 2>&1 || true)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "7-4: Softmax shape preserved [1,10] → [1,10]"
    else
        fail "7-4: Softmax shape incorrect — $(echo "$OUTPUT" | grep -E 'SHAPE:|CHECK_FAIL' | head -1)"
    fi
else
    fail "7-4: Softmax + InferShapePass does not compile"
fi

# =====================================================================
# 7-5: BatchNorm shape inference (1 pt)
# Input [1,3,8,8] + scale/bias/mean/var [3] each → Output [1,3,8,8]
# =====================================================================
section "7-5: BatchNorm 形状推导"

cat > "$TMP_DIR/check_bn_shape.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"
#include "tiny_tvm/pass/graph/infer_shape_pass.h"

using namespace tiny_tvm::ir;

int main() {
    try {
        Graph graph;

        Tensor data;
        data.name = "data"; data.shape = {1, 3, 8, 8}; data.dtype = DType::kFloat32;
        int t0 = graph.add_tensor(std::move(data));

        Tensor gamma;
        gamma.name = "gamma"; gamma.shape = {3}; gamma.dtype = DType::kFloat32;
        gamma.is_constant = true;
        int t1 = graph.add_tensor(std::move(gamma));

        Tensor beta;
        beta.name = "beta"; beta.shape = {3}; beta.dtype = DType::kFloat32;
        beta.is_constant = true;
        int t2 = graph.add_tensor(std::move(beta));

        Tensor mean;
        mean.name = "running_mean"; mean.shape = {3}; mean.dtype = DType::kFloat32;
        mean.is_constant = true;
        int t3 = graph.add_tensor(std::move(mean));

        Tensor var;
        var.name = "running_var"; var.shape = {3}; var.dtype = DType::kFloat32;
        var.is_constant = true;
        int t4 = graph.add_tensor(std::move(var));

        Tensor output;
        output.name = "bn_out"; output.dtype = DType::kFloat32;
        int t5 = graph.add_tensor(std::move(output));

        Op bn;
        bn.kind = "BatchNorm";
        bn.inputs = {t0, t1, t2, t3, t4};
        bn.outputs = {t5};
        bn.attrs["epsilon"] = 1e-5;
        graph.add_op(std::move(bn));
        graph.graph_inputs().push_back(t0);
        graph.graph_outputs().push_back(t5);

        tiny_tvm::passes::PassManager pm;
        pm.add(std::make_unique<tiny_tvm::passes::InferShapePass>());
        pm.run(graph);

        auto& s = graph.tensor(t5).shape;
        std::cout << "SHAPE:";
        for (size_t i = 0; i < s.size(); i++) std::cout << (i ? "," : "") << s[i];
        std::cout << std::endl;

        if (s.size() == 4 && s[0] == 1 && s[1] == 3 && s[2] == 8 && s[3] == 8) {
            std::cout << "CHECK_OK" << std::endl;
        } else {
            std::cout << "CHECK_FAIL: expected 1,3,8,8" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "CHECK_FAIL: " << e.what() << std::endl;
    }
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_bn_shape.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_bn_shape" 2>/dev/null; then
    OUTPUT=$("$TMP_DIR/check_bn_shape" 2>&1 || true)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "7-5: BatchNorm shape preserved [1,3,8,8] → [1,3,8,8] (5 inputs)"
    else
        fail "7-5: BatchNorm shape incorrect — $(echo "$OUTPUT" | grep -E 'SHAPE:|CHECK_FAIL' | head -1)"
    fi
else
    fail "7-5: BatchNorm + InferShapePass does not compile"
fi

# =====================================================================
# 7-6: BNFoldPass exists (2 pts)
# =====================================================================
section "7-6: BNFoldPass 存在性"

BN_FOLD_H="$ROOT_DIR/include/tiny_tvm/pass/graph/bn_fold_pass.h"
if [ -f "$BN_FOLD_H" ]; then
    pass "7-6a: bn_fold_pass.h exists"
else
    fail "7-6a: bn_fold_pass.h not found"
fi

cat > "$TMP_DIR/check_bnfold_pass.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <memory>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass.h"
#include "tiny_tvm/pass/graph/bn_fold_pass.h"

int main() {
    try {
        auto p = std::make_unique<tiny_tvm::passes::BNFoldPass>();
        tiny_tvm::passes::Pass* base = p.get();
        std::string n = base->name();
        std::cout << "NAME:" << n << std::endl;
        std::cout << "CHECK_OK" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CHECK_FAIL: " << e.what() << std::endl;
    }
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_bnfold_pass.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_bnfold_pass" 2>/dev/null; then
    OUTPUT=$("$TMP_DIR/check_bnfold_pass" 2>&1 || true)
    if echo "$OUTPUT" | grep -q "CHECK_OK"; then
        pass "7-6b: BNFoldPass instantiates as Pass"
    else
        fail "7-6b: BNFoldPass runtime error — $(echo "$OUTPUT" | head -1)"
    fi
else
    fail "7-6b: BNFoldPass does not compile (expected class tiny_tvm::passes::BNFoldPass)"
fi

# =====================================================================
# 7-7: BNFoldPass folds Conv2D + BatchNorm (3 pts)
# Conv2D([1,3,8,8], w[16,3,3,3]) → [1,16,8,8] → BN → [1,16,8,8]
# After fold: BN op removed, op count decreases
# =====================================================================
section "7-7: BNFoldPass 折叠 Conv+BN"

cat > "$TMP_DIR/check_bnfold_run.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include <cstring>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"
#include "tiny_tvm/pass/graph/bn_fold_pass.h"

using namespace tiny_tvm::ir;

int main() {
    try {
        Graph graph;

        // Conv2D input
        Tensor conv_in;
        conv_in.name = "conv_in"; conv_in.shape = {1, 3, 8, 8}; conv_in.dtype = DType::kFloat32;
        int t0 = graph.add_tensor(std::move(conv_in));

        // Conv2D weight
        Tensor conv_w;
        conv_w.name = "conv_w"; conv_w.shape = {16, 3, 3, 3}; conv_w.dtype = DType::kFloat32;
        conv_w.is_constant = true;
        conv_w.data.resize(16 * 3 * 3 * 3 * sizeof(float), 0);
        int t1 = graph.add_tensor(std::move(conv_w));

        // Conv2D output
        Tensor conv_out;
        conv_out.name = "conv_out"; conv_out.shape = {1, 16, 8, 8}; conv_out.dtype = DType::kFloat32;
        int t2 = graph.add_tensor(std::move(conv_out));

        // BN parameters (gamma, beta, mean, var) — channel dim = 16
        auto make_bn_param = [&](const char* name) {
            Tensor t;
            t.name = name; t.shape = {16}; t.dtype = DType::kFloat32;
            t.is_constant = true;
            t.data.resize(16 * sizeof(float), 0);
            // Fill with 1.0 for gamma/var, 0.0 for beta/mean
            if (std::string(name) == "bn_gamma" || std::string(name) == "bn_var") {
                float one = 1.0f;
                for (int i = 0; i < 16; i++)
                    std::memcpy(t.data.data() + i * sizeof(float), &one, sizeof(float));
            }
            return graph.add_tensor(std::move(t));
        };
        int t_gamma = make_bn_param("bn_gamma");
        int t_beta  = make_bn_param("bn_beta");
        int t_mean  = make_bn_param("bn_mean");
        int t_var   = make_bn_param("bn_var");

        // BN output
        Tensor bn_out;
        bn_out.name = "bn_out"; bn_out.shape = {1, 16, 8, 8}; bn_out.dtype = DType::kFloat32;
        int t_bn_out = graph.add_tensor(std::move(bn_out));

        // Conv2D op
        Op conv;
        conv.kind = "Conv2D";
        conv.inputs = {t0, t1};
        conv.outputs = {t2};
        conv.attrs["kernel_size"] = std::vector<int64_t>{3, 3};
        conv.attrs["stride"]      = std::vector<int64_t>{1, 1};
        conv.attrs["padding"]     = std::vector<int64_t>{1, 1};
        graph.add_op(std::move(conv));

        // BatchNorm op
        Op bn;
        bn.kind = "BatchNorm";
        bn.inputs = {t2, t_gamma, t_beta, t_mean, t_var};
        bn.outputs = {t_bn_out};
        bn.attrs["epsilon"] = 1e-5;
        graph.add_op(std::move(bn));

        graph.graph_inputs().push_back(t0);
        graph.graph_outputs().push_back(t_bn_out);

        size_t ops_before = graph.op_count();
        std::cout << "OPS_BEFORE:" << ops_before << std::endl;

        // Run BNFoldPass
        tiny_tvm::passes::PassManager pm;
        pm.add(std::make_unique<tiny_tvm::passes::BNFoldPass>());
        pm.run(graph);

        size_t ops_after = graph.op_count();
        std::cout << "OPS_AFTER:" << ops_after << std::endl;

        // Check: is there still a BatchNorm op?
        bool bn_found = false;
        for (size_t i = 0; i < graph.op_count(); i++) {
            if (graph.op(i).kind == "BatchNorm") {
                bn_found = true;
                break;
            }
        }

        if (!bn_found) {
            std::cout << "BN_REMOVED" << std::endl;
        } else {
            std::cout << "BN_STILL_PRESENT" << std::endl;
        }

        if (ops_after < ops_before) {
            std::cout << "OP_COUNT_DECREASED" << std::endl;
        }

        std::cout << "CHECK_OK" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CHECK_FAIL: " << e.what() << std::endl;
    }
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_bnfold_run.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_bnfold_run" 2>/dev/null; then
    pass "7-7a: Conv+BN graph with BNFoldPass compiles"
    OUTPUT=$("$TMP_DIR/check_bnfold_run" 2>&1 || true)
    if echo "$OUTPUT" | grep -q "BN_REMOVED"; then
        pass "7-7b: BatchNorm op removed after BNFoldPass"
    else
        fail "7-7b: BatchNorm op still present after BNFoldPass"
    fi
    if echo "$OUTPUT" | grep -q "OP_COUNT_DECREASED"; then
        pass "7-7c: Op count decreased after BNFoldPass"
    else
        fail "7-7c: Op count did not decrease — $(echo "$OUTPUT" | grep 'OPS_' | tr '\n' ' ')"
    fi
else
    fail "7-7a: Conv+BN graph with BNFoldPass does not compile"
    fail "7-7b: BNFoldPass fold (skipped — compile failed)"
    fail "7-7c: BNFoldPass op count (skipped — compile failed)"
fi

# =====================================================================
# 7-8: FusePass pattern matching extension (3 pts)
# Check FusePattern struct or MatMul+Add fusion
# =====================================================================
section "7-8: FusePass 模式匹配扩展"

FUSE_H="$ROOT_DIR/include/tiny_tvm/pass/graph/fuse_pass.h"
if [ -f "$FUSE_H" ]; then
    pass "7-8a: fuse_pass.h exists"
else
    fail "7-8a: fuse_pass.h not found"
fi

cat > "$TMP_DIR/check_fuse_matmul_add.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass_manager.h"
#include "tiny_tvm/pass/graph/fuse_pass.h"

using namespace tiny_tvm::ir;

int main() {
    try {
        Graph graph;

        Tensor t_in;
        t_in.name = "input"; t_in.shape = {1, 4}; t_in.dtype = DType::kFloat32;
        int i0 = graph.add_tensor(std::move(t_in));

        Tensor t_w;
        t_w.name = "weight"; t_w.shape = {4, 4}; t_w.dtype = DType::kFloat32;
        t_w.is_constant = true;
        int i1 = graph.add_tensor(std::move(t_w));

        Tensor t_mm;
        t_mm.name = "mm_out"; t_mm.shape = {1, 4}; t_mm.dtype = DType::kFloat32;
        int i2 = graph.add_tensor(std::move(t_mm));

        Tensor t_b;
        t_b.name = "bias"; t_b.shape = {1, 4}; t_b.dtype = DType::kFloat32;
        t_b.is_constant = true;
        int i3 = graph.add_tensor(std::move(t_b));

        Tensor t_out;
        t_out.name = "output"; t_out.shape = {1, 4}; t_out.dtype = DType::kFloat32;
        int i4 = graph.add_tensor(std::move(t_out));

        Op matmul;
        matmul.kind = "MatMul";
        matmul.inputs = {i0, i1};
        matmul.outputs = {i2};
        graph.add_op(std::move(matmul));

        Op add;
        add.kind = "Add";
        add.inputs = {i2, i3};
        add.outputs = {i4};
        graph.add_op(std::move(add));

        graph.graph_inputs().push_back(i0);
        graph.graph_outputs().push_back(i4);

        size_t before = graph.op_count();
        std::cout << "OPS_BEFORE:" << before << std::endl;

        tiny_tvm::passes::PassManager pm;
        pm.add(std::make_unique<tiny_tvm::passes::FusePass>());
        pm.run(graph);

        size_t after = graph.op_count();
        std::cout << "OPS_AFTER:" << after << std::endl;

        // Check for fused op (e.g., "MatMulAdd", "MatMul+Add", contains both substrings)
        bool found_fused = false;
        for (size_t i = 0; i < graph.op_count(); i++) {
            const auto& kind = graph.op(i).kind;
            if (kind.find("MatMul") != std::string::npos &&
                kind != "MatMul") {
                found_fused = true;
                std::cout << "FUSED_OP:" << kind << std::endl;
                break;
            }
        }

        if (after < before || found_fused) {
            std::cout << "FUSE_OK" << std::endl;
        } else {
            std::cout << "FUSE_FAIL: MatMul+Add not fused (ops unchanged)" << std::endl;
        }

        std::cout << "CHECK_OK" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CHECK_FAIL: " << e.what() << std::endl;
    }
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_fuse_matmul_add.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_fuse_matmul_add" 2>/dev/null; then
    pass "7-8b: FusePass with MatMul+Add graph compiles"
    OUTPUT=$("$TMP_DIR/check_fuse_matmul_add" 2>&1 || true)
    if echo "$OUTPUT" | grep -q "FUSE_OK"; then
        pass "7-8c: FusePass fuses MatMul+Add (op count decreased or fused op found)"
    else
        fail "7-8c: FusePass did not fuse MatMul+Add — $(echo "$OUTPUT" | grep -E 'OPS_|FUSE_FAIL' | tr '\n' ' ')"
    fi
else
    fail "7-8b: FusePass with MatMul+Add does not compile"
    fail "7-8c: FusePass fusion (skipped — compile failed)"
fi

# =====================================================================
# 7-9: CNN with new ops — end-to-end (2 pts)
# =====================================================================
section "7-9: 新算子 CNN 端到端测试"

CNN_BN_JSON="$ROOT_DIR/examples/json/cnn_bn.json"
if [ -f "$CNN_BN_JSON" ]; then
    pass "7-9a: examples/json/cnn_bn.json exists"
else
    fail "7-9a: examples/json/cnn_bn.json not found"
fi

if [ -f "$CNN_BN_JSON" ] && [ -f "$TTVMC" ]; then
    OUT_CNN="$TMP_DIR/out_cnn_bn"
    mkdir -p "$OUT_CNN"
    if "$TTVMC" compile "$CNN_BN_JSON" -o "$OUT_CNN" > /dev/null 2>&1; then
        if [ -f "$OUT_CNN/deploy.c" ] || [ -f "$OUT_CNN/libdeploy.so" ]; then
            pass "7-9b: ttvmc compile cnn_bn.json succeeds (deploy artifacts produced)"
        else
            fail "7-9b: ttvmc compile ran but no deploy artifacts produced"
        fi
    else
        fail "7-9b: ttvmc compile cnn_bn.json failed"
    fi
else
    if [ ! -f "$CNN_BN_JSON" ]; then
        fail "7-9b: ttvmc compile skipped (cnn_bn.json missing)"
    else
        fail "7-9b: ttvmc compile skipped (ttvmc binary not found)"
    fi
fi

# =====================================================================
# 汇总
# =====================================================================
echo ""
echo "========== Phase 7 Test Results =========="
echo "Score: $PASS / $TOTAL"
echo "Pass: $PASS  Fail: $FAIL"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo "🎉 Phase 7 所有检测项通过！Graph Transforms & Operator Expansion 完成。"
    exit 0
else
    echo "⚠️  Phase 7 有 $FAIL 项未通过，请根据上方 [FAIL] 提示修复。"
    exit 1
fi
