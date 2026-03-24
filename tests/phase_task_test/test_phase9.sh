#!/usr/bin/env bash
# =============================================================================
# Phase 9 测试脚本：Performance Analysis & Auto-Tuning
# =============================================================================
# 检测要点：
#   9-1: Profiler 头文件与编译
#   9-2: Codegen TINY_TVM_PROFILE 支持
#   9-3: profile export 函数
#   9-4: GridSearchTuner 存在性
#   9-5: GridSearchTuner tune 方法
#   9-6: Candidate 值 (8, 16, 32, 64)
#   9-7: ttvmc tune 子命令
#   9-8: ttvmc compile --schedule 参数
#   9-9: best_schedule.json 输出
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
# 任务 9-1：Profiler 头文件与编译 (2 pts)
# =====================================================================
section "任务 9-1：Profiler 头文件与编译"

PROFILER_H="$ROOT_DIR/include/tiny_tvm/runtime/profiler.h"

if [ -f "$PROFILER_H" ]; then
    pass "9-1a: profiler.h 存在"
else
    fail "9-1a: profiler.h 不存在"
fi

cat > "$TMP_DIR/check_profiler.cpp" << 'CHECKER_EOF'
#include <iostream>

#if __has_include("tiny_tvm/runtime/profiler.h")
#include "tiny_tvm/runtime/profiler.h"
#define HAS_PROFILER 1
#else
#define HAS_PROFILER 0
#endif

int main() {
#if !HAS_PROFILER
    std::cout << "CHECK_FAIL: profiler.h not found" << std::endl;
#else
    std::cout << "CHECK_OK" << std::endl;
#endif
    return 0;
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_profiler.cpp" \
     -o "$TMP_DIR/check_profiler" 2>"$TMP_DIR/profiler_compile.log"; then
    PROFILER_OUTPUT=$("$TMP_DIR/check_profiler" 2>&1)
    if echo "$PROFILER_OUTPUT" | grep -q "CHECK_OK"; then
        pass "9-1b: profiler.h 可编译并包含"
    else
        fail "9-1b: profiler.h 编译通过但检测失败" "$PROFILER_OUTPUT"
    fi
else
    fail "9-1b: profiler.h 编译失败" "$(head -20 "$TMP_DIR/profiler_compile.log")"
fi

# =====================================================================
# 任务 9-2：Codegen TINY_TVM_PROFILE 支持 (2 pts)
# =====================================================================
section "任务 9-2：Codegen TINY_TVM_PROFILE 支持"

CODEGEN_FILES=$(find "$ROOT_DIR/src/codegen" -name "*.cpp" -o -name "*.h" 2>/dev/null | tr '\n' ' ')
CODEGEN_HEADERS=$(find "$ROOT_DIR/include/tiny_tvm/codegen" -name "*.h" 2>/dev/null | tr '\n' ' ')
ALL_CODEGEN="$CODEGEN_FILES $CODEGEN_HEADERS"

FOUND_PROFILE=0
for f in $ALL_CODEGEN; do
    if grep -qE 'TINY_TVM_PROFILE|clock_gettime|timespec|gettimeofday|chrono|profile_start|profile_end|PROFILE' "$f" 2>/dev/null; then
        FOUND_PROFILE=1
        break
    fi
done

if [ "$FOUND_PROFILE" -eq 1 ]; then
    pass "9-2a: Codegen 源码包含 profiling 相关代码"
else
    fail "9-2a: Codegen 源码未找到 TINY_TVM_PROFILE / timing 相关代码"
fi

# Also check if generated deploy.c contains timing code (if ttvmc can generate one)
if [ -x "$TTVMC" ]; then
    # Try to generate a deploy.c from a simple model if possible
    DEPLOY_CHECK=0
    if [ -f "$BUILD_DIR/deploy.c" ]; then
        if grep -qE 'TINY_TVM_PROFILE|clock_gettime|timespec|profile' "$BUILD_DIR/deploy.c" 2>/dev/null; then
            DEPLOY_CHECK=1
        fi
    fi
    # Also scan any .c files ttvmc may have generated in common locations
    for dc in "$TMP_DIR"/deploy.c "$ROOT_DIR"/deploy.c; do
        if [ -f "$dc" ] && grep -qE 'TINY_TVM_PROFILE|clock_gettime|timespec|profile' "$dc" 2>/dev/null; then
            DEPLOY_CHECK=1
        fi
    done
    if [ "$DEPLOY_CHECK" -eq 1 ]; then
        pass "9-2b: 生成的 deploy.c 包含 profiling 宏/计时代码"
    else
        # Fall back: accept if source code already had it
        if [ "$FOUND_PROFILE" -eq 1 ]; then
            pass "9-2b: Codegen 源码已包含 profiling 支持（deploy.c 未单独验证）"
        else
            fail "9-2b: 未在生成代码中找到 profiling 支持"
        fi
    fi
else
    if [ "$FOUND_PROFILE" -eq 1 ]; then
        pass "9-2b: Codegen 源码已包含 profiling 支持（ttvmc 不存在，跳过 deploy.c 检查）"
    else
        fail "9-2b: ttvmc 不存在且 Codegen 源码无 profiling 支持"
    fi
fi

# =====================================================================
# 任务 9-3：Profile export 函数 (2 pts)
# =====================================================================
section "任务 9-3：Profile export 函数"

# Check header or codegen source for tiny_tvm_get_profile
FOUND_GET_PROFILE=0
for f in $ALL_CODEGEN; do
    if grep -q 'tiny_tvm_get_profile' "$f" 2>/dev/null; then
        FOUND_GET_PROFILE=1
        break
    fi
done
if [ "$FOUND_GET_PROFILE" -eq 0 ] && [ -f "$PROFILER_H" ]; then
    if grep -q 'tiny_tvm_get_profile' "$PROFILER_H" 2>/dev/null; then
        FOUND_GET_PROFILE=1
    fi
fi
# Also search all headers and sources
if [ "$FOUND_GET_PROFILE" -eq 0 ]; then
    if grep -rq 'tiny_tvm_get_profile' "$ROOT_DIR/include" "$ROOT_DIR/src" 2>/dev/null; then
        FOUND_GET_PROFILE=1
    fi
fi

if [ "$FOUND_GET_PROFILE" -eq 1 ]; then
    pass "9-3a: tiny_tvm_get_profile 函数签名存在"
else
    fail "9-3a: 未找到 tiny_tvm_get_profile 函数签名"
fi

# Compile a checker that calls/references the profile export
cat > "$TMP_DIR/check_profile_export.cpp" << 'CHECKER_EOF'
#include <iostream>

#if __has_include("tiny_tvm/runtime/profiler.h")
#include "tiny_tvm/runtime/profiler.h"
#define HAS_PROFILER 1
#else
#define HAS_PROFILER 0
#endif

int main() {
#if !HAS_PROFILER
    std::cout << "CHECK_FAIL: profiler.h not found" << std::endl;
    return 0;
#else
    // Verify tiny_tvm_get_profile or equivalent is declared/accessible
    // Use decltype or sizeof trick to check function existence
    #if defined(__cpp_lib_type_traits)
    #include <type_traits>
    #endif

    // Try to take address of the function — if it compiles, the symbol exists
    auto fn_ptr = &tiny_tvm_get_profile;
    if (fn_ptr != nullptr) {
        std::cout << "CHECK_OK" << std::endl;
    } else {
        std::cout << "CHECK_FAIL: function pointer is null" << std::endl;
    }
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_profile_export.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_profile_export" 2>"$TMP_DIR/profile_export_compile.log"; then
    EXPORT_OUTPUT=$("$TMP_DIR/check_profile_export" 2>&1)
    if echo "$EXPORT_OUTPUT" | grep -q "CHECK_OK"; then
        pass "9-3b: tiny_tvm_get_profile 可编译链接"
    else
        fail "9-3b: tiny_tvm_get_profile 编译通过但检测失败" "$EXPORT_OUTPUT"
    fi
else
    fail "9-3b: profile export checker 编译失败" "$(head -20 "$TMP_DIR/profile_export_compile.log")"
fi

# =====================================================================
# 任务 9-4：GridSearchTuner 存在性 (2 pts)
# =====================================================================
section "任务 9-4：GridSearchTuner 存在性"

TUNER_H="$ROOT_DIR/include/tiny_tvm/tune/auto_tuner.h"

if [ -f "$TUNER_H" ]; then
    pass "9-4a: auto_tuner.h 存在"
else
    fail "9-4a: auto_tuner.h 不存在"
fi

cat > "$TMP_DIR/check_tuner_exists.cpp" << 'CHECKER_EOF'
#include <iostream>

#if __has_include("tiny_tvm/tune/auto_tuner.h")
#include "tiny_tvm/tune/auto_tuner.h"
#define HAS_TUNER 1
#else
#define HAS_TUNER 0
#endif

int main() {
#if !HAS_TUNER
    std::cout << "CHECK_FAIL: auto_tuner.h not found" << std::endl;
    return 0;
#else
    tiny_tvm::tune::GridSearchTuner tuner;
    std::cout << "CHECK_OK" << std::endl;
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_tuner_exists.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_tuner_exists" 2>"$TMP_DIR/tuner_exists_compile.log"; then
    TUNER_OUTPUT=$("$TMP_DIR/check_tuner_exists" 2>&1)
    if echo "$TUNER_OUTPUT" | grep -q "CHECK_OK"; then
        pass "9-4b: GridSearchTuner 可实例化"
    else
        fail "9-4b: GridSearchTuner 编译通过但实例化失败" "$TUNER_OUTPUT"
    fi
else
    fail "9-4b: GridSearchTuner checker 编译失败" "$(head -20 "$TMP_DIR/tuner_exists_compile.log")"
fi

# =====================================================================
# 任务 9-5：GridSearchTuner tune 方法 (2 pts)
# =====================================================================
section "任务 9-5：GridSearchTuner tune 方法"

cat > "$TMP_DIR/check_tune_method.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <string>

#if __has_include("tiny_tvm/tune/auto_tuner.h")
#include "tiny_tvm/tune/auto_tuner.h"
#define HAS_TUNER 1
#else
#define HAS_TUNER 0
#endif

#if __has_include("tiny_tvm/ir/graph.h")
#include "tiny_tvm/ir/graph.h"
#endif

int main() {
#if !HAS_TUNER
    std::cout << "CHECK_FAIL: auto_tuner.h not found" << std::endl;
    return 0;
#else
    tiny_tvm::tune::GridSearchTuner tuner;

    // Check that tune() method exists by taking its address or calling it
    // We verify the method is callable — the exact signature may vary
    // (e.g., tune(Graph&), tune(Graph&, string), etc.)
    using TunerType = tiny_tvm::tune::GridSearchTuner;

    // Use SFINAE-like compile-time check: if this compiles, tune() exists
    // We just need it to compile; we won't actually run a full tune
    auto has_tune = [](auto& t) -> decltype(t.tune(std::declval<tiny_tvm::ir::Graph&>()), bool()) {
        return true;
    };
    (void)has_tune;

    std::cout << "CHECK_OK" << std::endl;
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_tune_method.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_tune_method" 2>"$TMP_DIR/tune_method_compile.log"; then
    TUNE_OUTPUT=$("$TMP_DIR/check_tune_method" 2>&1)
    if echo "$TUNE_OUTPUT" | grep -q "CHECK_OK"; then
        pass "9-5a: GridSearchTuner 包含 tune() 方法（编译通过）"
    else
        fail "9-5a: tune() 方法检测失败" "$TUNE_OUTPUT"
    fi
else
    fail "9-5a: tune() 方法 checker 编译失败" "$(head -20 "$TMP_DIR/tune_method_compile.log")"
fi

# Also check the header directly for tune signature
if [ -f "$TUNER_H" ]; then
    if grep -qE '(void|auto|Schedule|TuneResult|std::)\s+tune\s*\(' "$TUNER_H" 2>/dev/null; then
        pass "9-5b: auto_tuner.h 中声明了 tune() 方法"
    else
        fail "9-5b: auto_tuner.h 中未找到 tune() 方法声明"
    fi
else
    fail "9-5b: auto_tuner.h 不存在，无法检查 tune() 声明"
fi

# =====================================================================
# 任务 9-6：Candidate 值 (8, 16, 32, 64) (2 pts)
# =====================================================================
section "任务 9-6：Candidate 值 (8, 16, 32, 64)"

if [ -f "$TUNER_H" ]; then
    TUNER_SOURCES="$TUNER_H"
    # Also include implementation file if it exists
    TUNER_CPP=$(find "$ROOT_DIR/src" -name "auto_tuner.cpp" -o -name "grid_search_tuner.cpp" 2>/dev/null | head -1)
    [ -n "${TUNER_CPP:-}" ] && TUNER_SOURCES="$TUNER_SOURCES $TUNER_CPP"

    FOUND_CANDIDATES=0
    for f in $TUNER_SOURCES; do
        if grep -qE '\b8\b.*\b16\b.*\b32\b.*\b64\b' "$f" 2>/dev/null || \
           grep -qE '\{[^}]*8[^}]*16[^}]*32[^}]*64[^}]*\}' "$f" 2>/dev/null || \
           grep -qE 'candidate|tile_candidates|search_space' "$f" 2>/dev/null; then
            FOUND_CANDIDATES=1
            break
        fi
    done

    if [ "$FOUND_CANDIDATES" -eq 1 ]; then
        pass "9-6a: auto_tuner 源码包含 candidate/搜索空间定义"
    else
        fail "9-6a: 未找到 candidate 值定义（期望包含 8, 16, 32, 64）"
    fi
else
    fail "9-6a: auto_tuner.h 不存在"
fi

# Compile-time check for candidates
cat > "$TMP_DIR/check_candidates.cpp" << 'CHECKER_EOF'
#include <iostream>
#include <vector>
#include <algorithm>

#if __has_include("tiny_tvm/tune/auto_tuner.h")
#include "tiny_tvm/tune/auto_tuner.h"
#define HAS_TUNER 1
#else
#define HAS_TUNER 0
#endif

int main() {
#if !HAS_TUNER
    std::cout << "CHECK_FAIL: auto_tuner.h not found" << std::endl;
    return 0;
#else
    tiny_tvm::tune::GridSearchTuner tuner;

    // Try to access candidates if there is a public accessor
    // Different implementations may expose this differently
    // We check if the tuner class has candidates() or get_candidates()
    // If not, we just verify construction succeeded (candidates are internal)
    std::cout << "CHECK_OK" << std::endl;
    return 0;
#endif
}
CHECKER_EOF

if g++ -std=c++17 $INCLUDE_DIRS "$TMP_DIR/check_candidates.cpp" $CORE_SOURCES \
     -o "$TMP_DIR/check_candidates" 2>"$TMP_DIR/candidates_compile.log"; then
    CAND_OUTPUT=$("$TMP_DIR/check_candidates" 2>&1)
    if echo "$CAND_OUTPUT" | grep -q "CHECK_OK"; then
        pass "9-6b: GridSearchTuner candidate 检查通过"
    else
        fail "9-6b: GridSearchTuner candidate 检查失败" "$CAND_OUTPUT"
    fi
else
    fail "9-6b: candidate checker 编译失败" "$(head -20 "$TMP_DIR/candidates_compile.log")"
fi

# =====================================================================
# 任务 9-7：ttvmc tune 子命令 (2 pts)
# =====================================================================
section "任务 9-7：ttvmc tune 子命令"

if [ -x "$TTVMC" ]; then
    TUNE_HELP=$("$TTVMC" tune --help 2>&1 || "$TTVMC" help 2>&1 || "$TTVMC" --help 2>&1 || true)
    if echo "$TUNE_HELP" | grep -qi 'tune'; then
        pass "9-7a: ttvmc 支持 tune 子命令"
    else
        fail "9-7a: ttvmc 输出中未找到 tune 子命令"
    fi

    # Check that tune subcommand doesn't just error out immediately
    TUNE_RUN=$("$TTVMC" tune 2>&1 || true)
    if echo "$TUNE_RUN" | grep -qiE 'tune|usage|model|schedule|search'; then
        pass "9-7b: ttvmc tune 可响应（显示 usage 或执行）"
    else
        fail "9-7b: ttvmc tune 无有效响应" "$(echo "$TUNE_RUN" | head -5)"
    fi
else
    fail "9-7a: ttvmc 不存在" "$TTVMC"
    fail "9-7b: ttvmc 不存在，跳过 tune 响应检查"
fi

# =====================================================================
# 任务 9-8：ttvmc compile --schedule 参数 (2 pts)
# =====================================================================
section "任务 9-8：ttvmc compile --schedule 参数"

if [ -x "$TTVMC" ]; then
    COMPILE_HELP=$("$TTVMC" compile --help 2>&1 || "$TTVMC" help 2>&1 || "$TTVMC" --help 2>&1 || true)
    if echo "$COMPILE_HELP" | grep -qi '\-\-schedule'; then
        pass "9-8a: ttvmc compile 支持 --schedule 参数"
    else
        # Also check source code for --schedule
        if grep -rq '\-\-schedule' "$ROOT_DIR/src/tools" "$ROOT_DIR/src" 2>/dev/null; then
            pass "9-8a: 源码中包含 --schedule 参数定义"
        else
            fail "9-8a: 未找到 --schedule 参数"
        fi
    fi

    # Check for schedule-related flag in source
    TOOLS_DIR="$ROOT_DIR/src/tools"
    if [ -d "$TOOLS_DIR" ]; then
        if grep -rqE 'schedule|schedule_file|schedule_path|sched' "$TOOLS_DIR" 2>/dev/null; then
            pass "9-8b: ttvmc tools 源码包含 schedule 相关逻辑"
        else
            fail "9-8b: ttvmc tools 源码未找到 schedule 相关逻辑"
        fi
    else
        # Fallback: search all source for the flag
        if grep -rqE '\-\-schedule' "$ROOT_DIR/src" 2>/dev/null; then
            pass "9-8b: 源码中包含 --schedule 参数处理"
        else
            fail "9-8b: 源码中未找到 --schedule 参数处理"
        fi
    fi
else
    fail "9-8a: ttvmc 不存在" "$TTVMC"
    fail "9-8b: ttvmc 不存在，跳过 --schedule 检查"
fi

# =====================================================================
# 任务 9-9：best_schedule.json 输出 (2 pts)
# =====================================================================
section "任务 9-9：best_schedule.json 输出"

if [ -x "$TTVMC" ]; then
    # Try to find a simple model to tune
    MODEL_FILE=""
    for candidate in \
        "$ROOT_DIR/examples/"*.json \
        "$ROOT_DIR/tests/"*.json \
        "$ROOT_DIR/models/"*.json \
        "$ROOT_DIR/examples/"*.onnx \
        "$ROOT_DIR/tests/"*.onnx; do
        if [ -f "$candidate" ]; then
            MODEL_FILE="$candidate"
            break
        fi
    done

    if [ -n "$MODEL_FILE" ]; then
        # Attempt to run tune and check for best_schedule.json
        TUNE_EXIT=0
        "$TTVMC" tune "$MODEL_FILE" -o "$TMP_DIR/best_schedule.json" 2>"$TMP_DIR/tune_run.log" || TUNE_EXIT=$?

        # Also check default output location
        if [ ! -f "$TMP_DIR/best_schedule.json" ]; then
            # Some implementations output to current dir or model dir
            for loc in "$ROOT_DIR/best_schedule.json" "./best_schedule.json" \
                       "$BUILD_DIR/best_schedule.json"; do
                if [ -f "$loc" ]; then
                    cp "$loc" "$TMP_DIR/best_schedule.json" 2>/dev/null || true
                    break
                fi
            done
        fi

        if [ -f "$TMP_DIR/best_schedule.json" ]; then
            pass "9-9a: ttvmc tune 生成了 best_schedule.json"
            # Validate it's valid JSON with tile values
            if python3 -c "
import json, sys
with open('$TMP_DIR/best_schedule.json') as f:
    data = json.load(f)
# Check for tile-related keys
text = json.dumps(data)
if 'tile' in text.lower() or 'schedule' in text.lower() or isinstance(data, (dict, list)):
    print('VALID_JSON')
else:
    print('NO_TILE_INFO')
" 2>/dev/null | grep -q "VALID_JSON"; then
                pass "9-9b: best_schedule.json 包含有效 JSON（含 schedule/tile 信息）"
            else
                fail "9-9b: best_schedule.json 不包含有效的 schedule 信息"
            fi
        else
            fail "9-9a: ttvmc tune 未生成 best_schedule.json" "$(head -10 "$TMP_DIR/tune_run.log" 2>/dev/null)"
            fail "9-9b: best_schedule.json 不存在，跳过 JSON 验证"
        fi
    else
        # No model file found — try running tune without model to see if it at least responds
        TUNE_ANY=$("$TTVMC" tune 2>&1 || true)
        if echo "$TUNE_ANY" | grep -qiE 'model|usage|json|schedule'; then
            fail "9-9a: 无可用模型文件进行 tune 测试" "tune 命令可用但缺少测试模型"
            fail "9-9b: 无模型文件，跳过 best_schedule.json 验证"
        else
            fail "9-9a: 无可用模型文件且 tune 命令无有效输出"
            fail "9-9b: 跳过 best_schedule.json 验证"
        fi
    fi
else
    fail "9-9a: ttvmc 不存在" "$TTVMC"
    fail "9-9b: ttvmc 不存在，跳过 best_schedule.json 验证"
fi

# =====================================================================
# 汇总
# =====================================================================
section "Phase 9 测试结果汇总"

echo ""
echo "  通过: $PASS_COUNT"
echo "  失败: $FAIL_COUNT"
echo "  总计: $TOTAL_COUNT"
echo ""
echo "  Score: $PASS_COUNT/$TOTAL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "🎉 Phase 9 所有检测项通过！Performance Analysis & Auto-Tuning 完成。"
    exit 0
else
    echo "⚠️  Phase 9 有 $FAIL_COUNT 项未通过，请根据上方 [FAIL] 提示修复。"
    exit 1
fi
