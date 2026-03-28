#include "tiny_tvm/pass/pass.h"

namespace tiny_tvm::passes {

void InitSchedulePass::run(ir::Graph& graph) {
    for (size_t i = 0; i < graph.op_count(); i++) {
        auto& op = graph.op(i);
        op.schedule = schedule::Schedule{};
    }
}

}  // namespace tiny_tvm::passes