#include "tiny_tvm/pass/pass.h"

namespace tiny_tvm::passes {

void InferShapePass::run(ir::Graph& graph) {
    for (size_t i = 0; i < graph.op_count(); i++) {
        auto& op = graph.op(i);
        if (op.kind == "MatMul") {
            auto& lhs = graph.tensor(op.inputs[0]);
            auto& rhs = graph.tensor(op.inputs[1]);
            TTVM_CHECK(lhs.shape[1] == rhs.shape[0]);
            
            auto& out = graph.tensor(op.outputs[0]);
            out.shape = {lhs.shape[0], rhs.shape[1]};
            out.dtype = lhs.dtype;
        } else if (op.kind == "Add") {
            auto& a = graph.tensor(op.inputs[0]);
            auto& b = graph.tensor(op.inputs[1]);
            TTVM_CHECK(a.shape == b.shape);
            auto& out = graph.tensor(op.outputs[0]);
            out.shape = a.shape;
            out.dtype = a.dtype;
        } else if (op.kind == "Relu") {
            auto& in = graph.tensor(op.inputs[0]);
            auto& out = graph.tensor(op.outputs[0]);
            out.shape = in.shape;
            out.dtype = in.dtype;
        }
    }
}

}  // namespace tiny_tvm::passes