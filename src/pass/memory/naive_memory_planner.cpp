#include "tiny_tvm/pass/pass.h"

namespace tiny_tvm::passes {

void NaiveMemoryPlanner::run(ir::Graph& graph) {
    size_t ws_offset = 0;
    size_t param_offset = 0;

    for (size_t i = 0; i < graph.tensor_count(); i++) {
        auto& t = graph.tensor(i);
        t.nbytes = ir::num_elements(t) * ir::dtype_size(t.dtype);

        if (t.is_constant) {
            t.param_offset = ir::align_up(param_offset, 64);
            param_offset = t.param_offset + t.nbytes;  // param offset
        } else {
            t.offset = ir::align_up(ws_offset, 64); // workspace offset
            ws_offset = t.offset + t.nbytes;
        }
    }
}

}  // namespace tiny_tvm::passes