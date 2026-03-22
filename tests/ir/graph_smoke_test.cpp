#include <iostream>
#include <memory>
#include <string>

#include "tiny_tvm/codegen/c_codegen.h"
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass.h"
#include "tiny_tvm/pass/pass_manager.h"

int main() {
    using tiny_tvm::ir::DType;
    using tiny_tvm::ir::Graph;
    using tiny_tvm::ir::Op;
    using tiny_tvm::ir::Tensor;

    Graph graph;

    Tensor input;
    input.name = "input";
    input.shape = {1, 4};
    input.dtype = DType::kFloat32;
    input.nbytes = 16;

    Tensor weight;
    weight.name = "weight";
    weight.shape = {4, 4};
    weight.dtype = DType::kFloat32;
    weight.nbytes = 64;
    weight.is_constant = true;

    Tensor output;
    output.name = "output";
    output.shape = {1, 4};
    output.dtype = DType::kFloat32;
    output.nbytes = 16;

    const int input_id = graph.add_tensor(std::move(input));
    const int weight_id = graph.add_tensor(std::move(weight));
    const int output_id = graph.add_tensor(std::move(output));

    Op op;
    op.kind = "MatMul";
    op.inputs = {input_id, weight_id};
    op.outputs = {output_id};
    graph.add_op(std::move(op));

    graph.graph_inputs().push_back(input_id);
    graph.graph_outputs().push_back(output_id);

    tiny_tvm::passes::PassManager pass_manager;
    pass_manager.add(std::make_unique<tiny_tvm::passes::NoOpPass>());
    pass_manager.run(graph);

    if (graph.tensor_count() != 3 || graph.op_count() != 1) {
        std::cerr << "Unexpected graph shape: " << graph.summary() << '\n';
        return 1;
    }

    const std::string generated = tiny_tvm::codegen::emit_c_module(graph);
    if (generated.find("tiny_tvm_run") == std::string::npos) {
        std::cerr << "Generated C skeleton does not contain runtime entry point.\n";
        return 1;
    }

    return 0;
}
