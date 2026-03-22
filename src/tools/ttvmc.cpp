#include <iostream>
#include <memory>
#include <string>

#include "tiny_tvm/codegen/c_codegen.h"
#include "tiny_tvm/ir/graph.h"
#include "tiny_tvm/pass/pass.h"
#include "tiny_tvm/pass/pass_manager.h"
#include "tiny_tvm/runtime/runtime.h"
#include "tiny_tvm/support/version.h"

namespace {

tiny_tvm::ir::Graph make_demo_graph() {
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

    Op matmul;
    matmul.kind = "MatMul";
    matmul.inputs = {input_id, weight_id};
    matmul.outputs = {output_id};
    graph.add_op(std::move(matmul));

    graph.graph_inputs().push_back(input_id);
    graph.graph_outputs().push_back(output_id);
    return graph;
}

void print_usage(const char* argv0) {
    std::cout << "Tiny TVM CLI skeleton\n"
              << "Usage:\n"
              << "  " << argv0 << " help\n"
              << "  " << argv0 << " version\n"
              << "  " << argv0 << " smoke\n"
              << "  " << argv0 << " compile <model>\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    const std::string command = argv[1];
    if (command == "help" || command == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    if (command == "version" || command == "--version") {
        std::cout << tiny_tvm::support::kVersion << '\n';
        return 0;
    }

    if (command == "smoke") {
        tiny_tvm::ir::Graph graph = make_demo_graph();
        tiny_tvm::passes::PassManager pass_manager;
        pass_manager.add(std::make_unique<tiny_tvm::passes::NoOpPass>());
        pass_manager.run(graph);

        const tiny_tvm::runtime::Runtime runtime(1024);
        std::cout << graph.summary() << '\n';
        std::cout << runtime.describe() << '\n';
        std::cout << tiny_tvm::codegen::emit_c_module(graph);
        return 0;
    }

    if (command == "compile") {
        std::cerr << "compile pipeline is not implemented yet; start with phase 1 tasks in docs/phase_task_checklist.md.\n";
        return 1;
    }

    std::cerr << "Unknown command: " << command << '\n';
    print_usage(argv[0]);
    return 1;
}
