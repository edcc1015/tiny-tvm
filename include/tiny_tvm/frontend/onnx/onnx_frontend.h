#pragma once

#include <string>

#include "tiny_tvm/ir/graph.h"

namespace tiny_tvm::frontend::onnx {

struct ImportResult {
    bool ok = false;
    ir::Graph graph;
    std::string message;
};

ImportResult load_file(const std::string& path);

}  // namespace tiny_tvm::frontend::onnx
