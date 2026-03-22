#pragma once

#include <string>

#include "tiny_tvm/ir/graph.h"

namespace tiny_tvm::codegen {

std::string emit_c_module(const ir::Graph& graph);

}  // namespace tiny_tvm::codegen
