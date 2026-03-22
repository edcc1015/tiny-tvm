#pragma once

#include <string>

#include "tiny_tvm/ir/graph.h"

namespace tiny_tvm::frontend::json {

struct ParseResult {
    bool ok = false;
    ir::Graph graph;
    std::string message;
};

ParseResult parse_text(const std::string& text);
ParseResult load_file(const std::string& path);

}  // namespace tiny_tvm::frontend::json
