#include "tiny_tvm/frontend/json/json_frontend.h"

#include <fstream>
#include <iterator>
#include <vector>

#include "third_party/nlohmann/json.hpp"
#include "include/tiny_tvm/ir/graph.h"

namespace tiny_tvm::frontend::json {

ParseResult parse_text(const std::string& text) {
    auto j = nlohmann::json::parse(text);
    tiny_tvm::ir::Graph graph;

    /* parse tensor */
    for (auto& jt : j["tensors"]) {
        tiny_tvm::ir::Tensor t;
        t.name = jt["name"];
        t.shape = jt["shape"].get<std::vector<int64_t>>();
        t.dtype = tiny_tvm::ir::str_to_dtype(jt["dtype"]);
        t.is_constant = jt.value("is_constant", false);
        if (t.is_constant && jt.contains("data")) {
            /* json array -> raw bytes */
            auto floats = jt["data"].get<std::vector<float>>();
            t.data.resize(floats.size() * sizeof(float));
            memcpy(t.data.data(), floats.data(), t.data.size());
        }
        graph.add_tensor(std::move(t));
    }

    /* parse op */
    for (auto& jo : j["ops"]) {
        tiny_tvm::ir::Op op;
        op.kind = jo["kind"];
        op.inputs = jo["inputs"].get<std::vector<int>>();
        op.outputs = jo["outputs"].get<std::vector<int>>();
        graph.add_op(std::move(op));
    }

    graph.graph_inputs() = j["graph_inputs"].get<std::vector<int>>();
    graph.graph_outputs() = j["graph_outputs"].get<std::vector<int>>();

    return {true, std::move(graph), ""};
}

ParseResult load_file(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        return ParseResult{false, {}, "Failed to open JSON model file: " + path};
    }

    const std::string contents((std::istreambuf_iterator<char>(stream)),
                               std::istreambuf_iterator<char>());
    return parse_text(contents);
}

}  // namespace tiny_tvm::frontend::json
