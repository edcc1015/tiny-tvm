#include "tiny_tvm/codegen/c_codegen.h"

#include <sstream>

namespace tiny_tvm::codegen {

std::string emit_c_module(const ir::Graph& graph) {
    std::ostringstream stream;
    stream << "/* Tiny TVM generated C skeleton */\n";
    stream << "/* " << graph.summary() << " */\n";
    stream << "#include <stddef.h>\n\n";
    stream << "void tiny_tvm_run(float* input, float* output) {\n";
    stream << "    (void)input;\n";
    stream << "    (void)output;\n";
    stream << "    /* TODO: emit loops from Graph + Schedule. */\n";
    stream << "}\n";
    return stream.str();
}

}  // namespace tiny_tvm::codegen
