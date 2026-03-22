#include "tiny_tvm/frontend/onnx/onnx_frontend.h"

#include <fstream>

namespace tiny_tvm::frontend::onnx {

ImportResult load_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return ImportResult{false, {}, "Failed to open ONNX model file: " + path};
    }

    return ImportResult{
        false,
        {},
        "ONNX frontend skeleton is in place, but importing is not implemented yet.",
    };
}

}  // namespace tiny_tvm::frontend::onnx
