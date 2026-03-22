#include "tiny_tvm/frontend/json/json_frontend.h"

#include <fstream>
#include <iterator>

namespace tiny_tvm::frontend::json {

ParseResult parse_text(const std::string& text) {
    ParseResult result;
    if (text.empty()) {
        result.message = "JSON input is empty.";
        return result;
    }

    result.message = "JSON frontend skeleton is in place, but parsing is not implemented yet.";
    return result;
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
