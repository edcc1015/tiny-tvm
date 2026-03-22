#include "tiny_tvm/runtime/runtime.h"

#include <string>

namespace tiny_tvm::runtime {

Runtime::Runtime(std::size_t workspace_bytes) : workspace_bytes_(workspace_bytes) {}

std::size_t Runtime::workspace_bytes() const noexcept {
    return workspace_bytes_;
}

std::string Runtime::describe() const {
    return "Runtime(workspace_bytes=" + std::to_string(workspace_bytes_) + ")";
}

}  // namespace tiny_tvm::runtime
