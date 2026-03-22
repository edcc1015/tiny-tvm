#pragma once

#include <cstddef>
#include <string>

namespace tiny_tvm::runtime {

class Runtime {
public:
    explicit Runtime(std::size_t workspace_bytes = 0);

    std::size_t workspace_bytes() const noexcept;
    std::string describe() const;

private:
    std::size_t workspace_bytes_ = 0;
};

}  // namespace tiny_tvm::runtime
