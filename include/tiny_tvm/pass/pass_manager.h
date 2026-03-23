#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tiny_tvm/pass/pass.h"

namespace tiny_tvm::passes {

class PassManager {
public:
    void add(std::unique_ptr<Pass> pass);
    void run(ir::Graph& graph) const;

    std::size_t size() const noexcept;
    std::vector<std::string> pass_names() const;

private:
    std::vector<std::unique_ptr<Pass>> passes_;
    bool verbose_ = false;
};

}  // namespace tiny_tvm::passes
