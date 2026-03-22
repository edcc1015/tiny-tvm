#include "tiny_tvm/pass/pass_manager.h"

#include <utility>

namespace tiny_tvm::passes {

void PassManager::add(std::unique_ptr<Pass> pass) {
    passes_.push_back(std::move(pass));
}

void PassManager::run(ir::Graph& graph) const {
    for (const auto& pass : passes_) {
        pass->run(graph);
    }
}

std::size_t PassManager::size() const noexcept {
    return passes_.size();
}

std::vector<std::string> PassManager::pass_names() const {
    std::vector<std::string> names;
    names.reserve(passes_.size());
    for (const auto& pass : passes_) {
        names.push_back(pass->name());
    }
    return names;
}

}  // namespace tiny_tvm::passes
