#pragma once

#include <string>

#include "tiny_tvm/ir/graph.h"

namespace tiny_tvm::passes {

class Pass {
public:
    virtual ~Pass() = default;
    virtual std::string name() const = 0;
    virtual void run(ir::Graph& graph) = 0;
};

class NoOpPass final : public Pass {
public:
    std::string name() const override {
        return "NoOpPass";
    }

    void run(ir::Graph& graph) override {
        (void)graph;
    }
};

}  // namespace tiny_tvm::passes
