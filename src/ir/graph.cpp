#include "tiny_tvm/ir/graph.h"

#include <sstream>
#include <utility>

namespace tiny_tvm::ir {

int Graph::add_tensor(const Tensor& tensor) {
    tensors_.push_back(tensor);
    return static_cast<int>(tensors_.size() - 1);
}

int Graph::add_tensor(Tensor&& tensor) {
    tensors_.push_back(std::move(tensor));
    return static_cast<int>(tensors_.size() - 1);
}

int Graph::add_op(const Op& op) {
    ops_.push_back(op);
    return static_cast<int>(ops_.size() - 1);
}

int Graph::add_op(Op&& op) {
    ops_.push_back(std::move(op));
    return static_cast<int>(ops_.size() - 1);
}

Tensor& Graph::tensor(int index) {
    return tensors_.at(static_cast<std::size_t>(index));
}

const Tensor& Graph::tensor(int index) const {
    return tensors_.at(static_cast<std::size_t>(index));
}

Op& Graph::op(int index) {
    return ops_.at(static_cast<std::size_t>(index));
}

const Op& Graph::op(int index) const {
    return ops_.at(static_cast<std::size_t>(index));
}

const std::vector<Tensor>& Graph::tensors() const noexcept {
    return tensors_;
}

std::vector<Tensor>& Graph::mutable_tensors() noexcept {
    return tensors_;
}

const std::vector<Op>& Graph::ops() const noexcept {
    return ops_;
}

std::vector<Op>& Graph::mutable_ops() noexcept {
    return ops_;
}

std::vector<int>& Graph::graph_inputs() noexcept {
    return graph_inputs_;
}

const std::vector<int>& Graph::graph_inputs() const noexcept {
    return graph_inputs_;
}

std::vector<int>& Graph::graph_outputs() noexcept {
    return graph_outputs_;
}

const std::vector<int>& Graph::graph_outputs() const noexcept {
    return graph_outputs_;
}

std::size_t Graph::tensor_count() const noexcept {
    return tensors_.size();
}

std::size_t Graph::op_count() const noexcept {
    return ops_.size();
}

bool Graph::empty() const noexcept {
    return tensors_.empty() && ops_.empty();
}

std::string Graph::summary() const {
    std::ostringstream stream;
    stream << "Graph(tensors=" << tensors_.size()
           << ", ops=" << ops_.size()
           << ", inputs=" << graph_inputs_.size()
           << ", outputs=" << graph_outputs_.size() << ")";
    return stream.str();
}

}  // namespace tiny_tvm::ir
