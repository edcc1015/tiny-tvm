#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "tiny_tvm/schedule/schedule.h"

namespace tiny_tvm::ir {

enum class DType {
    kUnknown,
    kFloat32,
    kInt32,
};

inline std::size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::kFloat32:
        case DType::kInt32:
            return 4;
        case DType::kUnknown:
        default:
            return 0;
    }
}

using AttrValue = std::variant<std::int64_t, double, std::string, std::vector<std::int64_t>>;
using AttrMap = std::unordered_map<std::string, AttrValue>;

struct Tensor {
    std::string name;
    std::vector<std::int64_t> shape;
    DType dtype = DType::kUnknown;
    std::size_t nbytes = 0;
    std::size_t offset = 0;
    bool is_constant = false;
};

struct Op {
    std::string kind;
    std::vector<int> inputs;
    std::vector<int> outputs;
    AttrMap attrs;
    schedule::Schedule schedule;
};

class Graph {
public:
    int add_tensor(const Tensor& tensor);
    int add_tensor(Tensor&& tensor);
    int add_op(const Op& op);
    int add_op(Op&& op);

    Tensor& tensor(int index);
    const Tensor& tensor(int index) const;
    Op& op(int index);
    const Op& op(int index) const;

    const std::vector<Tensor>& tensors() const noexcept;
    std::vector<Tensor>& mutable_tensors() noexcept;
    const std::vector<Op>& ops() const noexcept;
    std::vector<Op>& mutable_ops() noexcept;

    std::vector<int>& graph_inputs() noexcept;
    const std::vector<int>& graph_inputs() const noexcept;
    std::vector<int>& graph_outputs() noexcept;
    const std::vector<int>& graph_outputs() const noexcept;

    std::size_t tensor_count() const noexcept;
    std::size_t op_count() const noexcept;
    bool empty() const noexcept;
    std::string summary() const;

private:
    std::vector<Tensor> tensors_;
    std::vector<Op> ops_;
    std::vector<int> graph_inputs_;
    std::vector<int> graph_outputs_;
};

}  // namespace tiny_tvm::ir
