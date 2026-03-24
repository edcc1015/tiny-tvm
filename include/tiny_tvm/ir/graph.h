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

inline DType str_to_dtype(const std::string& str) {
    static const std::unordered_map<std::string, DType> table = {
        {"float32", DType::kFloat32},
        {"f32",     DType::kFloat32},
        {"int32",   DType::kInt32},
        {"i32",     DType::kInt32},
    };

    auto it = table.find(str);
    if (it != table.end()) {
        return it->second;
    }
    return DType::kUnknown;
}

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
    std::size_t param_offset = 0;  /* constant in params.bin */
    std::vector<uint8_t> data;
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

    /* return tensor and op by index */
    Tensor& tensor(int index);
    const Tensor& tensor(int index) const;

    Op& op(int index);
    const Op& op(int index) const;

    /* TODO */
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

inline int64_t num_elements(const Tensor& t) {
    int64_t n = 1;
    for (auto d : t.shape) n *= d;
    return n;
}

inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace tiny_tvm::ir
