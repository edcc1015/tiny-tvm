#pragma once

namespace tiny_tvm::schedule {

struct Schedule {
    int tile_m = -1;
    int tile_n = -1;
    int tile_k = -1;
    bool reorder_ikj = false;
    bool unroll_inner = false;

    bool is_default() const noexcept {
        return tile_m < 0 && tile_n < 0 && tile_k < 0 && !reorder_ikj && !unroll_inner;
    }
};

}  // namespace tiny_tvm::schedule
