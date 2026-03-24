
#include <iostream>

#define TTVM_CHECK(cond) \
    if (!(cond)) { \
        std::cerr << "Check failed: " #cond \
                  << " at " << __FILE__ << ":" << __LINE__ \
                  << std::endl; \
        std::abort(); \
    }

namespace tiny_tvm {



}  // namespace tiny_tvm