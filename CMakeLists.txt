cmake_minimum_required(VERSION 3.16)

project(tiny_tvm VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

option(TINY_TVM_BUILD_TESTS "Build tiny_tvm tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TINY_TVM_CORE_SOURCES
    src/codegen/c_codegen.cpp
    src/frontend/json/json_frontend.cpp
    src/frontend/onnx/onnx_frontend.cpp
    src/ir/graph.cpp
    src/pass/pass_manager.cpp
    src/runtime/runtime.cpp
)

add_library(tiny_tvm_core STATIC ${TINY_TVM_CORE_SOURCES})
target_include_directories(tiny_tvm_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(tiny_tvm_core PUBLIC cxx_std_17)

add_executable(ttvmc src/tools/ttvmc.cpp)
target_link_libraries(ttvmc PRIVATE tiny_tvm_core)

if(BUILD_TESTING AND TINY_TVM_BUILD_TESTS)
    add_executable(graph_smoke_test tests/ir/graph_smoke_test.cpp)
    target_link_libraries(graph_smoke_test PRIVATE tiny_tvm_core)
    add_test(NAME graph_smoke_test COMMAND graph_smoke_test)
endif()
