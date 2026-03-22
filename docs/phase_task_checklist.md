# Tiny TVM 阶段任务与实现指南（无 BRAM / 只负责写代码版）

这份文档不是泛泛的计划，而是给你一份**按阶段推进、按文件落地、按方法实现**的执行说明。  
你不需要再自己拆任务，直接照着本文件补代码即可。

当前仓库已经有一版 C++17 工程骨架，因此本文件默认你是在**继续完善现有骨架**，而不是从零重新建工程。

---

## 1. 全局实现约定

在进入各阶段之前，先固定几条全局约束。后面所有实现都按这些约定来，不要中途改风格。

### 1.1 统一目录与职责

- `include/tiny_tvm/`：头文件，放对外类型和接口
- `src/`：实现文件
- `tests/`：单元测试和端到端测试
- `examples/`：输入模型样例
- `docs/`：设计和执行文档
- `scripts/`：构建、运行、部署脚本
- `third_party/`：外部依赖源码或单头文件

### 1.2 统一数据结构约定

从阶段 0 开始，就按最终可扩展的方向定义核心结构，避免后面大改。

#### Tensor 最终建议字段

```cpp
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype = DType::kUnknown;
    size_t nbytes = 0;
    size_t offset = 0;        // workspace offset，非 constant tensor 使用
    size_t param_offset = 0;  // params.bin offset，constant tensor 使用
    bool is_constant = false;
    std::vector<uint8_t> data;
};
```

说明：

- `offset` 只给激活张量使用
- `param_offset` 只给常量张量使用
- `data` 只在编译期保存常量内容，runtime 不依赖它

#### Op 最终建议字段

```cpp
struct Op {
    std::string kind;
    std::vector<int> inputs;
    std::vector<int> outputs;
    AttrMap attrs;
    schedule::Schedule schedule;
};
```

说明：

- `inputs` / `outputs` 一律存 tensor index，不存指针
- `attrs` 只存算子参数，不存分析结果
- `schedule` 先服务于 `MatMul`，后面再逐步扩展

#### Graph 最终建议字段

```cpp
struct Graph {
    std::vector<Tensor> tensors;
    std::vector<Op> ops;
    std::vector<int> graph_inputs;
    std::vector<int> graph_outputs;
};
```

### 1.3 统一图约定

- `ops` 默认按拓扑顺序存放
- `graph_inputs` 只放外部输入，不放常量
- `graph_outputs` 放模型输出 tensor
- 任何 Pass 都直接修改 `Graph`
- 在没有引入 AnalysisManager 之前，分析结果直接回写到 `Tensor` 或局部辅助结构中即可

### 1.4 统一编译输出目录

从阶段 1 开始，固定编译输出目录格式：

```text
out/<model_name>/
├── graph.json
├── params.bin
├── deploy.c
├── libdeploy.so
└── compile_report.json   # 阶段3开始补齐
```

### 1.5 统一生成代码接口

生成的 C 代码从阶段 1 开始统一暴露这两个符号：

```c
size_t tiny_tvm_workspace_size(void);
int tiny_tvm_run(const void* params,
                 void* workspace,
                 const void* const* inputs,
                 void* const* outputs);
```

原因：

- 可以兼容多输入/多输出
- runtime 侧不需要关心具体算子细节
- 后面上 ARM 时接口不用重改

### 1.6 统一参数文件格式

不要设计复杂二进制协议。这个项目以练手为主，参数格式固定为：

- `params.bin`：所有 constant tensor 的原始二进制内容顺序拼接
- `graph.json`：记录每个 constant tensor 的 `param_offset`、`nbytes`、`dtype`、`shape`

也就是说：**二进制里只放纯数据，元信息都放 `graph.json`**。

### 1.7 统一错误处理策略

- Frontend 解析失败：返回结构化错误信息，不要 silent fail
- Pass 发现非法图：直接报错，说明具体 op / tensor
- Runtime 加载失败：打印缺失文件或符号名
- CLI 参数不合法：打印 usage 并返回非 0

### 1.8 统一阶段推进规则

每一阶段都遵守：

1. 先把本阶段最小闭环做通
2. 再补测试
3. 再进入下一阶段

不要同时并行做多个阶段。

---

## 2. 阶段 0：工程骨架完善

### 2.1 本阶段目标

把当前已有骨架补成一个真正可持续开发的基础工程。阶段 0 完成后，你应该拥有：

- 稳定的 IR 结构
- 稳定的 Pass 框架
- 稳定的 CLI 入口
- 至少 1 个 smoke test
- 稳定的构建入口

### 2.2 本阶段要重点改的文件

- `CMakeLists.txt`
- `include/tiny_tvm/ir/graph.h`
- `src/ir/graph.cpp`
- `include/tiny_tvm/pass/pass.h`
- `include/tiny_tvm/pass/pass_manager.h`
- `src/pass/pass_manager.cpp`
- `include/tiny_tvm/schedule/schedule.h`
- `src/tools/ttvmc.cpp`
- `tests/ir/graph_smoke_test.cpp`
- `scripts/build_host.sh`

### 2.3 任务 0-1：把 IR 定义补到后面不需要大改

#### 要做什么

- 给 `Tensor` 补 `param_offset`、`data`
- 给 `Graph` 补完整访问接口
- 增加常用辅助函数

#### 具体实现方法

1. 在 `include/tiny_tvm/ir/graph.h` 中补齐 `Tensor` 的最终字段。
2. 增加以下辅助函数：
   - `dtype_size(DType)`
   - `num_elements(const Tensor&)`
   - `align_up(size_t value, size_t alignment)`
3. `Graph` 至少提供：
   - `add_tensor`
   - `add_op`
   - `tensor(index)`
   - `op(index)`
   - `summary()`
4. `summary()` 输出固定格式，后面 CLI 和测试直接复用。

#### 完成标准

- 后续阶段不需要再因为 Tensor/Graph 字段不够而返工
- `graph.summary()` 能稳定打印 tensor/op/input/output 数量

### 2.4 任务 0-2：把 Pass 基础设施做成可长期复用的版本

#### 要做什么

- 固定 `Pass` 基类接口
- 固定 `PassManager` 行为
- 为后面加 dump 留接口

#### 具体实现方法

1. `Pass` 保持最小接口：`name()` + `run(Graph&)`。
2. `PassManager` 先不要引入复杂注册系统，直接维护 `std::vector<std::unique_ptr<Pass>>`。
3. `PassManager::run()` 按顺序执行 pass。
4. 预留一个简单调试入口，例如：
   - `bool verbose = false`
   - verbose 时在每个 pass 前后打印 `graph.summary()`
5. 现在只保留 `NoOpPass` 作为 smoke 用例，不要过早实现真实 pass。

#### 完成标准

- 可以连续执行多个 pass
- 增加新 pass 时只需要新增类并 `add()`，不需要改框架

### 2.5 任务 0-3：固定 CLI 骨架

#### 要做什么

让 `ttvmc` 成为后面所有功能的统一入口。

#### 具体实现方法

1. 当前保留这些子命令：
   - `help`
   - `version`
   - `smoke`
   - `compile`
2. `smoke` 命令构造一个内置 demo graph，执行 `NoOpPass`，打印：
   - graph summary
   - runtime summary
   - 生成的 C 代码骨架
3. `compile` 先保留占位报错，阶段 1 再接通真实逻辑。
4. usage 文案固定，不要每个阶段都改命令格式。

#### 完成标准

- `./build/ttvmc smoke` 能作为每次重构后的快速回归命令

### 2.6 任务 0-4：把测试和构建入口固定下来

#### 要做什么

- 让工程至少有 1 个测试
- 让脚本具备依赖检查

#### 具体实现方法

1. `tests/ir/graph_smoke_test.cpp` 做最小验证：
   - 构造 graph
   - 跑 `NoOpPass`
   - 调用 `emit_c_module`
   - 断言包含 `tiny_tvm_run`
2. `scripts/build_host.sh` 先检查 `cmake` 是否存在。
3. CMake 中保留：
   - `tiny_tvm_core`
   - `ttvmc`
   - `graph_smoke_test`
4. 不要在阶段 0 引入新的外部依赖。

#### 完成标准

- `cmake` 环境下可以 build + test
- 没有 `cmake` 时脚本报错清晰

### 2.7 阶段 0 结束后你应该能做什么

- 稳定构建工程
- 跑 smoke test
- 在不破坏骨架的前提下继续开发后续功能

---

## 3. 阶段 1：最小闭环（JSON -> IR -> Pass -> Codegen -> Host）

### 3.1 本阶段目标

把最小编译链路跑通：

```text
json model -> Graph -> pass pipeline -> deploy.c -> libdeploy.so -> host runtime
```

只支持最小算子集合：

- `Input`
- `Constant`
- `MatMul`
- `Add`
- `Relu`

### 3.2 本阶段要新增或重点修改的文件

- `third_party/nlohmann/json.hpp` 或等价 JSON 依赖
- `include/tiny_tvm/frontend/json/json_frontend.h`
- `src/frontend/json/json_frontend.cpp`
- `include/tiny_tvm/pass/graph/infer_shape_pass.h`
- `src/pass/graph/infer_shape_pass.cpp`
- `include/tiny_tvm/pass/memory/naive_memory_planner.h`
- `src/pass/memory/naive_memory_planner.cpp`
- `include/tiny_tvm/pass/schedule/init_schedule_pass.h`
- `src/pass/schedule/init_schedule_pass.cpp`
- `include/tiny_tvm/codegen/c_codegen.h`
- `src/codegen/c_codegen.cpp`
- `src/tools/ttvmc.cpp`
- `src/tools/run_model.cpp`
- `tests/frontend/json_frontend_test.cpp`
- `tests/pass/infer_shape_test.cpp`
- `tests/runtime/mlp_e2e_test.cpp`
- `examples/json/mlp.json`

### 3.3 任务 1-1：固定 JSON 模型格式并实现 Frontend

#### 推荐 JSON 格式

```json
{
  "tensors": [
    {
      "name": "input",
      "shape": [1, 4],
      "dtype": "float32"
    },
    {
      "name": "weight",
      "shape": [4, 4],
      "dtype": "float32",
      "is_constant": true,
      "data": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0]
    },
    {
      "name": "output",
      "shape": [1, 4],
      "dtype": "float32"
    }
  ],
  "ops": [
    {
      "kind": "MatMul",
      "inputs": [0, 1],
      "outputs": [2]
    }
  ],
  "graph_inputs": [0],
  "graph_outputs": [2]
}
```

#### 具体实现方法

1. 直接使用 `nlohmann/json`，不要手写 JSON parser。
2. `json_frontend.cpp` 中按顺序解析：
   - tensors
   - ops
   - graph_inputs
   - graph_outputs
3. 解析 tensor 时：
   - 把 `dtype` 字符串映射到 `DType`
   - 如果 `is_constant = true`，把 `data` 转成 `std::vector<uint8_t>`
4. 解析 op 时先不做拓扑排序，只做**拓扑合法性检查**：
   - 输入 tensor index 必须合法
   - 输出 tensor index 必须合法
   - 当前 op 不能依赖“未来才会产生”的输出
5. 对 `attrs` 先支持：
   - `int64_t`
   - `double`
   - `string`
   - `vector<int64_t>`

#### 完成标准

- 能把 `examples/json/mlp.json` 成功转成内部 `Graph`
- 非法 JSON 能给出可读错误信息

### 3.4 任务 1-2：实现 `InferShapePass`

#### 要支持的规则

- `MatMul`: `[M, K] x [K, N] -> [M, N]`
- `Add`: 先只支持**完全同 shape**
- `Relu`: 输出 shape = 输入 shape
- `Constant` / `Input`: shape 来自前端

#### 具体实现方法

1. 新建 `InferShapePass`，放到 `pass/graph/`。
2. 依次遍历 `graph.ops()`，按 `op.kind` 分发。
3. 对每个 op：
   - 取输入 tensor
   - 校验 rank / dtype / shape
   - 回写输出 tensor 的 `shape` 和 `dtype`
4. `MatMul` 暂时只支持二维，不支持 batch matmul。
5. `Add` 阶段 1 不做 broadcast，直接要求两个输入 shape 完全相同。

#### 完成标准

- 对 `MatMul + Add + Relu` 模型能正确推出所有输出 shape
- shape 错误时能明确指出是哪一个 op 出错

### 3.5 任务 1-3：实现 `NaiveMemoryPlanner` 和 `InitSchedulePass`

#### `NaiveMemoryPlanner` 实现方法

1. 只给**非 constant tensor** 分配 workspace。
2. 常量 tensor 的数据不写 `offset`，而是写 `param_offset`。
3. `nbytes = num_elements(shape) * dtype_size(dtype)`。
4. workspace offset 按 tensor 顺序线性累加。
5. 对齐统一使用 `64` 字节。

#### `InitSchedulePass` 实现方法

1. 遍历所有 op。
2. 对每个 op 赋默认 schedule：
   - `tile_m = tile_n = tile_k = -1`
   - `reorder_ikj = false`
   - `unroll_inner = false`
3. 阶段 1 不做任何调度优化，只保证字段完整。

#### 完成标准

- 每个 tensor 都有正确的 `nbytes`
- 每个非 constant tensor 都有确定的 `offset`
- 每个 op 都有默认 `schedule`

### 3.6 任务 1-4：把 C codegen 跑通

#### 具体实现方法

1. 在 `c_codegen.cpp` 中，不要直接写一个大函数拼完所有代码，改成多个小发射器：
   - `emit_header()`
   - `emit_workspace_size_function()`
   - `emit_matmul()`
   - `emit_add()`
   - `emit_relu()`
   - `emit_run_function()`
2. 生成代码时，把 tensor 地址统一写成：
   - constant tensor：`params + param_offset`
   - activation tensor：`workspace + offset`
   - graph input/output：来自 `tiny_tvm_run()` 参数数组
3. 阶段 1 默认只支持 `float32`。
4. `tiny_tvm_run()` 中按 op 顺序发射执行代码。
5. `tiny_tvm_workspace_size()` 返回 memory planner 计算出的总 workspace 大小。

#### 完成标准

- 生成的 `deploy.c` 不是空壳
- 至少包含 `MatMul`、`Add`、`Relu` 的实际循环实现

### 3.7 任务 1-5：接通 `ttvmc compile` 和 `run_model`

#### `ttvmc compile` 实现方法

1. 参数格式固定：

```bash
./build/ttvmc compile examples/json/mlp.json -o out/mlp
```

2. `compile` 子命令按顺序做这些事：
   - 读取 JSON
   - 构建 Graph
   - 跑 pass pipeline
   - 导出 `graph.json`
   - 导出 `params.bin`
   - 生成 `deploy.c`
   - 调用系统 C 编译器生成 `libdeploy.so`
3. 编译器命令建议直接调用：
   - `cc -shared -fPIC deploy.c -O2 -o libdeploy.so`
4. `graph.json` 除了图结构，还要写出：
   - 每个 tensor 的 `shape`
   - `dtype`
   - `nbytes`
   - `offset`
   - `param_offset`

#### `run_model` 实现方法

1. 新增 `src/tools/run_model.cpp`。
2. 参数格式固定：

```bash
./build/run_model out/mlp input.bin output.bin
```

3. `run_model` 做这些事：
   - 读取 `graph.json`
   - 读取 `params.bin`
   - `dlopen("libdeploy.so")`
   - `dlsym()` 获取 `tiny_tvm_workspace_size` 和 `tiny_tvm_run`
   - 申请 workspace
   - 读取 `input.bin`
   - 写出 `output.bin`
4. 阶段 1 只要求单输入单输出模型。

#### 完成标准

- 能编译 MLP
- 能在 Host 上跑出 `output.bin`
- 测试里能和参考值比对

### 3.8 阶段 1 结束后你应该能做什么

- 从 JSON 编译出可执行模型
- 在 PC 上运行 MLP
- 看到真实的 `deploy.c` / `libdeploy.so` / `params.bin`

---

## 4. 阶段 2：推理系统增强（Params / Conv / Fusion）

### 4.1 本阶段目标

把“最小闭环”扩成一个能支撑简单 CNN 的版本。

本阶段新增重点：

- `Conv2D`
- 完整 `params.bin` 流程
- `ConstantFoldPass`
- `FusePass`

### 4.2 本阶段要新增或重点修改的文件

- `include/tiny_tvm/pass/graph/constant_fold_pass.h`
- `src/pass/graph/constant_fold_pass.cpp`
- `include/tiny_tvm/pass/graph/fuse_pass.h`
- `src/pass/graph/fuse_pass.cpp`
- `src/codegen/c_codegen.cpp`
- `src/runtime/runtime.cpp`
- `src/tools/ttvmc.cpp`
- `src/tools/run_model.cpp`
- `tests/pass/fuse_pass_test.cpp`
- `tests/runtime/cnn_e2e_test.cpp`
- `examples/json/cnn.json`

### 4.3 任务 2-1：把参数导出和加载做完整

#### 具体实现方法

1. 从阶段 1 开始的 `params.bin` 格式继续沿用，不新增 header。
2. 导出时按 `graph.tensors()` 顺序遍历：
   - 如果 `is_constant == false`，跳过
   - 如果 `is_constant == true`，先对齐，再写入 raw bytes
   - 回写 `tensor.param_offset`
3. runtime 只负责把整个 `params.bin` 读到一块连续内存。
4. 生成代码只通过 `param_offset` 访问常量，不直接读 `Tensor::data`。

#### 完成标准

- 编译后的常量全部从 `params.bin` 加载
- 运行时不依赖编译期内存结构

### 4.4 任务 2-2：实现 `Conv2D`（NCHW / float32 / naive）

#### 约束先固定

- 数据格式：`NCHW`
- 权重格式：`OIHW`
- dtype：只支持 `float32`
- `groups = 1`
- `dilation = 1`
- 支持 `stride` 和 `padding`

#### 具体实现方法

1. `attrs` 中读取：
   - `strides`
   - `pads`
   - `kernel_shape`
2. shape infer 公式：

```text
OH = (H + 2 * pad_h - KH) / stride_h + 1
OW = (W + 2 * pad_w - KW) / stride_w + 1
```

3. codegen 中直接发射 7 层循环：
   - `n`
   - `oc`
   - `oh`
   - `ow`
   - `ic`
   - `kh`
   - `kw`
4. 越界部分按 padding 处理：越界时跳过乘加。
5. 阶段 2 不做 im2col，不做 Winograd，不做 layout 变换。

#### 完成标准

- 小 CNN 在 Host 上可运行
- `Conv2D` 数值和参考实现一致

### 4.5 任务 2-3：实现 `ConstantFoldPass`

#### 第一版支持范围

只折叠这些情况：

- `Add(constant, constant)`
- `Relu(constant)`
- `Reshape(constant)`（如果你在前一阶段已经支持）

#### 具体实现方法

1. 遍历 op。
2. 若所有输入 tensor 都是 constant，则在编译期直接计算输出数据。
3. 把输出 tensor 标为 `is_constant = true`，并填充 `data`。
4. 当前 op 可以先保留或删除，推荐做法是：
   - 把 op 记为待删除
   - 最后统一清理
5. 不要在阶段 2 引入复杂图重写器，简单 vector 重建就够了。

#### 完成标准

- 常量图子段不会进入 runtime 执行
- 折叠前后结果一致

### 4.6 任务 2-4：实现 `FusePass`（先做 Conv + Relu）

#### 具体实现方法

1. 先统计每个 tensor 的 consumer 数量。
2. 识别模式：
   - 当前 op 是 `Conv`
   - 输出 tensor 只有 1 个 consumer
   - 下一个 op 是 `Relu`
   - `Relu` 的输入正好是这个 `Conv` 输出
3. 满足条件时：
   - 把 `Conv` 改成 `ConvRelu`
   - `Conv` 的输出改成 `Relu` 的输出 tensor
   - 删除 `Relu` op
4. 中间 tensor 可以暂时保留在 `tensors` 里，不必在阶段 2 立刻清理；阶段 3 的 DCE 会处理。
5. codegen 新增 `ConvRelu` 分支：卷积写回前做 `max(0, x)`。

#### 完成标准

- `Conv + Relu` 图能被融合
- 融合后生成代码中不再出现单独的 `Relu` op
- 数值结果和融合前一致

### 4.7 阶段 2 结束后你应该能做什么

- 编译并运行 CNN
- 支持参数文件完整导出/加载
- 具备第一个真实优化 pass

---

## 5. 阶段 3：编译器化（ONNX + 完整 Pass 体系）

### 5.1 本阶段目标

把项目从“能跑样例”升级成“像一个小型编译器”。

本阶段关键点：

- ONNX Frontend
- 算子标准化
- DCE
- Liveness Analysis
- Memory Reuse
- 编译报告和 dump

### 5.2 本阶段要新增或重点修改的文件

- `third_party/onnx/` 或 `third_party/protobuf/` 相关依赖
- `include/tiny_tvm/frontend/onnx/onnx_frontend.h`
- `src/frontend/onnx/onnx_frontend.cpp`
- `include/tiny_tvm/pass/graph/dead_code_elimination_pass.h`
- `src/pass/graph/dead_code_elimination_pass.cpp`
- `include/tiny_tvm/pass/op/op_canonicalize_pass.h`
- `src/pass/op/op_canonicalize_pass.cpp`
- `include/tiny_tvm/pass/memory/liveness_analysis_pass.h`
- `src/pass/memory/liveness_analysis_pass.cpp`
- `include/tiny_tvm/pass/memory/memory_reuse_pass.h`
- `src/pass/memory/memory_reuse_pass.cpp`
- `src/tools/ttvmc.cpp`
- `tests/frontend/onnx_frontend_test.cpp`
- `tests/pass/dce_test.cpp`
- `tests/pass/memory_reuse_test.cpp`
- `examples/onnx/`

### 5.3 任务 3-1：实现 ONNX Frontend

#### 固定实现路线

不要自己解析 ONNX 二进制格式。固定方案如下：

1. 使用 protobuf。
2. 通过 `onnx.proto` 生成 C++ 类。
3. 在 `onnx_frontend.cpp` 中读取：
   - `ModelProto`
   - `GraphProto`
   - `NodeProto`
   - `TensorProto`
4. 初版只支持这些 ONNX 节点：
   - `MatMul`
   - `Gemm`
   - `Add`
   - `Relu`
   - `Conv`
   - `Flatten`
   - `Reshape`

#### 具体实现方法

1. 建立 `std::unordered_map<std::string, int> tensor_name_to_id`。
2. 对 `initializer`：
   - 创建 constant tensor
   - 解析 raw_data 或 float_data
   - 写入 `Tensor::data`
3. 对 graph input/output：
   - 建立 tensor
   - 记录到 `graph_inputs` / `graph_outputs`
4. 对每个 node：
   - 转成内部 `Op`
   - 建立输出 tensor
   - 读取 attrs
5. 阶段 3 仍然要求 ONNX graph 顺序可直接作为拓扑顺序使用。

#### 完成标准

- 至少 1 个 ONNX MLP 或 CNN 模型能导入
- initializers 能转成 constant tensor

### 5.4 任务 3-2：实现 `OpCanonicalizePass`

#### 具体实现方法

1. 第一优先级只做 `Gemm -> MatMul + Add`。
2. `Gemm` 的 `alpha` / `beta` 先只支持默认值 1.0；遇到非默认值直接报“不支持”。
3. 如果 `Gemm` 带 bias：
   - 新建一个 `MatMul` op
   - 再新建一个 `Add` op
4. 阶段 3 不要做太多 canonicalize 规则，先把一条典型路径做完整。

#### 完成标准

- 后续 shape infer / codegen 不需要直接处理 `Gemm`

### 5.5 任务 3-3：实现 `DeadCodeEliminationPass`

#### 具体实现方法

1. 从 `graph_outputs` 出发，反向标记 live tensor。
2. 通过 live tensor 找到 producer op。
3. 反复扩展直到稳定。
4. 最后重建 `ops` 和 `tensors`：
   - 只保留 live 的元素
   - 重建 tensor index 映射
   - 更新所有 op 的输入输出 index
5. 这是第一次需要重建图索引，建议写成一个独立辅助函数，不要把重建逻辑散落在 pass 里。

#### 完成标准

- 死 op 会被真正删除
- 重建后的图索引仍然正确

### 5.6 任务 3-4：实现 `LivenessAnalysisPass` 和 `MemoryReusePass`

#### `LivenessAnalysisPass` 实现方法

1. 对每个 tensor 计算：
   - `def_index`：在哪个 op 被定义；graph input/constant 记为 `-1`
   - `last_use_index`：最后一次被哪个 op 使用
2. 这两个值可以先直接存在临时表里，也可以加到 `Tensor` 上。
3. 阶段 3 不需要复杂 SSA，只要按线性 op 顺序分析即可。

#### `MemoryReusePass` 实现方法

1. 只处理非 constant tensor。
2. 以 op 顺序为时间轴。
3. 维护一个可复用空闲块列表：
   - 块起始 offset
   - 块大小
   - 何时可复用
4. 当新的 tensor 需要内存时：
   - 先找大小足够的空闲块
   - 找不到再分配新 offset
5. 这是典型线性扫描分配，不要在这个项目里上复杂图着色。

#### 完成标准

- 对同一模型，`MemoryReusePass` 后 workspace 总量低于 `NaiveMemoryPlanner`
- 数值结果不变

### 5.7 任务 3-5：补 dump 和编译报告

#### 具体实现方法

1. `ttvmc compile` 增加一个可选参数：
   - `--dump-dir <dir>`
2. 每个关键 pass 后写一个 dump 文件，例如：
   - `00_after_parse.json`
   - `10_after_infer_shape.json`
   - `20_after_fuse.json`
   - `30_after_memory_reuse.json`
3. 再生成 `compile_report.json`，至少包含：
   - 模型名
   - op 数量
   - tensor 数量
   - param bytes
   - workspace bytes
   - 执行过的 pass 列表

#### 完成标准

- 调试时可以直接看每个 pass 后图的变化
- 能快速判断内存优化是否生效

### 5.8 阶段 3 结束后你应该能做什么

- 导入 ONNX 子集模型
- 做基本图优化和内存优化
- 输出有调试价值的编译报告

---

## 6. 阶段 4：Schedule 深化（先服务 MatMul）

### 6.1 本阶段目标

让 Schedule 从“元数据占位”变成“真正控制代码生成”。

### 6.2 设计原则

本项目不要直接做通用 TensorIR。  
阶段 4 固定策略：**只把 Schedule 做到足以控制 MatMul codegen**。

### 6.3 本阶段要新增或重点修改的文件

- `include/tiny_tvm/schedule/schedule.h`
- `include/tiny_tvm/pass/schedule/tiling_pass.h`
- `src/pass/schedule/tiling_pass.cpp`
- `include/tiny_tvm/pass/schedule/loop_reorder_pass.h`
- `src/pass/schedule/loop_reorder_pass.cpp`
- `include/tiny_tvm/pass/schedule/unroll_pass.h`
- `src/pass/schedule/unroll_pass.cpp`
- `src/codegen/c_codegen.cpp`
- `tests/codegen/matmul_schedule_codegen_test.cpp`
- `tests/runtime/matmul_tiled_e2e_test.cpp`

### 6.4 任务 4-1：先把 `Schedule` 定义收敛到可执行版本

#### 推荐字段

```cpp
enum class LoopOrder {
    kIJK,
    kIKJ,
};

struct Schedule {
    int tile_m = -1;
    int tile_n = -1;
    int tile_k = -1;
    LoopOrder order = LoopOrder::kIJK;
    bool unroll_inner = false;
};
```

#### 具体实现方法

1. 不要再用布尔值 `reorder_ikj`，改成显式 `LoopOrder`。
2. 这个结构先只给 `MatMul` 使用，其他 op 保持默认值即可。
3. `InitSchedulePass` 里统一填默认值。

#### 完成标准

- `Schedule` 能明确表达“是否切块、用什么循环顺序、是否展开”

### 6.5 任务 4-2：实现 `TilingPass`

#### 具体实现方法

1. 遍历所有 `MatMul` op。
2. 读取输出 shape 和 reduce 维度。
3. 使用一个固定启发式：
   - 若 `M/N/K >= 64`，tile 取 `32`
   - 若 `M/N/K >= 32`，tile 取 `16`
   - 否则不切块
4. 只对 `MatMul` 设置 `tile_m/tile_n/tile_k`。
5. 阶段 4 不做 auto-tuning，不读硬件 cache 参数。

#### 完成标准

- 至少部分 `MatMul` op 的 schedule 出现非默认 tile 值

### 6.6 任务 4-3：实现 `LoopReorderPass` 和 `UnrollPass`

#### `LoopReorderPass` 具体实现方法

1. 只处理 `MatMul`。
2. 默认顺序是 `IJK`。
3. 当开启 tile 或 `N` 较大时，改成 `IKJ`。
4. 不要支持过多排列组合，先只保留 `IJK` 和 `IKJ` 两种。

#### `UnrollPass` 具体实现方法

1. 只看最内层 `k` 方向。
2. 当 `tile_k` 在 `4~8` 范围内时，开启 `unroll_inner`。
3. codegen 不要依赖编译器 pragma，直接手动展开最内层循环。

#### 完成标准

- schedule 确实影响最终代码形态
- 不是只改字段，不改 codegen

### 6.7 任务 4-4：让 codegen 真的吃到 schedule

#### 具体实现方法

1. 把 `emit_matmul()` 拆成两个分支：
   - `emit_matmul_naive()`
   - `emit_matmul_tiled()`
2. 当 schedule 默认值时，走 naive。
3. 当 tile 有值时，走 tiled emitter。
4. 如果 `order = kIKJ`，发射对应循环顺序。
5. 如果 `unroll_inner = true`，在最内层按固定展开系数输出代码。

#### 完成标准

- 生成的 C 代码在 schedule 开关前后明显不同
- 运行结果一致

### 6.8 阶段 4 结束后你应该能做什么

- 观察 schedule 对生成代码的直接影响
- 在至少一个 MatMul case 上看到可观测性能收益

---

## 7. 阶段 5：ARM / QEMU 部署

### 7.1 本阶段目标

把 Host 上能运行的同一套模型，搬到 ARM 环境中运行。

### 7.2 固定路线：使用 `qemu-arm` 用户态模拟

为了降低复杂度，本项目阶段 5 固定采用：

- 交叉编译：`arm-linux-gnueabihf-g++`
- 运行方式：`qemu-arm -L /usr/arm-linux-gnueabihf`

不要在这个练手项目里一上来做 `qemu-system-arm + kernel + rootfs`，那会把重点从编译器转移到系统启动。

### 7.3 本阶段要新增或重点修改的文件

- `cmake/toolchains/arm-linux-gnueabihf.cmake`
- `scripts/build_arm.sh`
- `scripts/run_qemu.sh`
- `src/tools/run_model.cpp`
- `CMakeLists.txt`
- `tests/runtime/arm_compare_test.md` 或等价说明文档

### 7.4 任务 5-1：打通 ARM 构建

#### 具体实现方法

1. `build_arm.sh` 中固定调用：

```bash
cmake -S . -B build-arm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-linux-gnueabihf.cmake
cmake --build build-arm
```

2. 交叉构建至少要产出：
   - `run_model`
   - 如果需要，也可产出 `ttvmc` 的 ARM 版，但不是必须
3. 编译生成的 `deploy.c` 时，也用同一套 ARM 工具链生成 `libdeploy.so`。

#### 完成标准

- `build-arm/` 中能产出 ARM 可执行文件和 ARM 版动态库

### 7.5 任务 5-2：打通 `qemu-arm` 运行

#### 具体实现方法

1. `run_qemu.sh` 固定为这种调用方式：

```bash
qemu-arm -L /usr/arm-linux-gnueabihf build-arm/run_model out/mlp input.bin output_arm.bin
```

2. 如果运行依赖 `libdeploy.so`，确保：
   - `run_model` 能找到同目录或指定目录下的 `.so`
   - 必要时设置 `LD_LIBRARY_PATH`
3. 先用最小 MLP 跑通，再上 CNN。

#### 完成标准

- ARM 版 `run_model` 能在 `qemu-arm` 下启动并执行

### 7.6 任务 5-3：做 Host / ARM 对比

#### 具体实现方法

1. 在 Host 上跑一次，得到 `output_host.bin`。
2. 在 ARM / QEMU 上跑一次，得到 `output_arm.bin`。
3. 写一个简单对比程序或脚本：
   - 逐元素比较
   - 允许一个很小的误差阈值，例如 `1e-5`
4. 先比较 MLP，再比较 CNN。

#### 完成标准

- 同一个模型在 Host / ARM 上结果一致或近似一致

### 7.7 阶段 5 结束后你应该能做什么

- 在 ARM 目标环境中运行自己编译出来的模型
- 证明这不是只在本机有效的 demo

---

## 8. 每个阶段必须补的测试

这个部分不是“有空再做”，而是每一阶段都必须跟着补。

### 阶段 0 测试

- `tests/ir/graph_smoke_test.cpp`
- 验证 Graph / PassManager / codegen skeleton

### 阶段 1 测试

- `tests/frontend/json_frontend_test.cpp`
- `tests/pass/infer_shape_test.cpp`
- `tests/runtime/mlp_e2e_test.cpp`

### 阶段 2 测试

- `tests/pass/fuse_pass_test.cpp`
- `tests/runtime/cnn_e2e_test.cpp`
- 常量折叠一致性测试

### 阶段 3 测试

- `tests/frontend/onnx_frontend_test.cpp`
- `tests/pass/dce_test.cpp`
- `tests/pass/memory_reuse_test.cpp`

### 阶段 4 测试

- `tests/codegen/matmul_schedule_codegen_test.cpp`
- `tests/runtime/matmul_tiled_e2e_test.cpp`

### 阶段 5 测试

- Host / ARM 输出对比脚本或测试

---

## 9. 推荐开发顺序（严格按这个顺序走）

1. 完成阶段 0，保证骨架稳定
2. 完成阶段 1，跑通 MLP
3. 完成阶段 2，跑通 CNN
4. 完成阶段 3，接入 ONNX + DCE + MemoryReuse
5. 完成阶段 4，让 Schedule 真正影响 MatMul codegen
6. 完成阶段 5，把模型跑到 ARM / QEMU

不要跳步，尤其不要在阶段 1 没闭环前就去碰 ONNX 或 Schedule。

---

## 10. 你现在应该怎么开始

如果你现在就要进入编码，直接按下面顺序开工：

1. 先完成阶段 0 里对 `Tensor`、`Graph`、`PassManager` 的补全
2. 然后进入阶段 1，先接 `nlohmann/json`
3. 先让 `examples/json/mlp.json` 能成功解析
4. 再实现 `InferShapePass`
5. 再实现 `NaiveMemoryPlanner`
6. 再接 `deploy.c` 生成
7. 最后接通 `ttvmc compile` 和 `run_model`

做到这里，你就已经完成了这个项目最关键的第一条主链路。后面所有阶段，本质上都是在这条主链路上继续扩展。
