# Tiny TVM 各阶段任务详细描述

> 本文档是 `phase_task_checklist.md` 的详解版。对每个阶段的每个任务，展开说明：**做什么**、**实现逻辑**、**为什么要做**、**这个任务的作用**、**完成标准**。
>
> 用户只需要按本文档编写代码，然后运行 `tests/phase_task_test/test_phaseN.sh` 来验证。

---

## 阶段 0：工程骨架完善

> **阶段目标**：把已有的 C++17 工程骨架补成一个稳定、可持续开发的基础。完成后应该能稳定构建、跑 smoke test、并且后续阶段不需要回来改骨架。

### 任务 0-1：把 IR 定义补到后面不需要大改

#### 做什么

在 `include/tiny_tvm/ir/graph.h` 中完善 `Tensor` 和 `Graph` 的定义，使其包含后续所有阶段需要的字段和辅助函数。

**具体改动：**

1. 给 `Tensor` 补充两个字段：
   - `size_t param_offset = 0;` — 常量张量在 `params.bin` 中的偏移
   - `std::vector<uint8_t> data;` — 编译期保存常量内容的原始字节
2. 新增三个辅助函数：
   - `num_elements(const Tensor&)` — 计算张量元素总数（shape 各维度相乘）
   - `align_up(size_t value, size_t alignment)` — 内存对齐辅助（向上取整到 alignment 的倍数）
   - `dtype_size(DType)` — 已存在，确认返回值正确（float32→4, int32→4, unknown→0）
3. 确认 `Graph` 提供以下接口：
   - `add_tensor()`、`add_op()` — 添加元素
   - `tensor(index)`、`op(index)` — 按索引访问
   - `summary()` — 固定格式的摘要字符串

#### 实现逻辑

```cpp
// graph.h 中补充
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype = DType::kUnknown;
    size_t nbytes = 0;
    size_t offset = 0;         // workspace offset，非 constant tensor 使用
    size_t param_offset = 0;   // params.bin offset，constant tensor 使用
    bool is_constant = false;
    std::vector<uint8_t> data; // 编译期常量内容
};

// 辅助函数
inline int64_t num_elements(const Tensor& t) {
    int64_t n = 1;
    for (auto d : t.shape) n *= d;
    return n;
}

inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}
```

#### 为什么要做

这是**整个项目的数据基础**。后续所有阶段——前端解析、Pass 分析、代码生成、运行时——都直接操作 `Tensor` 和 `Graph`。如果阶段 0 不把字段定义完整，后面每个阶段都要回来改结构体，导致频繁返工。

- `param_offset` 在阶段 1 的 `params.bin` 导出和代码生成中是必须的
- `data` 在阶段 2 的 `ConstantFoldPass` 中是必须的
- `num_elements` 在内存规划中是必须的
- `align_up` 在 workspace offset 分配中是必须的

#### 作用

建立一套稳定的核心数据结构，使后续开发可以专注于功能逻辑而非结构改动。

#### 完成标准

- `Tensor` 包含 `param_offset` 和 `data` 字段
- `num_elements()`、`align_up()`、`dtype_size()` 存在且行为正确
- `Graph` 可以添加/访问 tensor 和 op，`summary()` 输出非空

---

### 任务 0-2：把 Pass 基础设施做成可长期复用的版本

#### 做什么

固定 `Pass` 基类接口和 `PassManager` 行为，使得后续添加任何新 Pass 时只需新增类并 `add()`，不需要修改框架代码。

**具体改动：**

1. `Pass` 基类保持最小接口：`name()` + `run(Graph&)`
2. `PassManager` 维护 `std::vector<std::unique_ptr<Pass>>`
3. `PassManager::run()` 按顺序执行所有 pass
4. 预留 verbose 调试入口

#### 实现逻辑

当前代码已经有正确的框架，主要确认：

```cpp
class PassManager {
public:
    void add(std::unique_ptr<Pass> pass);
    void run(ir::Graph& graph) const;  // 按序执行每个 pass
    size_t size() const noexcept;
    std::vector<std::string> pass_names() const;
private:
    std::vector<std::unique_ptr<Pass>> passes_;
    bool verbose_ = false;  // 预留：verbose 时每个 pass 前后打印 summary
};
```

`run()` 的核心逻辑就是 `for (auto& p : passes_) p->run(graph);`。verbose 模式下在每个 pass 前后调用 `graph.summary()` 打印日志。

#### 为什么要做

Pass 是编译器的核心抽象。TVM、LLVM、MLIR 都采用类似的 Pass 管线设计。固定这个框架可以：

- 让后续 InferShapePass、MemoryPlanner、FusePass 等都无缝接入
- 提供统一的调试入口（verbose dump）
- 保证 pass 执行顺序可控

#### 作用

提供可扩展的 Pass 管线框架，是后续所有编译优化（shape 推导、内存规划、算子融合、DCE 等）的载体。

#### 完成标准

- 可连续执行多个 pass，顺序正确
- 添加新 pass 只需 `add(std::make_unique<XxxPass>())`

---

### 任务 0-3：固定 CLI 骨架

#### 做什么

让 `ttvmc` 成为整个项目的统一命令行入口，支持 `help`、`version`、`smoke`、`compile` 四个子命令。

**具体改动：**

1. `help`：打印固定格式的 usage
2. `version`：打印版本号
3. `smoke`：构造内置 demo graph → 跑 NoOpPass → 打印 graph summary + runtime summary + C 代码骨架
4. `compile`：阶段 0 先占位报错，阶段 1 接通

#### 实现逻辑

当前 `ttvmc.cpp` 已基本实现，确认 `smoke` 子命令输出三部分内容：
1. `graph.summary()` — 图结构摘要
2. `runtime.describe()` — 运行时信息
3. `emit_c_module(graph)` — 生成的 C 代码

#### 为什么要做

CLI 是开发者和系统交互的唯一入口。固定 CLI 格式后：

- `smoke` 可以作为每次重构后的**快速回归命令**
- `compile` 的参数格式固定后，后续阶段只需填充逻辑，不需要改接口
- 统一的 usage 降低使用门槛

#### 作用

提供稳定的用户接口，后续所有功能都通过这个入口暴露。

#### 完成标准

- `./build/ttvmc smoke` 正常输出 graph summary + runtime info + C skeleton
- `./build/ttvmc help` 和 `version` 正常工作
- `compile` 有占位或实际实现

---

### 任务 0-4：把测试和构建入口固定下来

#### 做什么

确保工程至少有 1 个测试，构建脚本有依赖检查。

**具体改动：**

1. `tests/ir/graph_smoke_test.cpp`：验证 Graph 构造 → NoOpPass → codegen 包含 `tiny_tvm_run`
2. `scripts/build_host.sh`：检查 cmake 存在 → cmake configure → cmake build → ctest
3. CMakeLists.txt 定义 `tiny_tvm_core`、`ttvmc`、`graph_smoke_test` 三个目标

#### 实现逻辑

`graph_smoke_test.cpp` 的核心验证逻辑：
```cpp
Graph graph = /* 构造 3 tensors + 1 MatMul op */;
PassManager pm;
pm.add(make_unique<NoOpPass>());
pm.run(graph);
assert(graph.tensor_count() == 3);
assert(graph.op_count() == 1);
string code = emit_c_module(graph);
assert(code.find("tiny_tvm_run") != string::npos);
```

#### 为什么要做

没有测试的代码无法保证正确性。smoke test 是最基本的**冒烟测试**，确保核心链路不会因为重构而断裂。构建脚本的依赖检查避免用户在缺少工具时得到不可理解的错误。

#### 作用

建立 CI 级别的最小验证能力，保障后续开发的稳定性。

#### 完成标准

- `cmake` 环境下可以 build + test
- `graph_smoke_test` 通过
- ctest 报告全绿

---

## 阶段 1：最小闭环（JSON → IR → Pass → Codegen → Host）

> **阶段目标**：跑通第一条完整编译链路——从 JSON 模型定义到在 PC 上执行推理并输出正确结果。这是项目最关键的里程碑。

### 任务 1-1：固定 JSON 模型格式并实现 Frontend

#### 做什么

实现 JSON 前端，能把 `examples/json/mlp.json` 解析成内部 `Graph` 结构。

**具体改动：**

1. 引入 `nlohmann/json.hpp`（单头文件 JSON 库）到 `third_party/nlohmann/`
2. 实现 `src/frontend/json/json_frontend.cpp` 中的 `parse_text()` 和 `load_file()`

#### 实现逻辑

```cpp
ParseResult parse_text(const std::string& text) {
    auto j = nlohmann::json::parse(text);
    Graph graph;

    // 1. 解析 tensors
    for (auto& jt : j["tensors"]) {
        Tensor t;
        t.name = jt["name"];
        t.shape = jt["shape"].get<vector<int64_t>>();
        t.dtype = str_to_dtype(jt["dtype"]);  // "float32" → DType::kFloat32
        t.is_constant = jt.value("is_constant", false);
        if (t.is_constant && jt.contains("data")) {
            // 将 JSON 数组转为 raw bytes
            auto floats = jt["data"].get<vector<float>>();
            t.data.resize(floats.size() * sizeof(float));
            memcpy(t.data.data(), floats.data(), t.data.size());
        }
        graph.add_tensor(std::move(t));
    }

    // 2. 解析 ops
    for (auto& jo : j["ops"]) {
        Op op;
        op.kind = jo["kind"];
        op.inputs = jo["inputs"].get<vector<int>>();
        op.outputs = jo["outputs"].get<vector<int>>();
        // 解析 attrs（可选）
        graph.add_op(std::move(op));
    }

    // 3. 解析 graph_inputs / graph_outputs
    graph.graph_inputs() = j["graph_inputs"].get<vector<int>>();
    graph.graph_outputs() = j["graph_outputs"].get<vector<int>>();

    return {true, std::move(graph), ""};
}
```

#### 为什么要做

**前端是编译器的入口**。没有前端，就无法从外部模型定义构建内部 IR。JSON 格式简单易调试，是理想的第一个前端格式。后续阶段 3 会加入更复杂的 ONNX 前端。

选择 `nlohmann/json` 的原因：
- 单头文件，零依赖
- C++11/17 兼容
- API 直观

#### 作用

将外部模型定义转化为内部 `Graph` 表示，是编译链路的第一环。

#### 完成标准

- 能把 `examples/json/mlp.json` 成功转成 `Graph`（3 tensors, 1 op）
- 非法 JSON 给出可读错误信息
- 常量 tensor 的 `data` 被正确填充

---

### 任务 1-2：实现 InferShapePass

#### 做什么

新建 `InferShapePass`，根据算子语义自动推导输出 tensor 的 shape 和 dtype。

**支持的规则：**
- `MatMul`: `[M, K] × [K, N] → [M, N]`
- `Add`: 输入 shape 完全相同，输出 = 输入 shape
- `Relu`: 输出 shape = 输入 shape
- `Constant`/`Input`: shape 来自前端，不需要推导

#### 实现逻辑

```cpp
void InferShapePass::run(Graph& graph) {
    for (size_t i = 0; i < graph.op_count(); i++) {
        auto& op = graph.op(i);
        if (op.kind == "MatMul") {
            auto& lhs = graph.tensor(op.inputs[0]);  // [M, K]
            auto& rhs = graph.tensor(op.inputs[1]);  // [K, N]
            CHECK(lhs.shape[1] == rhs.shape[0]);      // K 必须匹配
            auto& out = graph.tensor(op.outputs[0]);
            out.shape = {lhs.shape[0], rhs.shape[1]}; // [M, N]
            out.dtype = lhs.dtype;
        } else if (op.kind == "Add") {
            auto& a = graph.tensor(op.inputs[0]);
            auto& b = graph.tensor(op.inputs[1]);
            CHECK(a.shape == b.shape);  // 阶段 1 不做 broadcast
            auto& out = graph.tensor(op.outputs[0]);
            out.shape = a.shape;
            out.dtype = a.dtype;
        } else if (op.kind == "Relu") {
            auto& in = graph.tensor(op.inputs[0]);
            auto& out = graph.tensor(op.outputs[0]);
            out.shape = in.shape;
            out.dtype = in.dtype;
        }
    }
}
```

#### 为什么要做

**Shape 推导是所有后续 Pass 的前提**。内存规划需要知道每个 tensor 的大小（`nbytes = num_elements × dtype_size`），代码生成需要知道循环边界。如果 shape 不对，整个编译链路都是错的。

在真实的 TVM 中，shape 推导（type inference）是编译器的核心功能之一，ONNX Runtime 也有独立的 shape inference 组件。

#### 作用

为每个 tensor 确定 shape 和 dtype，是内存规划和代码生成的基础。

#### 完成标准

- `MatMul [1,4]×[4,8]` 正确推出 `[1,8]`
- shape 错误时明确指出是哪个 op 出错

---

### 任务 1-3：实现 NaiveMemoryPlanner 和 InitSchedulePass

#### 做什么

为每个 tensor 分配内存：
- **NaiveMemoryPlanner**：给非 constant tensor 分配 workspace offset，给 constant tensor 分配 param_offset
- **InitSchedulePass**：给每个 op 赋默认 schedule

#### 实现逻辑

**NaiveMemoryPlanner：**
```cpp
void NaiveMemoryPlanner::run(Graph& graph) {
    size_t ws_offset = 0;
    size_t param_offset = 0;

    for (size_t i = 0; i < graph.tensor_count(); i++) {
        auto& t = graph.tensor(i);
        t.nbytes = num_elements(t) * dtype_size(t.dtype);

        if (t.is_constant) {
            t.param_offset = align_up(param_offset, 64);
            param_offset = t.param_offset + t.nbytes;
        } else {
            t.offset = align_up(ws_offset, 64);
            ws_offset = t.offset + t.nbytes;
        }
    }
}
```

**InitSchedulePass：**
```cpp
void InitSchedulePass::run(Graph& graph) {
    for (size_t i = 0; i < graph.op_count(); i++) {
        auto& op = graph.op(i);
        op.schedule = Schedule{};  // 所有字段默认值
    }
}
```

#### 为什么要做

**内存规划是连接编译器和运行时的关键桥梁**。代码生成需要知道每个 tensor 的内存地址才能发射正确的 load/store 指令。

- `offset` 告诉生成的 C 代码到 workspace 的哪个位置读写激活值
- `param_offset` 告诉生成的 C 代码到 params 的哪个位置读取权重
- 64 字节对齐是为了后续可能的 SIMD 优化（ARM NEON 通常要求 16/32 字节对齐，64 更保守）

NaiveMemoryPlanner 是最简单的"线性堆叠"方案，阶段 3 会引入 MemoryReusePass 来复用空间。

#### 作用

- NaiveMemoryPlanner：确定每个 tensor 在运行时的内存位置
- InitSchedulePass：为 op 的 schedule 字段填充默认值，保证后续 codegen 可以读到完整的 schedule

#### 完成标准

- 每个 tensor 都有正确的 `nbytes`（= 元素数 × 类型大小）
- 每个非 constant tensor 都有确定的 `offset`
- 每个 op 都有默认 `schedule`（`is_default() == true`）

---

### 任务 1-4：把 C codegen 跑通

#### 做什么

让 `c_codegen.cpp` 能生成**真正可编译执行**的 C 代码，包含 MatMul、Add、Relu 的循环实现。

**具体改动：**

1. 把 `emit_c_module()` 拆成多个子函数：`emit_header()`、`emit_workspace_size_function()`、`emit_matmul()`、`emit_add()`、`emit_relu()`、`emit_run_function()`
2. 函数签名改为标准接口：
```c
size_t tiny_tvm_workspace_size(void);
int tiny_tvm_run(const void* params, void* workspace,
                 const void* const* inputs, void* const* outputs);
```

#### 实现逻辑

以 MatMul `[M,K]×[K,N]→[M,N]` 为例：
```c
// emit_matmul() 生成的代码
{
    const float* a = (const float*)(inputs[0]);  // 或 workspace + offset
    const float* b = (const float*)(params + param_offset);
    float* c = (float*)(workspace + offset);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}
```

地址计算规则：
- **graph input tensor** → 从 `inputs[]` 数组取
- **graph output tensor** → 写到 `outputs[]` 数组
- **constant tensor** → `(const float*)((const char*)params + param_offset)`
- **中间 activation** → `(float*)((char*)workspace + offset)`

#### 为什么要做

**Codegen 是编译器的最终产出**。前面所有的解析、分析、优化，最终都要落地成可执行代码。C 代码是最简单、最可移植的后端——可以被任何 C 编译器（gcc、clang、ARM 交叉编译器）编译。

TVM 的 C codegen 也是其部署方案 MicroTVM 的核心组件。

#### 作用

将 IR 图翻译成可编译的 C 代码，是从"编译器内部表示"到"可执行代码"的最后一步。

#### 完成标准

- 生成的 `deploy.c` 不是空壳
- 包含 `tiny_tvm_run` 和 `tiny_tvm_workspace_size` 函数
- MatMul/Add/Relu 有实际的循环实现

---

### 任务 1-5：接通 ttvmc compile 和 run_model

#### 做什么

让 `ttvmc compile` 完成完整编译流程，让 `run_model` 能加载编译产物并执行推理。

**`ttvmc compile` 流程：**
```
读取 JSON → 构建 Graph → 跑 pass pipeline → 导出 graph.json → 导出 params.bin → 生成 deploy.c → 编译 libdeploy.so
```

**`run_model` 流程：**
```
读取 graph.json → 读取 params.bin → dlopen(libdeploy.so) → dlsym(tiny_tvm_run) → 执行推理 → 写出 output.bin
```

#### 实现逻辑

**ttvmc compile 核心代码：**
```cpp
// 1. 解析
auto result = json::load_file(model_path);
auto& graph = result.graph;

// 2. Pass pipeline
PassManager pm;
pm.add(make_unique<InferShapePass>());
pm.add(make_unique<NaiveMemoryPlanner>());
pm.add(make_unique<InitSchedulePass>());
pm.run(graph);

// 3. 导出
export_graph_json(graph, out_dir + "/graph.json");
export_params_bin(graph, out_dir + "/params.bin");
string code = emit_c_module(graph);
write_file(out_dir + "/deploy.c", code);

// 4. 编译 .so
system("cc -shared -fPIC deploy.c -O2 -o libdeploy.so");
```

**run_model 核心代码：**
```cpp
// 1. 加载
auto graph_info = load_graph_json(dir + "/graph.json");
auto params = read_file(dir + "/params.bin");

// 2. dlopen
void* handle = dlopen((dir + "/libdeploy.so").c_str(), RTLD_NOW);
auto ws_size = (size_t(*)())dlsym(handle, "tiny_tvm_workspace_size");
auto run = (int(*)(const void*, void*, const void* const*, void* const*))
           dlsym(handle, "tiny_tvm_run");

// 3. 执行
vector<uint8_t> workspace(ws_size());
auto input_data = read_file(input_path);
const void* inputs[] = {input_data.data()};
void* outputs[] = {output_buf.data()};
run(params.data(), workspace.data(), inputs, outputs);
write_file(output_path, output_buf);
```

#### 为什么要做

这是**最小闭环的最后一步**。只有当用户能从一个 JSON 文件出发，经过编译，得到可执行的模型并看到正确的推理结果，这个项目才算"能用"。

`ttvmc compile` 对标 TVM 的 `tvmc compile`，`run_model` 对标 TVM 的 `tvmc run`。

#### 作用

把前面所有组件（Frontend、Pass、Codegen）串成一条完整的用户可用的链路。

#### 完成标准

- `ttvmc compile mlp.json -o out/mlp` 产出 `graph.json`、`params.bin`、`deploy.c`、`libdeploy.so`
- `run_model out/mlp input.bin output.bin` 产出正确数值
- graph.json 中每个 tensor 包含 shape、dtype、nbytes、offset/param_offset

---

## 阶段 2：推理系统增强（Params / Conv / Fusion）

> **阶段目标**：将"最小闭环"扩展为能支撑简单 CNN 的版本。新增 Conv2D 算子、完整参数流程、常量折叠和算子融合。

### 任务 2-1：把参数导出和加载做完整

#### 做什么

确保 `params.bin` 的导出和加载流程正确完整：编译时按 tensor 顺序写出常量数据，运行时读入后通过 `param_offset` 访问。

#### 实现逻辑

**导出：**
```cpp
void export_params_bin(const Graph& graph, const string& path) {
    ofstream ofs(path, ios::binary);
    size_t offset = 0;
    for (auto& t : graph.tensors()) {
        if (!t.is_constant) continue;
        size_t aligned = align_up(offset, 64);
        // 填充对齐字节
        vector<uint8_t> padding(aligned - offset, 0);
        ofs.write((char*)padding.data(), padding.size());
        ofs.write((char*)t.data.data(), t.data.size());
        offset = aligned + t.data.size();
    }
}
```

**加载（runtime）：** 把整个 `params.bin` 读到一块连续内存，然后 `tiny_tvm_run` 直接通过 `params + param_offset` 访问。

#### 为什么要做

真实部署中，模型权重必须持久化到文件。`params.bin` 是最简单的"原始字节拼接"格式——二进制里只放纯数据，元信息都放 `graph.json`。这样：

- 运行时不需要解析复杂二进制格式
- 编译器输出和运行时输入解耦
- 后续加新的 constant tensor 不需要改格式

#### 作用

实现编译期常量数据的序列化和运行时反序列化，是模型部署的基本能力。

#### 完成标准

- 编译后的常量全部从 `params.bin` 加载
- 运行时不依赖编译期内存结构

---

### 任务 2-2：实现 Conv2D（NCHW / float32 / naive）

#### 做什么

在 InferShapePass 和 C Codegen 中新增 Conv2D 支持。

**约束：** NCHW 数据格式、OIHW 权重格式、float32、groups=1、dilation=1。

#### 实现逻辑

**Shape 推导：**
```
输入: [N, CI, H, W]
权重: [CO, CI, KH, KW]
输出: [N, CO, OH, OW]
OH = (H + 2*pad_h - KH) / stride_h + 1
OW = (W + 2*pad_w - KW) / stride_w + 1
```

**Codegen 生成 7 层循环：**
```c
for (int n = 0; n < N; n++)
  for (int oc = 0; oc < CO; oc++)
    for (int oh = 0; oh < OH; oh++)
      for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        for (int ic = 0; ic < CI; ic++)
          for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
              int ih = oh * stride_h - pad_h + kh;
              int iw = ow * stride_w - pad_w + kw;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum += input[n][ic][ih][iw] * weight[oc][ic][kh][kw];
            }
        output[n][oc][oh][ow] = sum;
      }
```

#### 为什么要做

Conv2D 是 CNN 的核心算子。没有 Conv2D，项目只能处理全连接网络（MLP），无法处理图像识别等经典任务。7 层嵌套循环是最直观的 naive 实现，虽然性能不高，但**正确性有保证**，后续可以在此基础上做优化（im2col、Winograd 等）。

#### 作用

扩展算子覆盖范围，使项目能编译和运行 CNN 模型。

#### 完成标准

- 小 CNN 在 Host 上可运行
- Conv2D 数值和参考实现一致

---

### 任务 2-3：实现 ConstantFoldPass

#### 做什么

如果一个 op 的所有输入都是常量，在编译期直接计算结果，将输出标为常量。

**第一版支持：** `Add(const, const)`、`Relu(const)`

#### 实现逻辑

```cpp
void ConstantFoldPass::run(Graph& graph) {
    vector<size_t> to_remove;
    for (size_t i = 0; i < graph.op_count(); i++) {
        auto& op = graph.op(i);
        bool all_const = true;
        for (int idx : op.inputs) {
            if (!graph.tensor(idx).is_constant) { all_const = false; break; }
        }
        if (!all_const) continue;

        // 在编译期执行 op
        if (op.kind == "Add") {
            auto& a = graph.tensor(op.inputs[0]);
            auto& b = graph.tensor(op.inputs[1]);
            auto& out = graph.tensor(op.outputs[0]);
            out.data.resize(a.data.size());
            float* pa = (float*)a.data.data();
            float* pb = (float*)b.data.data();
            float* po = (float*)out.data.data();
            for (size_t j = 0; j < a.data.size()/4; j++) po[j] = pa[j] + pb[j];
            out.is_constant = true;
            out.shape = a.shape;
            out.dtype = a.dtype;
            out.nbytes = a.nbytes;
        }
        to_remove.push_back(i);
    }
    // 反向删除
    for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it) {
        graph.mutable_ops().erase(graph.mutable_ops().begin() + *it);
    }
}
```

#### 为什么要做

**常量折叠是编译器最基本的优化之一**。如果两个常量相加，没必要在运行时做，编译期就可以算好。这样可以：

- 减少运行时的计算量
- 减少 workspace 使用（折叠后的 tensor 变成常量，放入 params.bin）
- 简化后续的图分析

GCC/LLVM 在编译 C 代码时也做常量折叠。

#### 作用

在编译期消除可预计算的操作，减少运行时开销。

#### 完成标准

- `Add(const_a=[1,2,3,4], const_b=[5,6,7,8])` 折叠后输出 `[6,8,10,12]`
- 折叠前后结果一致

---

### 任务 2-4：实现 FusePass（先做 Conv + Relu）

#### 做什么

将 `Conv2D` 后面紧跟的 `Relu` 融合成一个 `ConvRelu` 算子。

#### 实现逻辑

```
识别模式：
1. 统计每个 tensor 的 consumer 数量
2. 找到 Conv op，其输出 tensor 只有 1 个 consumer
3. 该 consumer 是 Relu op
4. 满足条件 → 合并为 ConvRelu

合并操作：
- Conv.kind = "ConvRelu"
- Conv.outputs = [Relu 的输出 tensor]
- 删除 Relu op
```

Codegen 中 `ConvRelu` 的代码和 `Conv2D` 相同，但在写回结果前加 `max(0, sum)`。

#### 为什么要做

**算子融合是深度学习编译器的核心优化**。Conv + Relu 是最常见的组合模式。融合后：

- 减少一次中间结果的读写（Conv 输出不需要写回内存再由 Relu 读取）
- 减少一个 tensor 的 workspace 占用
- 在实际硬件上可以利用 SIMD/流水线优化

TVM、TensorRT、ONNX Runtime 都会做 Conv+Relu 融合。

#### 作用

减少内存带宽压力和计算冗余，是性能优化的第一步。

#### 完成标准

- `Conv + Relu` 被融合为 `ConvRelu`
- 融合后不再有独立的 Relu op
- 数值结果和融合前一致

---

## 阶段 3：编译器化（ONNX + 完整 Pass 体系）

> **阶段目标**：把项目从"能跑样例"升级成"像一个小型编译器"——能导入标准格式模型、做图优化、做内存优化、输出调试信息。

### 任务 3-1：实现 ONNX Frontend

#### 做什么

使用 protobuf 解析 `.onnx` 模型文件，转成内部 `Graph`。

**支持的 ONNX 节点：** MatMul、Gemm、Add、Relu、Conv、Flatten、Reshape

#### 实现逻辑

1. 用 `onnx.proto` 生成 C++ 类（`ModelProto`、`GraphProto`、`NodeProto`、`TensorProto`）
2. 建立 `tensor_name_to_id` 映射
3. 先处理 `initializer`（常量权重）→ 创建 constant tensor
4. 再处理 graph input/output
5. 最后按顺序处理每个 node → 创建 Op + 输出 tensor

#### 为什么要做

ONNX 是深度学习模型的事实标准交换格式。支持 ONNX 意味着：

- 可以导入 PyTorch 导出的模型
- 可以导入 TensorFlow 通过 tf2onnx 转换的模型
- 使项目从"只能跑自定义 JSON"升级为"能跑业界标准模型"

#### 作用

打通与上游训练框架的连接，使编译器能处理真实模型。

#### 完成标准

- 至少 1 个 ONNX MLP 或 CNN 模型能成功导入
- initializers 正确转成 constant tensor

---

### 任务 3-2：实现 OpCanonicalizePass

#### 做什么

将高级复合算子拆解为基础算子。第一优先级：`Gemm → MatMul + Add`。

#### 实现逻辑

```cpp
if (op.kind == "Gemm") {
    // 新建中间 tensor (MatMul 输出)
    Tensor mid;
    mid.name = op.name + "_matmul_out";
    mid.dtype = DType::kFloat32;
    int mid_id = graph.add_tensor(std::move(mid));

    // 原 op 变成 MatMul
    Op matmul;
    matmul.kind = "MatMul";
    matmul.inputs = {op.inputs[0], op.inputs[1]};
    matmul.outputs = {mid_id};

    // 新建 Add op
    Op add;
    add.kind = "Add";
    add.inputs = {mid_id, op.inputs[2]};  // bias
    add.outputs = op.outputs;

    // 替换原 op
}
```

#### 为什么要做

**算子标准化简化了后续处理**。如果每个复合算子都在 InferShapePass、Codegen 中单独处理，代码会爆炸式增长。标准化后，后续只需要处理基础算子集合。

这和 LLVM 的 InstCombine、Canonicalize pass 是同一个思路。

#### 作用

减少后续 Pass 和 Codegen 需要处理的算子种类。

#### 完成标准

- 后续 shape infer / codegen 不需要直接处理 `Gemm`
- Gemm 被正确拆为 MatMul + Add

---

### 任务 3-3：实现 DeadCodeEliminationPass

#### 做什么

删除不影响最终输出的"死"op 和 tensor。

#### 实现逻辑

```
1. 从 graph_outputs 出发，反向标记 live tensor
2. 通过 live tensor 找到 producer op → 该 op 的所有输入也是 live
3. 反复扩展直到稳定
4. 重建 ops 和 tensors：只保留 live 的元素
5. 重建 tensor index 映射（old_id → new_id），更新所有 op 的 inputs/outputs
```

**重建图索引的辅助函数很重要**：建议写成独立函数，因为后续其他 Pass 也可能需要。

#### 为什么要做

常量折叠、算子融合后可能产生不再使用的 tensor 和 op。DCE 负责清理它们：

- 减少不必要的计算和内存占用
- 简化图结构，利于后续分析
- 这是编译器的标准优化（LLVM 也有 DCE pass）

#### 作用

清理图中的无用节点，减少运行时开销。

#### 完成标准

- 死 op 被真正删除（op_count 减少）
- 重建后的图索引仍然正确

---

### 任务 3-4：实现 LivenessAnalysisPass 和 MemoryReusePass

#### 做什么

分析每个 tensor 的生命周期，然后复用不再需要的内存空间。

#### 实现逻辑

**LivenessAnalysisPass：**
```
对每个 tensor 计算：
- def_index：在哪个 op 被定义（graph input/constant 记为 -1）
- last_use_index：最后一次被哪个 op 使用

按线性 op 顺序扫描即可，不需要复杂 SSA。
```

**MemoryReusePass：**
```
1. 维护空闲块列表：[(offset, size, free_after_op)]
2. 按 op 顺序遍历：
   - 在当前 op 之前释放的 tensor → 加入空闲列表
   - 当前 op 的输出 tensor 需要分配：
     - 先在空闲列表中找大小足够的块
     - 找到 → 复用
     - 找不到 → 分配新 offset
3. 这是典型的线性扫描寄存器分配算法的内存版本
```

#### 为什么要做

NaiveMemoryPlanner 对每个 tensor 都分配独立空间，实际上很多 tensor 的生命周期不重叠，可以共用同一块内存。例如一个 3 层 MLP：

- Naive: input + hidden1 + hidden2 + output = 4 块
- Reuse: input + hidden(复用) + output = 最少 3 块

对于大模型，内存复用可以显著减少 workspace 大小，这在嵌入式部署中尤其重要。

#### 作用

通过分析 tensor 生命周期来复用内存，减少运行时的 workspace 需求。

#### 完成标准

- MemoryReusePass 后 workspace 总量低于 NaiveMemoryPlanner
- 数值结果不变

---

### 任务 3-5：补 dump 和编译报告

#### 做什么

为 `ttvmc compile` 增加 `--dump-dir` 参数，在每个关键 pass 后写 dump 文件，并生成 `compile_report.json`。

#### 实现逻辑

```
dump 文件命名：
- 00_after_parse.json
- 10_after_infer_shape.json
- 20_after_fuse.json
- 30_after_memory_reuse.json

compile_report.json 内容：
{
    "model_name": "mlp",
    "op_count": 3,
    "tensor_count": 7,
    "param_bytes": 1024,
    "workspace_bytes": 512,
    "passes": ["InferShapePass", "NaiveMemoryPlanner", "InitSchedulePass"]
}
```

#### 为什么要做

**可观测性是编译器开发的生命线**。没有 dump，你根本不知道某个 pass 做了什么、是否正确。编译报告让用户快速了解编译结果，也方便自动化测试验证。

TVM 有完善的 dump 系统（`relay.build` 的 debug 输出），LLVM 有 `-print-after-all`。

#### 作用

提供编译过程的可观测性，方便调试和验证。

#### 完成标准

- `--dump-dir` 生成 pass 后的图快照
- `compile_report.json` 包含关键统计信息

---

## 阶段 4：Schedule 深化（先服务 MatMul）

> **阶段目标**：让 Schedule 从"元数据占位"变成"真正控制代码生成"。

### 任务 4-1：把 Schedule 定义收敛到可执行版本

#### 做什么

用 `LoopOrder` 枚举替换布尔值 `reorder_ikj`，使 Schedule 能明确表达循环策略。

#### 实现逻辑

```cpp
enum class LoopOrder { kIJK, kIKJ };

struct Schedule {
    int tile_m = -1;
    int tile_n = -1;
    int tile_k = -1;
    LoopOrder order = LoopOrder::kIJK;
    bool unroll_inner = false;
};
```

#### 为什么要做

布尔值 `reorder_ikj` 只能表示两种顺序，将来可能需要更多排列。枚举更清晰、更可扩展。

#### 作用

为后续的 TilingPass、LoopReorderPass 提供清晰的调度描述。

#### 完成标准

- Schedule 能明确表达"是否切块、用什么循环顺序、是否展开"

---

### 任务 4-2：实现 TilingPass

#### 做什么

对大 MatMul 自动设置分块参数。

#### 实现逻辑

```cpp
void TilingPass::run(Graph& graph) {
    for (auto& op : graph.mutable_ops()) {
        if (op.kind != "MatMul") continue;
        auto& out = graph.tensor(op.outputs[0]);
        int M = out.shape[0], N = out.shape[1];
        int K = graph.tensor(op.inputs[0]).shape[1];

        auto pick_tile = [](int dim) -> int {
            if (dim >= 64) return 32;
            if (dim >= 32) return 16;
            return -1;  // 不切块
        };

        op.schedule.tile_m = pick_tile(M);
        op.schedule.tile_n = pick_tile(N);
        op.schedule.tile_k = pick_tile(K);
    }
}
```

#### 为什么要做

**循环分块（Tiling）是高性能计算的核心优化**。CPU 有多级 cache（L1/L2/L3），如果一次性访问整个矩阵，cache miss 会很严重。分块后，每次只处理一小块数据，能充分利用 cache。

这是 TVM 的 `te.schedule` 中 `tile()` 原语的简化版。

#### 作用

通过循环分块提高 cache 利用率，减少内存访问延迟。

#### 完成标准

- 至少部分 MatMul op 的 schedule 出现非默认 tile 值

---

### 任务 4-3：实现 LoopReorderPass 和 UnrollPass

#### 做什么

- **LoopReorderPass**：对 MatMul 将默认 IJK 顺序改为 IKJ（reduce 维度提前，利于 cache）
- **UnrollPass**：对最内层小循环做手动展开

#### 实现逻辑

**LoopReorderPass：** 当 tile 开启或 N 较大时，改为 IKJ 顺序。IKJ 顺序下，内层循环是 `C[i][j] += A[i][k] * B[k][j]`，对 B 的访问是连续的。

**UnrollPass：** 当 `tile_k` 在 4~8 范围内时，设置 `unroll_inner = true`。Codegen 不用 pragma，直接手动展开：
```c
sum += a[0] * b[0]; sum += a[1] * b[1]; sum += a[2] * b[2]; sum += a[3] * b[3];
```

#### 为什么要做

- IKJ 顺序减少 cache miss（矩阵 B 的列访问变成行访问）
- 循环展开减少分支预测开销，给编译器更多向量化机会

#### 作用

进一步优化 MatMul 的执行效率。

#### 完成标准

- schedule 确实影响最终代码形态（不只是改字段，codegen 也要改）

---

### 任务 4-4：让 codegen 真的吃到 schedule

#### 做什么

`emit_matmul()` 根据 schedule 生成不同的代码：
- schedule 默认 → `emit_matmul_naive()`
- tile 有值 → `emit_matmul_tiled()`
- order = kIKJ → 发射 IKJ 循环
- unroll_inner = true → 手动展开最内层

#### 实现逻辑

```cpp
void emit_matmul(const Op& op, ...) {
    if (op.schedule.tile_m > 0) {
        emit_matmul_tiled(op, ...);
    } else {
        emit_matmul_naive(op, ...);
    }
}
```

Tiled 版本生成 6 层循环（外层 tile + 内层 tile 内）。

#### 为什么要做

如果 schedule 只改了字段但 codegen 不读这些字段，那 schedule 系统就是摆设。**让 schedule 真正控制代码生成**是阶段 4 的核心目标。

#### 作用

实现 schedule 到代码的完整映射，证明调度系统确实有效。

#### 完成标准

- 生成的 C 代码在 schedule 开关前后明显不同
- 运行结果一致

---

## 阶段 5：ARM / QEMU 部署

> **阶段目标**：把 Host 上能运行的模型搬到 ARM 环境运行，证明这不是只在本机有效的 demo。

### 任务 5-1：打通 ARM 构建

#### 做什么

使用 ARM 交叉编译工具链编译 `run_model` 和 `libdeploy.so` 的 ARM 版本。

#### 实现逻辑

1. `cmake/toolchains/arm-linux-gnueabihf.cmake` 设置交叉编译器
2. `scripts/build_arm.sh` 调用 cmake 并指定 toolchain 文件
3. `deploy.c` 也用 ARM 编译器编译成 ARM 版 `.so`

#### 为什么要做

交叉编译是嵌入式部署的基本能力。深度学习模型通常在 x86 上编译，在 ARM（手机、开发板）上运行。

#### 作用

验证编译器输出的 C 代码可以跨平台编译和运行。

#### 完成标准

- `build-arm/` 中能产出 ARM 可执行文件和 ARM 版动态库
- `file` 命令确认是 ARM 二进制

---

### 任务 5-2：打通 qemu-arm 运行

#### 做什么

用 `qemu-arm` 用户态模拟执行 ARM 版 `run_model`。

#### 实现逻辑

```bash
qemu-arm -L /usr/arm-linux-gnueabihf \
    build-arm/run_model out/mlp input.bin output_arm.bin
```

`-L` 指定 ARM sysroot，让 QEMU 能找到动态链接库。

#### 为什么要做

不是每个人都有 ARM 开发板。QEMU 用户态模拟可以在 x86 主机上直接运行 ARM 二进制，大幅降低测试门槛。

#### 作用

无需物理 ARM 硬件即可验证 ARM 部署的正确性。

#### 完成标准

- ARM 版 `run_model` 能在 qemu-arm 下启动并产出输出文件

---

### 任务 5-3：做 Host / ARM 对比

#### 做什么

同一个模型分别在 Host 和 ARM/QEMU 上运行，比较输出是否一致。

#### 实现逻辑

1. Host 运行 → `output_host.bin`
2. ARM/QEMU 运行 → `output_arm.bin`
3. 逐元素比较，允许误差 ≤ 1e-5

#### 为什么要做

浮点运算在不同架构上可能有微小差异（IEEE 754 实现细节、编译器优化级别等），但差异应在可接受范围内。如果差异很大，说明某个环节有 bug。

#### 作用

证明编译器输出的代码在不同目标平台上行为一致。

#### 完成标准

- 同一模型在 Host 和 ARM 上结果一致或近似一致（误差 < 1e-5）

---

# 阶段 6：类型系统与多 dtype 支持

## 6.1 本阶段概述

目前编译器只处理 `float32`，但现实中深度学习模型经常使用 `float16`（半精度训练/推理）、`int8`（量化推理）等数据类型。这个阶段的目标是把类型系统从 "只有 float32" 升级为多类型支持，并实现 int8 量化推理的最小闭环。

**为什么这个阶段重要？**
- 量化推理是嵌入式部署的核心技术，int8 推理速度可比 float32 快 2-4 倍
- 类型错误是深度学习编译器中最常见的 bug 来源之一
- 完善类型系统是后续所有高级优化的基础

---

## 任务 6-1：完善 DType 系统

### 任务描述

扩展 `DType` 枚举，在现有 `kUnknown / kFloat32 / kInt32` 基础上增加 `kFloat16 / kInt8 / kUInt8`，补全 `dtype_size()` 函数，增加 `dtype_to_string()` 和 `string_to_dtype()` 辅助函数，并让 `InferShapePass` 同时推导输出 dtype。

### 实现逻辑

#### 步骤 1：扩展 DType 枚举

在 `include/tiny_tvm/ir/graph.h` 中修改：

```cpp
enum class DType {
    kUnknown = 0,
    kFloat32 = 1,
    kInt32 = 2,
    kFloat16 = 3,
    kInt8 = 4,
    kUInt8 = 5,
};
```

#### 步骤 2：补全 dtype_size()

```cpp
inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::kFloat32: return 4;
        case DType::kInt32:   return 4;
        case DType::kFloat16: return 2;
        case DType::kInt8:    return 1;
        case DType::kUInt8:   return 1;
        default: return 0;
    }
}
```

#### 步骤 3：增加转换函数

```cpp
inline std::string dtype_to_string(DType dt) {
    switch (dt) {
        case DType::kFloat32: return "float32";
        case DType::kInt32:   return "int32";
        case DType::kFloat16: return "float16";
        case DType::kInt8:    return "int8";
        case DType::kUInt8:   return "uint8";
        default: return "unknown";
    }
}

inline DType string_to_dtype(const std::string& s) {
    if (s == "float32") return DType::kFloat32;
    if (s == "int32")   return DType::kInt32;
    if (s == "float16") return DType::kFloat16;
    if (s == "int8")    return DType::kInt8;
    if (s == "uint8")   return DType::kUInt8;
    return DType::kUnknown;
}
```

#### 步骤 4：InferShapePass 推导 dtype

在 `InferShapePass` 中增加类型推导规则：
- `MatMul(int8, int8) → int32`（量化乘法结果需要更大的精度）
- `MatMul(float32, float32) → float32`
- `Add / Relu`：输出 dtype = 第一个输入 dtype

### 为什么要做

1. **扩展性**：只支持 float32 的编译器无法处理量化模型
2. **正确性**：dtype 推导是后续所有 pass 和 codegen 的基础，推导错误会导致内存越界或精度错误
3. **与 TVM 对齐**：TVM 的 `DataType` 是核心抽象之一，我们需要类似设计

### 作用

为编译器建立完整的类型基础设施，使其能表达和处理多种数据类型的 tensor。

### 完成标准

- `dtype_size()` 对 5 种已知类型返回正确字节数
- `dtype_to_string()` 和 `string_to_dtype()` 互为逆函数
- `InferShapePass` 对 int8 MatMul 推导出 int32 输出 dtype
- JSON frontend 能解析 "float16"、"int8"、"uint8" 字符串

---

## 任务 6-2：实现 TypeCheckPass

### 任务描述

在编译管线中 `InferShapePass` 之后、codegen 之前增加一个 `TypeCheckPass`，用于检查所有 op 的输入/输出 dtype 是否合法。不合法时给出具体的错误信息（哪个 op、哪个 tensor、实际 dtype 是什么）。

### 实现逻辑

#### 步骤 1：定义 TypeCheckPass

```cpp
// include/tiny_tvm/pass/graph/type_check_pass.h
class TypeCheckPass : public Pass {
public:
    std::string name() const override { return "TypeCheckPass"; }
    void run(Graph& graph) override;
};
```

#### 步骤 2：实现类型检查逻辑

```cpp
void TypeCheckPass::run(Graph& graph) {
    for (auto& op : graph.ops()) {
        switch (op.kind) {
            case OpKind::kMatMul: {
                auto& in0 = graph.tensor(op.inputs[0]);
                auto& in1 = graph.tensor(op.inputs[1]);
                if (in0.dtype != in1.dtype) {
                    throw std::runtime_error(
                        "TypeCheckPass: MatMul '" + op.name +
                        "' input dtype mismatch: " +
                        dtype_to_string(in0.dtype) + " vs " +
                        dtype_to_string(in1.dtype));
                }
                break;
            }
            case OpKind::kAdd: { /* 类似检查 */ break; }
            case OpKind::kRelu: {
                auto& in = graph.tensor(op.inputs[0]);
                auto& out = graph.tensor(op.outputs[0]);
                if (in.dtype != out.dtype) {
                    throw std::runtime_error(
                        "TypeCheckPass: Relu '" + op.name +
                        "' input/output dtype mismatch");
                }
                break;
            }
            // ... Conv2D 等
        }
    }
}
```

#### 步骤 3：注册到编译管线

在 `ttvmc compile` 的 pass 列表中，在 `InferShapePass` 之后添加 `TypeCheckPass`。

### 为什么要做

1. **尽早发现错误**：类型不匹配的错误如果传递到 codegen 会导致生成错误代码或运行时崩溃
2. **编译器基本功**：所有成熟编译器都有类型检查 pass，这是编译器正确性的第一道防线
3. **用户友好**：明确的报错信息比 segfault 更有助于调试

### 作用

作为编译管线中的一道"闸门"，确保只有类型合法的图才能进入 codegen 阶段。

### 完成标准

- 合法图（所有类型匹配）正常通过
- dtype 不匹配时抛出异常，信息包含 op 名、tensor 名、具体 dtype
- TypeCheckPass 在 PassManager 中可正常注册和运行

---

## 任务 6-3：int8 量化推理支持（入门版）

### 任务描述

让编译器能处理 int8 权重的模型，生成 int8 MatMul / Conv2D 的 C 代码，实现 int8 量化推理的最小闭环。

### 实现逻辑

#### 步骤 1：JSON Frontend 支持 int8

在 `json_frontend.cpp` 中，解析 tensor 的 "dtype" 字段，支持 "int8"。对 int8 tensor，从 JSON 中读取整数数组存入 `tensor.data`。

#### 步骤 2：Codegen 增加 int8 分支

```cpp
// 在 emit_matmul() 中
if (in0_dtype == DType::kInt8) {
    // int8 × int8 → int32 累加
    out << "    int32_t sum = 0;\n";
    out << "    for (int k = 0; k < K; k++) {\n";
    out << "      sum += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];\n";
    out << "    }\n";
    out << "    C[i*N+j] = sum;\n";
} else {
    // float32 版本
    // ...
}
```

#### 步骤 3：params.bin 支持 int8

int8 tensor 按 1 字节/元素存储到 params.bin。加载时根据 dtype 确定每个元素大小。

#### 步骤 4：准备测试模型

创建 `examples/json/mlp_int8.json`，包含 int8 权重的 MLP。

### 为什么要做

1. **实用性**：int8 量化是嵌入式 AI 部署的主流技术
2. **验证类型系统**：这是对 6-1 和 6-2 任务的端到端验证
3. **性能认知**：int8 计算量更小，内存占用更低

### 作用

证明编译器的多 dtype 支持不只是枚举值的扩展，而是从前端解析到 codegen 到运行时的完整链路。

### 完成标准

- int8 MLP 模型编译运行成功
- 输出 tensor 类型为 int32
- params.bin 中 int8 tensor 按 1 字节存储
- 数值结果正确（可通过手工计算验证小矩阵乘法）

---

# 阶段 7：图变换与算子扩展

## 7.1 本阶段概述

到目前为止编译器只支持 MatMul / Add / Relu / Conv2D 等少量算子。真实的 CNN 还需要 MaxPool、BatchNorm、Softmax 等。这个阶段的目标是：
1. 扩展算子覆盖面
2. 实现 BatchNorm 折叠优化
3. 把 FusePass 升级为可扩展的模式匹配框架

**为什么这个阶段重要？**
- 算子覆盖面决定了编译器能处理多少真实模型
- BatchNorm 折叠是 CNN 部署中最基本的优化之一
- 可扩展的融合框架是编译器长期演进的基础

---

## 任务 7-1：补充常用算子

### 任务描述

新增 4 个常用算子：`MaxPool2D`、`GlobalAvgPool2D`、`Softmax`、`BatchNorm`（推理模式），包括 shape 推导和 C 代码生成。

### 实现逻辑

#### MaxPool2D

**OpKind**：`kMaxPool2D`

**attrs**：`kernel_shape`（如 [2,2]）、`strides`（如 [2,2]）、`pads`（如 [0,0,0,0]）

**Shape 推导**：
```
输入: [N, C, H, W]
OH = (H + pad_top + pad_bottom - KH) / stride_h + 1
OW = (W + pad_left + pad_right - KW) / stride_w + 1
输出: [N, C, OH, OW]
```

**Codegen**：
```cpp
for (n) for (c) for (oh) for (ow) {
    float max_val = -FLT_MAX;
    for (kh) for (kw) {
        int ih = oh * stride_h - pad_top + kh;
        int iw = ow * stride_w - pad_left + kw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            max_val = fmax(max_val, input[n][c][ih][iw]);
    }
    output[n][c][oh][ow] = max_val;
}
```

#### GlobalAvgPool2D

**OpKind**：`kGlobalAvgPool2D`

**Shape 推导**：`[N,C,H,W] → [N,C,1,1]`

**Codegen**：
```cpp
for (n) for (c) {
    float sum = 0;
    for (h) for (w) sum += input[n][c][h][w];
    output[n][c][0][0] = sum / (H * W);
}
```

#### Softmax

**OpKind**：`kSoftmax`

**attrs**：`axis`（默认 -1，即最后一个维度）

**Shape 推导**：输出 shape = 输入 shape

**Codegen**（数值稳定版）：
```cpp
// 1. 找 max
float max_val = -FLT_MAX;
for (j) max_val = fmax(max_val, x[j]);
// 2. exp(x - max)
float sum = 0;
for (j) { exp_vals[j] = expf(x[j] - max_val); sum += exp_vals[j]; }
// 3. 归一化
for (j) y[j] = exp_vals[j] / sum;
```

#### BatchNorm（推理模式）

**OpKind**：`kBatchNorm`

**输入**：`x`（数据）、`scale`（gamma）、`bias`（beta）、`mean`、`var`

**attrs**：`epsilon`（默认 1e-5）

**Shape 推导**：输出 shape = 输入 shape

**Codegen**：
```cpp
for (n) for (c) for (h) for (w) {
    float norm = (x[n][c][h][w] - mean[c]) / sqrtf(var[c] + eps);
    output[n][c][h][w] = scale[c] * norm + bias[c];
}
```

### 为什么要做

1. **模型覆盖**：没有 MaxPool 和 BatchNorm，大部分 CNN（ResNet、VGG）都无法编译
2. **完整性**：Softmax 是分类任务必需的；GlobalAvgPool 是现代网络的标配
3. **算子扩展模式**：这 4 个算子展示了"增加新算子"的标准流程

### 作用

让编译器从"能跑简单 MLP/CNN"升级为"能处理真实 CNN 的主要算子"。

### 完成标准

- 4 个新算子的 shape infer 和 codegen 都正确
- 含新算子的模型能通过 `ttvmc compile` 编译并运行
- 算子数值正确（可通过小输入手工验证）

---

## 任务 7-2：实现 BatchNormFoldPass

### 任务描述

当 `BatchNorm` 紧跟 `Conv2D` 时，把 BN 的参数（gamma, beta, mean, var）折叠进 Conv 的 weight 和 bias 中，从而消除 BN op，减少推理时计算量。

### 实现逻辑

#### 数学推导

Conv2D 输出 `y[oc] = sum(w[oc] * x) + conv_bias[oc]`

BN 输出 `z[oc] = gamma[oc] * (y[oc] - mean[oc]) / sqrt(var[oc] + eps) + beta[oc]`

折叠后：
```
scale[oc] = gamma[oc] / sqrt(var[oc] + eps)
new_weight[oc, ic, kh, kw] = scale[oc] * old_weight[oc, ic, kh, kw]
new_bias[oc] = scale[oc] * (old_conv_bias[oc] - mean[oc]) + beta[oc]
```

#### 代码实现

```cpp
void BNFoldPass::run(Graph& graph) {
    for (int i = 0; i < graph.num_ops() - 1; i++) {
        auto& conv_op = graph.op(i);
        auto& bn_op = graph.op(i + 1);
        if (conv_op.kind != OpKind::kConv2D ||
            bn_op.kind != OpKind::kBatchNorm)
            continue;
        // 检查 conv 输出是 bn 的输入
        if (conv_op.outputs[0] != bn_op.inputs[0]) continue;

        // 获取 BN 参数
        auto& gamma = graph.tensor(bn_op.inputs[1]);
        auto& beta  = graph.tensor(bn_op.inputs[2]);
        auto& mean  = graph.tensor(bn_op.inputs[3]);
        auto& var   = graph.tensor(bn_op.inputs[4]);
        float eps = bn_op.attrs.count("epsilon") ?
                    std::stof(bn_op.attrs["epsilon"]) : 1e-5f;

        // 计算 scale
        int oc = gamma.shape[0];
        std::vector<float> scale(oc);
        for (int c = 0; c < oc; c++)
            scale[c] = gamma.data_as<float>()[c] /
                       std::sqrt(var.data_as<float>()[c] + eps);

        // 折叠 weight
        auto& weight = graph.tensor(conv_op.inputs[1]);
        // weight shape: [OC, IC, KH, KW]
        int per_oc = weight.num_elements() / oc;
        for (int c = 0; c < oc; c++)
            for (int j = 0; j < per_oc; j++)
                weight.data_as<float>()[c * per_oc + j] *= scale[c];

        // 折叠 bias
        auto& conv_bias = graph.tensor(conv_op.inputs[2]);
        for (int c = 0; c < oc; c++) {
            float new_b = scale[c] * (conv_bias.data_as<float>()[c] -
                          mean.data_as<float>()[c]) +
                          beta.data_as<float>()[c];
            conv_bias.data_as<float>()[c] = new_b;
        }

        // 删除 BN op，更新 conv 输出为 bn 输出
        conv_op.outputs[0] = bn_op.outputs[0];
        graph.remove_op(i + 1);
    }
}
```

### 为什么要做

1. **性能**：BN 折叠后推理时不再需要额外的 BN 计算，节省 ~10-20% 时间
2. **行业标准**：几乎所有推理框架（TFLite、ONNX Runtime、TensorRT）都做 BN 折叠
3. **验证 Pass 框架**：这是一个"改变图结构"的 pass，验证图编辑能力

### 作用

作为图级别优化的典型案例，展示编译器如何通过数学等价变换来优化模型。

### 完成标准

- Conv + BN 被折叠为单个 Conv
- BN op 从图中消失
- 折叠前后推理结果数值一致（误差 < 1e-5）

---

## 任务 7-3：扩展 FusePass 为模式匹配框架

### 任务描述

把现有的硬编码 FusePass（只处理 Conv+Relu）升级为基于模式描述的可扩展融合框架。新增融合规则只需注册一个 `FusePattern` 结构体。

### 实现逻辑

#### 步骤 1：定义 FusePattern

```cpp
struct FusePattern {
    std::string name;  // 融合后的名字，如 "ConvRelu"
    OpKind head;       // 第一个 op
    OpKind tail;       // 第二个 op
    OpKind fused_kind; // 融合后的 op kind
};
```

#### 步骤 2：FusePass 维护模式列表

```cpp
class FusePass : public Pass {
    std::vector<FusePattern> patterns_;
public:
    FusePass() {
        patterns_.push_back({"ConvRelu", OpKind::kConv2D,
                             OpKind::kRelu, OpKind::kConvRelu});
        patterns_.push_back({"MatMulAdd", OpKind::kMatMul,
                             OpKind::kAdd, OpKind::kMatMulAdd});
    }
    void add_pattern(FusePattern p) { patterns_.push_back(p); }
};
```

#### 步骤 3：通用匹配引擎

```cpp
void FusePass::run(Graph& graph) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& pat : patterns_) {
            for (int i = 0; i < graph.num_ops() - 1; i++) {
                auto& op1 = graph.op(i);
                auto& op2 = graph.op(i + 1);
                if (op1.kind != pat.head || op2.kind != pat.tail)
                    continue;
                // 检查 op1 输出是 op2 唯一输入
                if (op1.outputs[0] != op2.inputs[0]) continue;
                // 融合
                op1.kind = pat.fused_kind;
                op1.name = pat.name + "_" + op1.name;
                op1.outputs = op2.outputs;
                graph.remove_op(i + 1);
                changed = true;
                break;
            }
        }
    }
}
```

### 为什么要做

1. **可维护性**：硬编码的融合规则难以维护和扩展
2. **可扩展性**：新增融合规则只需一行 `add_pattern(...)`
3. **与 TVM 对齐**：TVM Relay 的 FuseOps 也是基于模式匹配的框架

### 作用

把编译器的融合能力从"只能融合 Conv+Relu"升级为"可融合任意注册的算子对"。

### 完成标准

- `Conv + Relu → ConvRelu` 和 `MatMul + Add → MatMulAdd` 都被正确融合
- 新增融合规则只需一行注册代码
- 融合前后数值结果一致

---

# 阶段 8：真正的 TensorIR 雏形（⭐ 最重要）

## 8.1 本阶段概述

这是整个项目中**最核心**的阶段。此前的 codegen 直接从 Op 生成 C 代码，Schedule 只是一些标志位。这个阶段引入 **LoopIR**，把计算描述（Op → 做什么运算）和调度描述（LoopIR → 怎么执行循环）完全分离。

**这正是 TVM TensorIR 的核心设计思想：**
- Op 只描述"语义"（MatMul 是矩阵乘法）
- LoopIR 描述"实现"（用哪些循环、什么顺序、如何分块）
- Schedule 原语（split、reorder、fuse）操作 LoopIR，而不是直接改 codegen

**为什么这个阶段最重要？**
- 这是"教学级 TVM"和"真正理解 TVM"的分水岭
- 计算-调度分离是 TVM 区别于其他框架的核心创新
- 完成这个阶段后，你才能说"我理解 TVM 的架构"

---

## 任务 8-1：抽象循环 IR（LoopIR）

### 任务描述

定义 LoopIR 的核心数据结构：`LoopVar`（循环变量）、`Block`（一个计算块，对应一个 Op）、`LoopProgram`（整个程序的循环表示）。

### 实现逻辑

#### LoopVar

```cpp
struct LoopVar {
    std::string name;    // 如 "i", "j", "k"
    int64_t extent = 0;  // 循环范围 [0, extent)
};
```

一个 LoopVar 对应 for 循环中的一个循环变量。

#### Block

```cpp
struct Block {
    std::string name;                     // 如 "matmul_0"
    std::vector<LoopVar> loop_vars;       // 所有循环变量
    std::vector<int> loop_order;          // 执行顺序（索引到 loop_vars）
    std::string compute_body;             // C 代码片段
    std::vector<int> read_tensors;        // 读取的 tensor index
    std::vector<int> write_tensors;       // 写入的 tensor index

    struct TileInfo {
        int var_index = -1;     // 被 split 的变量索引
        int factor = -1;        // split 因子
        int outer_index = -1;   // split 后外层变量索引
        int inner_index = -1;   // split 后内层变量索引
    };
    std::vector<TileInfo> tiles;          // split 记录
    bool unroll_innermost = false;        // 是否展开最内层
};
```

Block 是 LoopIR 的核心抽象。每个 Block 对应一个 Op 的循环嵌套。`loop_order` 定义了循环从外到内的顺序，split 操作会添加新的 LoopVar 并更新 loop_order。

#### LoopProgram

```cpp
struct LoopProgram {
    std::vector<Block> blocks;
    const Graph* source_graph = nullptr;
};
```

#### 设计要点

- **MatMul [M,K] × [K,N]**：3 个 loop_vars（i:M, j:N, k:K），默认 loop_order = [0,1,2]
- **Conv2D**：7 个 loop_vars（n, oc, oh, ow, ic, kh, kw）
- **逐元素 op（Add/Relu）**：按 tensor rank 生成 loop_vars

### 为什么要做

1. **核心抽象**：LoopIR 是计算-调度分离的基础数据结构
2. **可操作性**：有了 LoopIR，schedule 原语才有操作对象
3. **与 TVM 对齐**：TVM 的 `tir.PrimFunc` 本质上就是 LoopIR

### 作用

建立 LoopIR 数据结构，为后续的 lowering、schedule 原语、codegen 提供统一的中间表示。

### 完成标准

- LoopVar / Block / LoopProgram 结构体定义完成
- 能手工构造 MatMul 的 Block（3 个 loop_vars，正确的 compute_body）
- 能手工构造 Conv2D 的 Block（7 个 loop_vars）
- Block 的 loop_order 可以被修改

---

## 任务 8-2：实现 LowerToLoopIRPass 和 Schedule 原语

### 任务描述

实现从 Graph 到 LoopProgram 的 lowering pass，以及 split / reorder / fuse 三个 schedule 原语。

### 实现逻辑

#### LowerToLoopIRPass

```cpp
class LowerToLoopIRPass {
public:
    LoopProgram lower(const Graph& graph);
private:
    Block lower_matmul(const Op& op, const Graph& graph);
    Block lower_conv2d(const Op& op, const Graph& graph);
    Block lower_elementwise(const Op& op, const Graph& graph);
};
```

**MatMul lowering**：
```cpp
Block lower_matmul(const Op& op, const Graph& graph) {
    auto& A = graph.tensor(op.inputs[0]);  // [M, K]
    auto& B = graph.tensor(op.inputs[1]);  // [K, N]
    Block block;
    block.name = op.name;
    block.loop_vars = {{"i", A.shape[0]}, {"j", B.shape[1]}, {"k", A.shape[1]}};
    block.loop_order = {0, 1, 2};  // i, j, k
    block.compute_body = "C[i * N + j] += A[i * K + k] * B[k * N + j];";
    block.read_tensors = {op.inputs[0], op.inputs[1]};
    block.write_tensors = {op.outputs[0]};
    return block;
}
```

#### split 原语

```cpp
void split(Block& block, int var_index, int factor) {
    auto& var = block.loop_vars[var_index];
    int outer_extent = (var.extent + factor - 1) / factor;

    // 创建 outer 和 inner 变量
    LoopVar outer{var.name + "_outer", outer_extent};
    LoopVar inner{var.name + "_inner", factor};

    int outer_idx = block.loop_vars.size();
    int inner_idx = outer_idx + 1;
    block.loop_vars.push_back(outer);
    block.loop_vars.push_back(inner);

    // 更新 loop_order：把 var_index 替换为 outer, inner
    auto it = std::find(block.loop_order.begin(),
                        block.loop_order.end(), var_index);
    *it = outer_idx;
    block.loop_order.insert(it + 1, inner_idx);

    // 记录 tile 信息
    block.tiles.push_back({var_index, factor, outer_idx, inner_idx});
}
```

#### reorder 原语

```cpp
void reorder(Block& block, const std::vector<int>& new_order) {
    block.loop_order = new_order;
}
```

#### fuse_loops 原语

```cpp
void fuse_loops(Block& block, int var1_index, int var2_index) {
    auto& v1 = block.loop_vars[var1_index];
    auto& v2 = block.loop_vars[var2_index];
    int64_t new_extent = v1.extent * v2.extent;

    LoopVar fused{v1.name + "_" + v2.name, new_extent};
    int fused_idx = block.loop_vars.size();
    block.loop_vars.push_back(fused);

    // 更新 loop_order：把 var1, var2 替换为 fused
    auto it1 = std::find(block.loop_order.begin(),
                         block.loop_order.end(), var1_index);
    *it1 = fused_idx;
    block.loop_order.erase(
        std::find(block.loop_order.begin(),
                  block.loop_order.end(), var2_index));
}
```

### 为什么要做

1. **Lowering 是编译器标准流程**：从高级 IR（Graph）到低级 IR（LoopIR）的转换
2. **Schedule 原语是 TVM 的核心特性**：用户通过 schedule 原语控制代码生成
3. **分离关注点**：Op 只管语义，LoopIR 管实现细节

### 作用

- LowerToLoopIRPass 把 Graph 中的每个 Op 转化为 LoopIR Block
- Schedule 原语让用户可以优化 LoopIR 的循环结构
- 这两者结合实现了"同一个 MatMul，不同的执行方式"

### 完成标准

- MatMul / Conv2D / Add / Relu 都能正确 lower 为 Block
- split(i, 16) 后 loop_vars 增加 i_outer 和 i_inner
- reorder 后 loop_order 改变
- fuse 后两个变量合并为一个

---

## 任务 8-3：Codegen 从 LoopIR 生成 C 代码

### 任务描述

新建 `loop_ir_codegen.cpp`，从 LoopProgram 递归生成嵌套 for 循环的 C 代码。修改 `ttvmc compile` 管线，在 lowering 后使用 LoopIR codegen。

### 实现逻辑

#### 核心函数

```cpp
std::string emit_c_from_loop_program(const LoopProgram& prog) {
    std::ostringstream out;
    // 文件头
    out << "#include <stdint.h>\n#include <math.h>\n\n";

    // 为每个 block 生成一个函数
    for (auto& block : prog.blocks) {
        emit_block(out, block, *prog.source_graph);
    }

    // 生成 tiny_tvm_run 入口
    emit_run_function(out, prog);

    return out.str();
}
```

#### Block 代码生成

```cpp
void emit_block(std::ostream& out, const Block& block,
                const Graph& graph) {
    int indent = 1;
    // 按 loop_order 发射嵌套 for
    for (int idx : block.loop_order) {
        auto& var = block.loop_vars[idx];
        // 检查是否是 tile 的内层
        bool is_tile_inner = false;
        int tile_factor = 0;
        for (auto& t : block.tiles) {
            if (t.inner_index == idx) {
                is_tile_inner = true;
                tile_factor = t.factor;
                break;
            }
        }

        std::string spaces(indent * 2, ' ');
        if (block.unroll_innermost &&
            idx == block.loop_order.back()) {
            // 展开最内层
            for (int u = 0; u < var.extent; u++) {
                out << spaces << "// " << var.name << " = " << u << "\n";
                out << spaces << "{ int " << var.name << " = " << u << ";\n";
                out << spaces << "  " << block.compute_body << "\n";
                out << spaces << "}\n";
            }
        } else {
            out << spaces << "for (int " << var.name
                << " = 0; " << var.name << " < " << var.extent
                << "; " << var.name << "++) {\n";
            indent++;
        }
    }

    // 最内层循环体
    if (!block.unroll_innermost) {
        std::string spaces(indent * 2, ' ');
        out << spaces << block.compute_body << "\n";
    }

    // 关闭循环
    for (int i = block.loop_order.size() - 1; i >= 0; i--) {
        if (block.unroll_innermost &&
            block.loop_order[i] == block.loop_order.back())
            continue;
        indent--;
        std::string spaces(indent * 2, ' ');
        out << spaces << "}\n";
    }
}
```

#### 管线集成

在 `ttvmc compile` 中：
1. Parse → InferShape → ... → 现有 passes
2. **新增**：LowerToLoopIRPass → 可选 schedule 操作 → LoopIR Codegen
3. 保留旧 codegen 作为 `--legacy-codegen` fallback

### 为什么要做

1. **闭环验证**：LoopIR 只有能生成正确代码才有意义
2. **与旧 codegen 对比**：证明新的 LoopIR 方案功能等价
3. **schedule 效果可见**：split + reorder 后生成的代码结构确实不同

### 作用

把 LoopIR 从"数据结构"变成"可执行的编译流程"，完成计算-调度分离的完整链路。

### 完成标准

- LoopIR codegen 生成的代码能编译运行，结果与旧 codegen 一致
- split 后代码有双层循环（外层步进 factor，内层范围 factor）
- reorder 后循环嵌套顺序改变
- 管线可切换新旧 codegen

---

# 阶段 9：性能分析与 Auto-Tuning 雏形

## 9.1 本阶段概述

有了 LoopIR 和 schedule 原语后，自然的下一步是：怎么知道哪种 schedule 更快？这个阶段实现：
1. 按 op 粒度的性能 profiling
2. 基于 GridSearch 的简单 auto-tuner
3. `ttvmc tune` 命令

**为什么这个阶段重要？**
- 手动调参是不可扩展的，需要自动化搜索
- 这正是 TVM AutoTVM / AutoScheduler 的思想原型
- Profiling 是任何性能优化工作的基础

---

## 任务 9-1：编译时性能 Profile 基础设施

### 任务描述

在生成的 `deploy.c` 中为每个 op 插入计时桩（条件编译），让 `run_model --profile` 能输出每个 op 的执行时间。

### 实现逻辑

#### 步骤 1：Codegen 插入计时代码

在 `emit_c_module()`（或 LoopIR codegen）中：

```cpp
// 条件编译的计时支持
out << "#ifdef TINY_TVM_PROFILE\n";
out << "#include <time.h>\n";
out << "static double op_times[" << num_ops << "];\n";
out << "#endif\n\n";

// 每个 op 前后
out << "#ifdef TINY_TVM_PROFILE\n";
out << "  struct timespec _ts_start, _ts_end;\n";
out << "  clock_gettime(CLOCK_MONOTONIC, &_ts_start);\n";
out << "#endif\n";
// ... op 代码 ...
out << "#ifdef TINY_TVM_PROFILE\n";
out << "  clock_gettime(CLOCK_MONOTONIC, &_ts_end);\n";
out << "  op_times[" << op_idx << "] = "
    << "(_ts_end.tv_sec - _ts_start.tv_sec) * 1e6 + "
    << "(_ts_end.tv_nsec - _ts_start.tv_nsec) / 1e3;\n";
out << "#endif\n";
```

#### 步骤 2：导出 profile 数据

```cpp
out << "#ifdef TINY_TVM_PROFILE\n";
out << "void tiny_tvm_get_profile(double* out, int max_ops) {\n";
out << "    for (int i = 0; i < " << num_ops << " && i < max_ops; i++)\n";
out << "        out[i] = op_times[i];\n";
out << "}\n";
out << "#endif\n";
```

#### 步骤 3：run_model --profile

1. 编译时加 `-DTINY_TVM_PROFILE`
2. 运行模型多次（如 10 次），取中位数
3. 输出 `profile_report.json`：
   ```json
   {
     "ops": [
       {"name": "matmul_0", "time_us": 123.4},
       {"name": "add_0", "time_us": 5.6}
     ],
     "total_us": 129.0
   }
   ```

### 为什么要做

1. **量化优化效果**：没有计时数据，无法判断 schedule 变化是否真的更快
2. **Auto-Tuning 基础**：tuner 需要 profile 结果来评估候选方案
3. **性能分析能力**：找出瓶颈 op 是优化的第一步

### 作用

为编译器增加"测量"能力，把性能优化从猜测变成数据驱动。

### 完成标准

- `run_model --profile` 输出每个 op 的执行时间
- `profile_report.json` 格式正确、可解析
- 时间数值合理（不为 0，不为负数）

---

## 任务 9-2：实现简单 Auto-Tuner（GridSearch 版）

### 任务描述

对 MatMul 的 tile 参数进行网格搜索（GridSearch），自动找出最快的 (tile_m, tile_n, tile_k) 组合。

### 实现逻辑

#### 步骤 1：定义搜索空间

```cpp
class GridSearchTuner {
    std::vector<int> candidates_ = {8, 16, 32, 64};
public:
    struct TuneResult {
        int tile_m, tile_n, tile_k;
        double time_us;
    };

    TuneResult tune(const Graph& graph, int op_index);
};
```

#### 步骤 2：搜索循环

```cpp
TuneResult GridSearchTuner::tune(const Graph& graph, int op_index) {
    TuneResult best = {8, 8, 8, 1e18};

    for (int tm : candidates_) {
        for (int tn : candidates_) {
            for (int tk : candidates_) {
                // 1. 设置 schedule
                Schedule sched;
                sched.tile_m = tm; sched.tile_n = tn; sched.tile_k = tk;

                // 2. Lower to LoopIR + apply schedule
                auto prog = lower_with_schedule(graph, op_index, sched);

                // 3. Codegen
                auto code = emit_c_from_loop_program(prog);

                // 4. Compile + Profile
                double time_us = compile_and_profile(code);

                // 5. 记录
                if (time_us < best.time_us) {
                    best = {tm, tn, tk, time_us};
                }

                printf("  tile=(%d,%d,%d) → %.1f us\n", tm, tn, tk, time_us);
            }
        }
    }
    return best;
}
```

#### 步骤 3：输出 best_schedule.json

```json
{
  "ops": [
    {
      "name": "matmul_0",
      "tile_m": 32,
      "tile_n": 32,
      "tile_k": 16,
      "time_us": 45.2
    }
  ]
}
```

### 为什么要做

1. **自动化优化**：手动尝试 tile 组合不可扩展，需要自动搜索
2. **TVM 核心思想**：AutoTVM 就是通过搜索找最优 schedule
3. **闭环验证**：验证了 LoopIR + schedule 原语 + codegen + profile 的完整链路

### 作用

把"人肉调参"变成"自动搜索"，这是 TVM AutoTVM 的最小化原型。

### 完成标准

- GridSearch 遍历所有 (tile_m, tile_n, tile_k) 组合
- 最终选出的配置不慢于默认配置
- 输出 `best_schedule.json`

---

## 任务 9-3：ttvmc tune 子命令

### 任务描述

为 `ttvmc` CLI 增加 `tune` 子命令，用户通过一条命令就能完成模型的自动调参。

### 实现逻辑

#### 命令格式

```
./build/ttvmc tune model.json -o out/ --schedule-out best_schedule.json
```

#### 实现步骤

1. 在 `ttvmc.cpp` 中新增 `tune` 子命令分支。
2. 解析模型 → InferShape → 对每个 MatMul op 调用 `GridSearchTuner::tune()`。
3. 汇总结果，写入 `best_schedule.json`。
4. 打印调参进度和最终结果。

#### compile 读取 schedule

`ttvmc compile` 增加 `--schedule` 参数：
```
./build/ttvmc compile model.json -o out/ --schedule best_schedule.json
```

从 JSON 读取每个 op 的 tile 参数，应用到 LoopIR，然后 codegen。

### 为什么要做

1. **完整的工具链**：tune → compile → run 三步走
2. **用户体验**：一条命令完成调参，不需要手动编辑 schedule
3. **与 TVM 对齐**：`tvmc tune` + `tvmc compile` 就是这个工作流

### 作用

把 auto-tuning 能力暴露给用户，完成 tiny-tvm 工具链的最后一块拼图。

### 完成标准

- `ttvmc tune model.json -o out/ --schedule-out best.json` 运行成功
- `ttvmc compile model.json -o out/ --schedule best.json` 能读取和应用调参结果
- 调参后的模型性能不低于默认配置
