# Tiny TVM 工程化实现计划（无 BRAM / 练手版）

这份文档的目标不是描述“想法”，而是给你一份可以照着落地的实现路线。  
项目边界保持和现有 `tiny_tvm/plan.md` 一致：**不做 BRAM，只做 CPU + QEMU ARM + 完整编译链路**。

---

## 1. 项目定位

你要完成的是一个 **Mini TVM / Mini 深度学习编译器**，核心链路如下：

```text
JSON / ONNX
    ->
Frontend
    ->
Graph IR
    ->
Pass Pipeline
    ->
Schedule
    ->
C Codegen
    ->
Runtime
    ->
PC / QEMU ARM 运行
```

这个项目的重点不是“支持很多算子”，而是：

- 把编译器主链路完整跑通
- 把工程结构搭清楚
- 每个阶段都有可验证的交付物
- 让你能在实现过程中练到 IR / Pass / Codegen / Runtime / Schedule 的基本功

---

## 2. 范围与边界

### 2.1 本项目要做的内容

- JSON Frontend
- ONNX 子集 Frontend
- Graph IR
- PassManager + 多类 Pass
- Memory Planning
- Schedule 描述与调度 Pass
- C 代码生成
- Host Runtime
- ARM 交叉编译与 QEMU ARM 运行

### 2.2 明确不做的内容

- BRAM / FPGA / HLS
- 自动调优（auto-tuning）
- 多后端异构执行
- 量化、混合精度
- 完整 ONNX 算子覆盖
- 复杂图分区、分布式执行

先把主干跑通，比一开始追求“大而全”更重要。

---

## 3. 最终交付物

项目完成后，建议至少具备这些产物：

- 一个编译器入口程序，例如 `ttvmc`
- 一个可读的中间产物目录，例如：
  - `graph.json`
  - `params.bin`
  - `deploy.c`
  - `libdeploy.so`
- 一个 Host 侧 runtime，可在 PC 上运行
- 一套 ARM 侧构建与运行脚本，可在 QEMU ARM 中运行
- 至少 2 个示例模型：
  - MLP / MatMul + Relu 小模型
  - Conv + Relu 小 CNN
- 一组基础测试：
  - IR 测试
  - Pass 测试
  - Codegen / Runtime 端到端测试

---

## 4. 工程化目标

相比原始计划，这一版更强调“工程化”：

- **目录清晰**：头文件、源码、测试、样例、脚本分开
- **职责清晰**：Frontend / IR / Pass / Codegen / Runtime 各自边界明确
- **可构建**：任意阶段都能 `cmake` 构建
- **可测试**：每个阶段都补最小测试
- **可调试**：Pass 前后可 dump，关键数据结构可打印
- **可扩展**：后续加算子、加 Pass、加 Schedule 不需要推倒重来

一句话：**这个项目不是写 demo，而是写一个小而完整、结构干净的编译器练手工程。**

---

## 5. 推荐目录结构

建议从一开始就按下面的结构组织工程：

```text
tiny_tvm/
├── CMakeLists.txt
├── cmake/
├── docs/
│   ├── architecture.md
│   └── examples.md
├── include/
│   └── tiny_tvm/
│       ├── frontend/
│       ├── ir/
│       ├── pass/
│       ├── schedule/
│       ├── codegen/
│       ├── runtime/
│       └── support/
├── src/
│   ├── frontend/
│   │   ├── json/
│   │   └── onnx/
│   ├── ir/
│   ├── pass/
│   │   ├── graph/
│   │   ├── op/
│   │   ├── memory/
│   │   └── schedule/
│   ├── schedule/
│   ├── codegen/
│   ├── runtime/
│   ├── support/
│   └── tools/
├── tests/
│   ├── ir/
│   ├── frontend/
│   ├── pass/
│   ├── codegen/
│   └── runtime/
├── examples/
│   ├── json/
│   └── onnx/
├── scripts/
│   ├── build_host.sh
│   ├── build_arm.sh
│   └── run_qemu.sh
└── third_party/
```

如果你想尽量简化，也可以先不拆 `docs/` 和 `third_party/`，但 `include/`、`src/`、`tests/`、`examples/` 最好一开始就建立。

---

## 6. 模块职责划分

| 模块 | 职责 | 输出 |
|------|------|------|
| `frontend/json` | 解析自定义 JSON 模型，构建 Graph | `Graph` |
| `frontend/onnx` | 解析 ONNX 子集并映射到内部算子 | `Graph` |
| `ir` | 定义 Tensor / Op / Graph / Attr / DType | 核心数据结构 |
| `pass/graph` | 图级优化，如 shape infer、DCE、fusion | 变换后的 `Graph` |
| `pass/op` | 算子标准化、layout 变换 | 统一后的算子形式 |
| `pass/memory` | 生命周期分析、内存复用、offset 分配 | 内存计划信息 |
| `pass/schedule` | 默认调度、tiling、reorder、unroll | `Schedule` 信息 |
| `schedule` | 表达循环变换的元数据或 API | codegen 输入 |
| `codegen` | 从 Graph + Schedule 生成 C 代码 | `deploy.c` |
| `runtime` | 加载 graph/params，调用生成函数运行 | 推理执行结果 |
| `support` | 日志、错误处理、序列化、文件工具 | 公共工具 |

---

## 7. 建议的核心数据结构

原始计划里的数据结构可以继续用，但为了更工程化，建议从一开始多留几个关键字段。

### 7.1 Tensor

```cpp
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype;
    size_t nbytes = 0;
    size_t offset = 0;
    bool is_constant = false;
};
```

### 7.2 Op

```cpp
struct Op {
    std::string kind;
    std::vector<int> inputs;
    std::vector<int> outputs;
    AttrMap attrs;
    Schedule schedule;
};
```

这里建议 `inputs/outputs` 存 tensor index，而不是直接存裸指针。  
原因很简单：后期做序列化、重排、拷贝、图变换时，index 方案更稳。

### 7.3 Graph

```cpp
struct Graph {
    std::vector<Tensor> tensors;
    std::vector<Op> ops;
    std::vector<int> graph_inputs;
    std::vector<int> graph_outputs;
};
```

### 7.4 Pass 接口

```cpp
class Pass {
public:
    virtual ~Pass() = default;
    virtual std::string name() const = 0;
    virtual void run(Graph& graph) = 0;
};
```

### 7.5 Schedule

```cpp
struct Schedule {
    int tile_m = -1;
    int tile_n = -1;
    int tile_k = -1;
    bool reorder_ikj = false;
    bool unroll_inner = false;
};
```

阶段 4 再把它升级成更像 loop transform 的表示，不需要一开始就过度设计。

---

## 8. 统一构建与运行方式

为了把项目做成“工程”，建议尽早固定下面这套工作流。

### 8.1 Host 构建

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

### 8.2 编译模型

```bash
./build/tools/ttvmc compile examples/json/mlp.json -o out/mlp
```

### 8.3 Host 运行

```bash
./build/tools/run_model out/mlp
```

### 8.4 ARM 运行

```bash
./scripts/build_arm.sh
./scripts/run_qemu.sh
```

即使前期这些命令还只是空壳，也建议先把入口名字和参数习惯固定下来。

---

## 9. 分阶段实施计划

下面是建议你真正执行时采用的阶段划分。每一阶段都包含：

- 目标
- 必做任务
- 阶段输出
- 验收标准

### 阶段 0：工程骨架

#### 目标

先把“项目像个项目”这件事做好：能构建、能测试、能跑最小空流程。

#### 必做任务

- 建立 `CMakeLists.txt`
- 建立 `include/`、`src/`、`tests/`、`examples/`
- 写出最小 `Graph` / `Tensor` / `Op` / `Pass` / `PassManager`
- 提供一个空的 CLI 程序，例如 `ttvmc`
- 建一个最简单的样例 graph 或 JSON 输入
- 建一个 smoke test，验证空 pass pipeline 可以执行

#### 阶段输出

- 可编译的工程骨架
- 空的 IR 和 Pass 框架
- 一条可执行的最小命令链

#### 验收标准

- `cmake --build build` 可以成功
- `ctest` 至少有 1 个测试通过
- 可以构造一个最小 graph 并执行空的 pass pipeline

---

### 阶段 1：最小闭环（JSON -> IR -> Codegen -> Host Runtime）

#### 目标

先跑通一个最小但完整的编译链路，不要急着追求优化。

#### 必做任务

- 实现 JSON Frontend
- 支持最小算子集合：
  - `Input`
  - `Constant`
  - `MatMul`
  - `Add`
  - `Relu`
- 建立 Graph 的拓扑顺序
- 实现 `InferShapePass`
- 实现 `NaiveMemoryPlanner`
- 实现默认 `Schedule`
- 生成最朴素的 C 代码
- 做一个 Host Runtime，能够加载并调用生成的函数

#### 阶段输出

- `examples/json/mlp.json`
- 编译产物：
  - `deploy.c`
  - `graph.json`
  - `params.bin`（此阶段可以先做成最简版）
- Host 侧运行结果

#### 验收标准

- 小 MLP 模型能在 PC 上跑通
- 输出与参考实现一致
- 代码生成出来的循环逻辑是可读的

---

### 阶段 2：推理系统增强（Params / Conv / Fusion）

#### 目标

把项目从“最小 demo”推进到“能支撑简单 CNN”的程度。

#### 必做任务

- 完整实现 `params.bin` 导出与加载
- 实现 `Conv2D` 的 naive 版本
- 完善 `dtype` / `nbytes` 计算
- 实现 `ConstantFoldPass`
- 实现 `FusePass`：
  - `Conv + Relu -> ConvRelu`
- Runtime 支持加载参数、执行推理、输出结果

#### 阶段输出

- 一个简单 CNN 样例
- 参数文件导出 / 加载闭环
- 至少 1 个 fusion pass

#### 验收标准

- 小 CNN 能在 Host 上跑通
- 参数加载结果正确
- fusion 前后结果一致

---

### 阶段 3：编译器化（ONNX + 完整 Pass 体系）

#### 目标

从“能跑”升级到“像编译器”。

#### 必做任务

- 增加 ONNX Frontend（先支持子集）
- 做算子标准化，例如：
  - `Gemm -> MatMul + Add`
- 增强 `InferShapePass`
- 实现 `DeadCodeEliminationPass`
- 实现 `LivenessAnalysisPass`
- 实现 `MemoryReusePass`
- 输出 pass 前后 dump，便于调试
- 生成一份 compile report（可选但推荐）

#### 建议先支持的 ONNX 算子

- `MatMul`
- `Gemm`
- `Add`
- `Relu`
- `Conv`
- `Flatten`
- `Reshape`

#### 阶段输出

- ONNX 子集导入能力
- 更完整的 pass pipeline
- 内存复用后的图与内存报告

#### 验收标准

- 至少 1 个 ONNX 小模型能跑通
- DCE 生效，死节点被移除
- 内存复用后总内存占用下降

---

### 阶段 4：Schedule 深化

#### 目标

让 Schedule 真正影响 codegen，而不是只停留在字段上。

#### 必做任务

- 抽象 loop 层的表示
- 让 `Schedule` 能表达：
  - tile
  - reorder
  - unroll
- 实现：
  - `InitSchedulePass`
  - `TilingPass`
  - `LoopReorderPass`
  - `UnrollPass`
- codegen 根据 schedule 选择不同循环结构

#### 建议实现方式

不要一开始就上完整 TensorIR。  
这里更适合做一个“够用的 loop metadata + codegen 分支逻辑”版本。

例如：

```cpp
struct LoopNest {
    std::vector<Loop> loops;
};
```

或者在 `Schedule` 中先存可直接消费的变换信息。

#### 阶段输出

- schedule 控制的代码生成
- matmul 的 tiled 版本
- reorder / unroll 的最小实现

#### 验收标准

- 开启和关闭 tiling 时，生成代码明显不同
- 调度前后数值结果一致
- 至少 1 个 case 中可以观察到性能改善

---

### 阶段 5：ARM / QEMU 部署

#### 目标

把整个项目从“本机能跑”推进到“目标环境能跑”。

#### 必做任务

- 准备 ARM 交叉编译工具链
- 编译 runtime 与生成代码到 ARM 目标
- 准备 QEMU ARM 运行脚本
- 打通：
  - 编译
  - 打包
  - 启动 QEMU
  - 执行模型
  - 比对输出

#### 阶段输出

- ARM 侧可执行程序
- QEMU 启动脚本
- Host / ARM 输出对比结果

#### 验收标准

- 同一个模型可在 Host 和 QEMU ARM 上都跑通
- 输出结果一致或在可接受误差内一致
- 整条部署链路可以重复执行

---

## 10. 推荐的 Pass Pipeline

建议后期把 pipeline 固定成下面这个顺序：

```cpp
pm.add(InferShapePass());
pm.add(DeadCodeEliminationPass());
pm.add(FusePass());
pm.add(OpCanonicalizePass());
pm.add(LivenessAnalysisPass());
pm.add(MemoryReusePass());
pm.add(InitSchedulePass());
pm.add(TilingPass());
pm.add(LoopReorderPass());
pm.add(UnrollPass());
```

可以记成：

```text
Graph -> Op -> Memory -> Schedule -> Codegen
```

这个顺序不要轻易打乱。

---

## 11. 推荐的开发顺序（按可提交的里程碑）

如果你是拿这个项目练手，建议按下面顺序推进，而不是一上来横着铺开。

1. 工程骨架和构建系统
2. IR 数据结构
3. Pass 抽象与 PassManager
4. JSON Frontend
5. Shape Infer
6. Naive Memory Plan
7. 最小 C Codegen
8. Host Runtime
9. Params 导出 / 加载
10. Conv2D + Fusion
11. ONNX Frontend
12. DCE / Liveness / Memory Reuse
13. Schedule Pass
14. ARM / QEMU

每完成一项，就做一次小闭环，不要堆很多未验证功能。

---

## 12. 测试与验收矩阵

建议你从第一阶段开始就建立测试，不要拖到最后。

| 类型 | 目标 | 最小样例 |
|------|------|----------|
| IR 单测 | Graph / Tensor / Op 构造正确 | 空图、单节点图 |
| Frontend 单测 | JSON / ONNX 导入正确 | 2-3 个小模型 |
| Pass 单测 | shape / fusion / DCE / memory 结果正确 | 人工构造 graph |
| Codegen 测试 | 输出代码结构稳定 | golden file 或关键片段检查 |
| Runtime 端到端 | 编译后模型数值正确 | MLP / CNN |
| 跨平台测试 | Host / ARM 输出一致 | 1 个固定模型 |

你不一定要一开始就把测试框架做得很复杂，但至少每阶段都要有：

- 1 个单测
- 1 个端到端样例

---

## 13. 这份计划里最值得你坚持的工程规则

### 13.1 先闭环，再扩算子

先把 `MatMul + Add + Relu` 做完整，再碰 `Conv`、`ONNX`、`Schedule`。

### 13.2 先正确，再优化

所有优化都应建立在“关闭优化时结果也正确”的前提上。

### 13.3 每一阶段都保留可运行样例

不要等到最后才发现主链路已经断掉。

### 13.4 所有 Pass 都要可 dump

这是你以后排查 shape、fusion、memory 问题时最省时间的办法。

### 13.5 保持输出产物稳定

建议始终输出：

- `graph.json`
- `params.bin`
- `deploy.c`
- 运行日志 / 报告

这样后面调试最方便。

---

## 14. 不建议你一开始就做的事情

下面这些事情都很诱人，但会显著拖慢主线进度：

- 一开始就设计很复杂的 IR 层级
- 一开始就支持很多 ONNX 算子
- 一开始就做复杂 schedule API
- 一开始就做性能 benchmark 系统
- 一开始就做 ARM 目标调优

更好的做法是：

- 第 1 轮：先把闭环做出来
- 第 2 轮：再补优化和工程质量
- 第 3 轮：最后补部署和跨平台

---

## 15. 项目完成标准

如果下面这些条件都达成，这个项目就已经是一个很好的练手作品了：

- 能导入 JSON 模型并成功编译运行
- 能导入 ONNX 子集模型并成功编译运行
- 有清晰的 IR / Pass / Schedule / Codegen / Runtime 分层
- 至少实现 1 个 fusion、1 个 memory reuse、1 组 schedule 变换
- 能生成 C 代码并在 Host 运行
- 能在 QEMU ARM 中运行同一模型
- 有最小测试体系和示例模型

做到这里，这个项目已经足够体现你的编译器工程能力。

---

## 16. 你可以直接照着执行的最短路线

如果你想更直接一点，就按这条路线走：

1. 建工程骨架
2. 实现 Graph / PassManager
3. 做 JSON Frontend
4. 做 `InferShapePass`
5. 做 `NaiveMemoryPlanner`
6. 做 `MatMul/Add/Relu` codegen
7. 做 Host runtime
8. 跑通 MLP
9. 加 `Conv2D`、`params.bin`、`FusePass`
10. 跑通 CNN
11. 加 ONNX Frontend
12. 加 DCE / Liveness / Memory Reuse
13. 加 Schedule Pass
14. 上 QEMU ARM

这条路线是最适合练手的：**每一步都能看到成果，每一步都能暴露真实工程问题。**
