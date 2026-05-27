[README](../../README.md#documentation) > [中文入门指南](./GETTING_STARTED_CN.md) > **GEMM 代码详解教程**

# SYCL\*TLA GEMM 代码详解教程

本教程以 [`examples/00_bmg_gemm/00_bmg_gemm.cpp`](../../examples/00_bmg_gemm/00_bmg_gemm.cpp) 为例，
逐步讲解如何使用 SYCL\*TLA 构建一个完整的 GEMM（通用矩阵乘法）内核。
读完本教程后，您将理解 SYCL\*TLA 中 GEMM 的核心模板组件及其组装方式。

---

## 目录

- [GEMM 运算回顾](#gemm-运算回顾)
- [整体架构概览](#整体架构概览)
- [第一步：数据类型与布局定义](#第一步数据类型与布局定义)
- [第二步：分块形状（TileShape）](#第二步分块形状tileshape)
- [第三步：Copy Atom — 数据搬运](#第三步copy-atom--数据搬运)
- [第四步：TiledMMA — 矩阵乘累加](#第四步tiledmma--矩阵乘累加)
- [第五步：流水线与调度策略](#第五步流水线与调度策略)
- [第六步：尾声（Epilogue）](#第六步尾声epilogue)
- [第七步：组装完整 GEMM 内核](#第七步组装完整-gemm-内核)
- [第八步：内存管理与初始化](#第八步内存管理与初始化)
- [第九步：运行与验证](#第九步运行与验证)
- [简化方式：CollectiveBuilder](#简化方式collectivebuilder)
- [关键术语对照表](#关键术语对照表)

---

## GEMM 运算回顾

GEMM 计算的数学公式为：

```
D = α × (A × B) + β × C
```

其中：
- **A**：M × K 矩阵（输入）
- **B**：K × N 矩阵（输入）
- **C**：M × N 矩阵（输入，可选偏置）
- **D**：M × N 矩阵（输出）
- **α, β**：标量系数

在 SYCL\*TLA 中，矩阵乘法 `A × B` 由 **Mainloop**（主循环）完成，
而 `α × ... + β × C` 的缩放与加法由 **Epilogue**（尾声）完成。

---

## 整体架构概览

SYCL\*TLA 中一个完整的 GEMM 由以下层次组成：

```
GemmUniversalAdapter          ← 设备端启动接口
  └── GemmKernel              ← 内核定义（组合 Mainloop + Epilogue）
        ├── CollectiveMainloop ← 主循环：沿 K 维度迭代执行分块矩阵乘法
        │     ├── TiledMMA     ← 定义矩阵乘累加操作的分块方式
        │     └── TiledCopy    ← 定义数据从全局内存到寄存器的搬运方式
        └── CollectiveEpilogue ← 尾声：加载 C、执行后处理、写回 D
              └── FusionCallbacks ← 定义具体的后处理操作（如线性组合）
```

接下来，我们按顺序讲解每个组件。

---

## 第一步：数据类型与布局定义

```cpp
using ElementAccumulator = float;      // 累加器数据类型
using ElementComputeEpilogue = float;  // 尾声计算的数据类型
using ElementInputA = bfloat16_t;      // 输入矩阵 A 的元素类型
using ElementInputB = bfloat16_t;      // 输入矩阵 B 的元素类型
using ElementOutput = float;           // 输出矩阵 D 的元素类型

using LayoutA = cutlass::layout::RowMajor;  // A 矩阵行主序
using LayoutB = cutlass::layout::RowMajor;  // B 矩阵行主序
using LayoutC = cutlass::layout::RowMajor;  // C 矩阵行主序
using LayoutD = cutlass::layout::RowMajor;  // D 矩阵行主序
```

**要点：**
- **混合精度**：输入用 `bfloat16_t`（16位浮点），累加用 `float`（32位浮点）。这是深度学习中常见的做法 — 用低精度输入节省带宽，用高精度累加保证数值精度。
- **布局（Layout）**：`RowMajor` 表示矩阵按行存储在内存中。SYCL\*TLA 也支持 `ColumnMajor`（列主序）。

---

## 第二步：分块形状（TileShape）

```cpp
using TileShape = Shape<_256, _256, _32>;
```

`TileShape` 定义了**每个工作组（work-group）处理的数据块大小**：
- **M = 256**：每个工作组处理输出矩阵的 256 行
- **N = 256**：每个工作组处理输出矩阵的 256 列
- **K = 32**：每次迭代处理 K 维度的 32 个元素

整个 GEMM 的 K 维度通过主循环（Mainloop）迭代完成，每次处理 K=32 的一个块。

```
       ← N=256 →
   ┌────────────────┐
   │  一个工作组     │ ↑
   │  处理的输出块   │ M=256
   │                │ ↓
   └────────────────┘
         沿 K 维度迭代 →→→ （每次 K=32）
```

---

## 第三步：Copy Atom — 数据搬运

```cpp
using GmemTiledCopyA = void;  // 自动选择
using GmemTiledCopyB = void;  // 自动选择
```

**Copy Atom** 描述了如何将数据从全局内存搬运到寄存器。Intel GPU 提供了 **2D Block Messages**，可以高效搬运 2D 数据块。

设置为 `void` 时，`MainloopXeL1Staged` 会自动选择合适的拷贝操作。也可以手动指定：

```cpp
using GmemTiledCopyA = XE_LOAD_2D<16, 32, 32>;       // 普通 2D 块加载
using GmemTiledCopyB = XE_LOAD_2D_VNNI<16, 32, 32>;  // VNNI 格式加载（适配 DPAS）
```

**VNNI（Vertical Native Natural Instruction）格式** 是 Intel GPU 的特殊数据排列方式，使得 DPAS 指令可以直接使用加载的数据而无需额外转换。

---

## 第四步：TiledMMA — 矩阵乘累加

```cpp
using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
    Layout<TileShape>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>
>::TiledMMA;
```

这是核心的计算部分，由三层抽象组成：

### MMA Atom（最小计算单元）

```cpp
MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>
```

`XE_DPAS_TT<8, float, bfloat16_t>` 表示一条 DPAS 指令：
- **M = 8**：处理 8 行
- **N = 16**：处理 16 列（Intel GPU 固定值）
- **K = 16**：取决于数据类型（bfloat16 时为 16）
- **float**：累加器类型
- **bfloat16_t**：输入类型

### 子组（Sub-group）布局

```cpp
Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>
```

定义了子组（sub-group，类似于 NVIDIA 的 warp）在工作组中的排列方式：
- **8 × 4 × 1 = 32** 个子组
- `Stride<_4, _1, _0>` 表示行主序排列（stride-4 沿 M，stride-1 沿 N）

### TiledMMA 的计算分配

每个子组的工作量 = TileShape ÷ 子组数：
- M 维度：256 ÷ 8 = **32 行/子组**
- N 维度：256 ÷ 4 = **64 列/子组**
- K 维度：32 ÷ 1 = **32/子组**

每个子组在其负责的 32×64×32 块上执行多次 8×16×16 DPAS 操作。

---

## 第五步：流水线与调度策略

```cpp
constexpr int PipelineStages = 2;
using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;
```

- **PipelineStages = 2**：表示在处理当前 K 块时，同时预取接下来 2 个 K 块的数据。这是 Intel GPU 上性能的关键 — 通过预取隐藏内存延迟。
- **MainloopXeL1Staged**：Intel Xe 架构的分阶段主循环策略，利用 L1 缓存管理预取。
- **IntelXeGeneric**：通用的 Intel Xe 尾声调度策略。

```
时间线：
  ┌──────┬──────┬──────┬──────┐
  │ 计算 K₀│ 计算 K₁│ 计算 K₂│ ... │  ← DPAS 计算
  └──────┴──────┴──────┴──────┘
  ┌──────┬──────┬──────┬──────┐
  │预取 K₁,K₂│预取 K₂,K₃│预取 K₃,K₄│ ... │  ← 内存预取（2 stages）
  └──────┴──────┴──────┴──────┘
```

---

## 第六步：尾声（Epilogue）

尾声负责 GEMM 公式中的后处理部分：`D = α × (A×B) + β × C`。

```cpp
// 1. 定义尾声操作（线性组合）
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput,           // 输出类型
    ElementComputeEpilogue,  // 计算类型
    ElementAccumulator,      // 累加器类型
    ElementAccumulator,      // 源类型
    cutlass::FloatRoundStyle::round_to_nearest>;  // 舍入方式

// 2. 将操作绑定到实现
using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy,        // 调度策略
    EpilogueOp,                    // 操作类型
    TileShape,                     // 分块大小
    decltype(tile_shape(TiledMma()))>;  // MMA 分块大小

// 3. 组装完整尾声
using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    void,                 // 尾声分块（void = 自动）
    ElementAccumulator,
    cutlass::gemm::TagToStrideC_t<LayoutC>,  // C 矩阵步长
    ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD>,  // D 矩阵步长
    FusionCallbacks,
    void,                 // C 的 Copy Atom（void = 自动）
    void>;                // D 的 Copy Atom（void = 自动）
```

**`LinearCombination`** 执行的操作是：`D = α × Accumulator + β × C`

如果需要更复杂的后处理（如 ReLU 激活函数），可以替换 `EpilogueOp`。示例 `05_bmg_gemm_with_epilogues` 展示了更多尾声变体。

---

## 第七步：组装完整 GEMM 内核

```cpp
// 主循环：负责 A×B 矩阵乘法
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy,
    TileShape,
    ElementInputA,
    cutlass::gemm::TagToStrideA_t<LayoutA>,   // A 步长（行主序或列主序）
    ElementInputB,
    cutlass::gemm::TagToStrideB_t<LayoutB>,   // B 步长
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,  // A 的拷贝配置
    GmemTiledCopyB, void, void, cute::identity   // B 的拷贝配置
>;

// GEMM 内核 = 主循环 + 尾声
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,   // 运行时确定的问题大小 (M, N, K, L)
    CollectiveMainloop,
    CollectiveEpilogue
>;

// 设备端适配器：处理内核启动、工作空间管理等
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

组装顺序总结：

```
数据类型 + 布局 + TileShape
         ↓
    TiledMMA + Copy Atoms
         ↓
  CollectiveMainloop（主循环）
         ↓
  CollectiveEpilogue（尾声）
         ↓
     GemmKernel（内核）
         ↓
  GemmUniversalAdapter（设备接口）
```

---

## 第八步：内存管理与初始化

```cpp
// 分配设备内存
cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<ElementOutput> block_D;

// 计算步长（stride）
stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));

// 分配内存并随机初始化
block_A.reset(M * K * L);
initialize_block(block_A, seed + 2023);
```

**步长（Stride）** 描述了在内存中从一个元素移动到下一行/列/批次需要跳过多少元素。SYCL\*TLA 使用 CuTe 的步长描述符来灵活处理各种内存布局。

---

## 第九步：运行与验证

```cpp
// 构造参数
typename Gemm::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,  // GEMM 模式
    problem_size,                              // {M, N, K, L}
    {block_A.get(), stride_A, block_B.get(), stride_B},        // 主循环参数
    {{options.alpha, options.beta}, block_C.get(), stride_C,    // 尾声参数
     block_D.get(), stride_D},
    hw_info                                    // 硬件信息
};

// 初始化并运行
Gemm gemm_op;
CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
CUTLASS_CHECK(gemm_op.run());

// 等待完成
compat::wait();

// 与参考实现对比验证
bool passed = cutlass::reference::device::BlockCompareEqual(
    block_ref_D.get(), block_D.get(), block_D.size());
```

运行流程：
1. **构造参数**：包含问题大小、输入/输出指针、步长、标量系数、硬件信息
2. **初始化**：分配工作空间、配置内核
3. **运行**：提交到 SYCL 队列执行
4. **等待**：`compat::wait()` 等待内核完成
5. **验证**：与参考实现结果逐元素对比

---

## 简化方式：CollectiveBuilder

上面的手动方式需要显式指定 Copy Atom、TiledMMA、DispatchPolicy 等。SYCL\*TLA 提供了 **CollectiveBuilder**，可以根据数据类型和架构自动选择这些组件。

参见 [`examples/01_bmg_gemm_with_collective_builder`](../../examples/01_bmg_gemm_with_collective_builder/)：

```cpp
// 主循环 — 只需指定数据类型、布局、对齐和分块大小
using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::IntelXe,             // 目标架构
    cutlass::arch::OpClassTensorOp,     // 运算类
    ElementInputA, LayoutA, AlignmentA, // A 矩阵配置
    ElementInputB, LayoutB, AlignmentB, // B 矩阵配置
    ElementAccumulator,                 // 累加器类型
    TileShape,                          // 分块大小
    Shape<_1, _1, _1>,                  // ClusterShape（Intel Xe 始终为 1×1×1）
    cutlass::gemm::collective::StageCountAuto,    // 自动选择流水线级数
    cutlass::gemm::collective::KernelScheduleAuto // 自动选择调度策略
>::CollectiveOp;

// 尾声 — 同样自动配置
using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    TileShape, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementComputeEpilogue, ElementAccumulator,
    ElementAccumulator, LayoutC, AlignmentC,
    ElementOutput,      LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EpilogueOp  // 可以指定自定义的尾声操作，如 ReLU
>::CollectiveOp;
```

### 手动方式 vs CollectiveBuilder 对比

| 方面 | 手动方式（Example 00） | CollectiveBuilder（Example 01） |
|---|---|---|
| Copy Atom | 需要手动指定或设为 void | 自动选择 |
| TiledMMA | 需要手动配置 MMA Atom + 子组布局 | 自动选择 |
| DispatchPolicy | 需要手动选择 | 自动选择 |
| 灵活性 | 最高 — 完全控制每个细节 | 较高 — 仍可自定义尾声操作 |
| 适合场景 | 需要精细优化的高级用户 | 快速原型开发和标准用例 |

**建议**：初学者从 CollectiveBuilder 开始，在需要精细调优时再使用手动方式。

---

## 关键术语对照表

| 英文术语 | 中文翻译 | 说明 |
|---|---|---|
| GEMM | 通用矩阵乘法 | General Matrix-Matrix Multiplication |
| Tile / TileShape | 分块 / 分块形状 | 将大矩阵分成小块处理 |
| Work-group | 工作组 | 一组协作执行的线程（类似 CUDA thread block） |
| Sub-group | 子组 | 工作组内的子集，硬件同步执行（类似 CUDA warp） |
| MMA Atom | 矩阵乘累加原子操作 | 硬件支持的最小矩阵乘法单元 |
| DPAS | 点积累加脉动阵列 | Intel GPU 的核心矩阵运算指令 |
| Copy Atom | 拷贝原子操作 | 硬件支持的最小数据搬运单元 |
| Block 2D | 2D 块消息 | Intel GPU 的 2D 数据搬运指令 |
| VNNI | 垂直原生自然指令 | Intel 特有的数据排列格式 |
| Mainloop | 主循环 | 沿 K 维度迭代执行矩阵乘法 |
| Epilogue | 尾声 | GEMM 后处理（缩放、加偏置、激活函数等） |
| Pipeline Stages | 流水线级数 | 预取数据的提前量 |
| Stride | 步长 | 内存中相邻元素间的距离 |
| Layout | 布局 | 数据在内存中的排列方式 |
| Accumulator | 累加器 | 存储中间计算结果的寄存器 |
| Fusion | 融合 | 将多个操作合并到一个内核中执行 |
| EVT | 尾声访问者树 | Epilogue Visitor Tree — 灵活的尾声组合框架 |
| CollectiveBuilder | 集合构建器 | 自动配置 Mainloop/Epilogue 的辅助类 |

---

## 下一步学习

掌握基础 GEMM 后，可以继续探索：

1. **[混合精度 GEMM](../../examples/02_bmg_gemm_mixed_dtype/)** — 不同输入/输出精度的组合
2. **[Stream-K 调度](../../examples/03_bmg_gemm_streamk/)** — 更好的负载均衡策略
3. **[尾声融合](../../examples/05_bmg_gemm_with_epilogues/)** — EVT 实现复杂后处理
4. **[Flash Attention](../../examples/06_bmg_flash_attention/)** — 大模型注意力机制的高效实现
5. **[Xe 架构文档](./cpp/xe_rearchitecture.md)** — 深入理解 Intel GPU 硬件特性
6. **[CuTe 教程](./cpp/cute/00_quickstart.md)** — 学习 Layout 和 Tensor 抽象
