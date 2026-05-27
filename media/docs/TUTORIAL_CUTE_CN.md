[README](../../README.md#documentation) > [中文入门指南](./GETTING_STARTED_CN.md) > **CuTe 详解教程**

# CuTe 详解教程

CuTe（C++ Templates for Linear Algebra）是 SYCL\*TLA 中用于描述和操作**层次化多维布局**的核心模板抽象库。它提供了 `Layout` 和 `Tensor` 两大核心抽象，让程序员专注于算法的逻辑描述，而由 CuTe 自动完成复杂的索引计算。

本教程将从基础类型开始，逐步深入到高级概念，帮助你全面掌握 CuTe。

---

## 目录

- [为什么需要 CuTe](#为什么需要-cute)
- [整体概念地图](#整体概念地图)
- [基础类型：Integer 与 IntTuple](#基础类型integer-与-inttuple)
- [Layout：核心抽象](#layout核心抽象)
  - [创建 Layout](#创建-layout)
  - [Layout 的坐标到索引映射](#layout-的坐标到索引映射)
  - [Layout 的操作函数](#layout-的操作函数)
  - [Layout 的层次化访问](#layout-的层次化访问)
- [Layout 代数](#layout-代数)
  - [Coalesce（合并）](#coalesce合并)
  - [Composition（组合）](#composition组合)
  - [Product 与 Divide（乘积与除法）](#product-与-divide乘积与除法)
- [Tensor：数据 + 布局](#tensor数据--布局)
  - [创建 Tensor](#创建-tensor)
  - [Tensor 的操作](#tensor-的操作)
  - [内存空间标签](#内存空间标签)
- [算法](#算法)
  - [copy — 数据拷贝](#copy--数据拷贝)
  - [gemm — 矩阵乘累加](#gemm--矩阵乘累加)
  - [其他算法](#其他算法)
- [MMA Atom 与 TiledMMA](#mma-atom-与-tiledmma)
  - [Operation 结构体](#operation-结构体)
  - [MMA_Traits](#mma_traits)
  - [MMA_Atom](#mma_atom)
  - [TiledMMA](#tiledmma)
  - [Intel Xe 的 DPAS 支持](#intel-xe-的-dpas-支持)
- [Intel Xe 2D Block Copy](#intel-xe-2d-block-copy)
  - [命名规则](#命名规则)
  - [数据分发模式](#数据分发模式)
  - [在 GEMM 中的使用](#在-gemm-中的使用)
- [实战代码解析](#实战代码解析)
  - [基础 GEMM（sgemm_1_sycl.cpp）](#基础-gemmsgemm_1_syclcpp)
  - [Intel Xe GEMM（xe_gemm.cpp）](#intel-xe-gemmxe_gemmcpp)
- [CUDA 与 SYCL 概念对照](#cuda-与-sycl-概念对照)
- [调试与可视化](#调试与可视化)
- [术语表](#术语表)
- [下一步学习](#下一步学习)

---

## 为什么需要 CuTe

在 GPU 编程中，高效的矩阵乘法需要精确管理：
- 数据如何在多维度上排列（行主序 vs 列主序、分块、VNNI 格式...）
- 线程/子组如何协作搬运和计算数据
- 不同层次的存储（全局内存 → 共享内存 → 寄存器）之间的数据流

传统做法是手写索引计算，容易出错且难以维护。CuTe 通过 **Layout** 抽象将所有这些关系用数学函数统一表达，自动完成索引推导。

**核心优势：**
1. **一套代码适配多种布局** — 行主序和列主序仅是 Stride 的区别
2. **编译期优化** — 静态整数使编译器在编译时完成索引计算
3. **组合性** — Layout 可以组合、分割、映射，表达任意复杂的数据分配
4. **硬件抽象** — 同样的接口封装 DPAS 指令和 2D Block Copy

---

## 整体概念地图

```
┌─────────────────────────────────────────────────────────┐
│                    CuTe 概念层次                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  IntTuple（递归整数元组）                                │
│    ├── Shape   — 维度大小   例: (4, 8)                  │
│    ├── Stride  — 步长       例: (1, 4)                  │
│    └── Coord   — 坐标       例: (2, 3)                  │
│         │                                               │
│         ▼                                               │
│  Layout = (Shape, Stride)                               │
│    │  坐标 → 线性索引 的映射函数                         │
│    │  例: Layout((4,8), (1,4))  →  coord(2,3) = 14     │
│    │                                                    │
│    ├── Coalesce   — 简化布局                            │
│    ├── Composition — 函数组合                           │
│    └── Product/Divide — 分块/分配                       │
│         │                                               │
│         ▼                                               │
│  Tensor = Engine + Layout                               │
│    │  数据数组 + 索引逻辑 = 多维数组                     │
│    │                                                    │
│    ▼                                                    │
│  Algorithms（算法）                                      │
│    ├── copy  — 数据搬运（可调度到硬件指令）              │
│    ├── gemm  — 矩阵乘法（可调度到 MMA 指令）            │
│    ├── fill / clear / axpby  — 初始化和线性组合          │
│    │                                                    │
│    └── Atom — 硬件指令的元信息封装                       │
│         ├── MMA_Atom — 矩阵乘累加指令（如 DPAS）        │
│         └── Copy_Atom — 数据拷贝指令（如 2D Block）      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 基础类型：Integer 与 IntTuple

### 整数类型

CuTe 中有两种整数：

| 类型 | 说明 | 示例 |
|---|---|---|
| **动态整数** | 运行时确定的值 | `int{4}`, `size_t{16}` |
| **静态整数** | 编译时确定的值（零开销） | `Int<4>{}`, `_4{}`, `cute::C<4>{}` |

静态整数的值编码在**类型**中，编译器可以在编译期完成所有计算，生成更高效的代码。

```cpp
using namespace cute;

// 动态整数 — 运行时确定
int m = 128;

// 静态整数 — 编译时确定（零开销）
auto bM = Int<128>{};   // 等同于 _128
auto bK = _32;          // 简写形式

// 混合使用 — 静态 + 动态运算得到动态结果
auto result = bM + m;   // 结果为动态 int
```

### IntTuple

**IntTuple** 是 CuTe 中的递归概念：一个 IntTuple 要么是一个整数，要么是一个 IntTuple 的元组。

```cpp
// 单个整数是 IntTuple
int a = 6;                    // 表示：6

// 整数的元组是 IntTuple
auto b = make_tuple(4, 3);    // 表示：(4, 3)

// 嵌套元组也是 IntTuple
auto c = make_tuple(3, make_tuple(6, 2), 8);  // 表示：(3, (6, 2), 8)
```

IntTuple 的核心操作：

| 操作 | 含义 | 示例 |
|---|---|---|
| `rank(x)` | 最外层元素数 | `rank((4,3)) = 2` |
| `get<I>(x)` | 获取第 I 个元素 | `get<0>((4,3)) = 4` |
| `depth(x)` | 嵌套深度 | `depth((3,(6,2))) = 2` |
| `size(x)` | 所有元素的乘积 | `size((4,3)) = 12` |

---

## Layout：核心抽象

**Layout** 是 CuTe 最核心的概念 — 它是一个从**坐标空间**到**索引空间**的映射函数，由 **(Shape, Stride)** 对组成。

```
Layout = (Shape, Stride)

映射规则：index = coord · stride
即：将坐标的每个分量与对应步长相乘，然后求和
```

### 创建 Layout

```cpp
using namespace cute;

// === 1D Layout ===
// 8 个元素，步长为 1（连续）
auto layout_8 = make_layout(Int<8>{});         // Shape=8, Stride=1

// === 2D Layout ===
// 2×4 矩阵，列主序（ColumnMajor / M-major）
auto col_major = make_layout(make_shape(2, 4));
// Shape=(2,4), Stride=(1,2)  ← 默认列主序
//
// 内存布局（线性）：
//       col0  col1  col2  col3
// row0:  [0]   [2]   [4]   [6]
// row1:  [1]   [3]   [5]   [7]

// 2×4 矩阵，行主序（RowMajor / K-major）
auto row_major = make_layout(make_shape(2, 4), LayoutRight{});
// Shape=(2,4), Stride=(4,1)
//
// 内存布局（线性）：
//       col0  col1  col2  col3
// row0:  [0]   [1]   [2]   [3]
// row1:  [4]   [5]   [6]   [7]

// 自定义步长
auto custom = make_layout(make_shape(2, 4),
                          make_stride(12, 1));
// Shape=(2,4), Stride=(12,1)
// coord(0,0)=0, coord(0,1)=1, coord(1,0)=12, coord(1,1)=13

// === 层次化 Layout ===
auto hier = make_layout(make_shape (2, make_shape (2, 2)),
                        make_stride(4, make_stride(2, 1)));
// Shape=(2,(2,2)), Stride=(4,(2,1))
```

### Layout 的坐标到索引映射

Layout 的本质就是一个函数 `f(coord) → index`：

```cpp
auto layout = make_layout(make_shape(3, 4), make_stride(1, 3));
// Shape=(3,4), Stride=(1,3)

// 使用多维坐标
layout(0, 0);  // = 0*1 + 0*3 = 0
layout(1, 0);  // = 1*1 + 0*3 = 1
layout(2, 0);  // = 2*1 + 0*3 = 2
layout(0, 1);  // = 0*1 + 1*3 = 3
layout(2, 3);  // = 2*1 + 3*3 = 11

// 也可以使用 1D 线性索引（CuTe 自动转换）
layout(0);     // = 0  (相当于 coord(0,0))
layout(5);     // = 5  (相当于 coord(2,1))
```

**图解：3×4 列主序**

```
Shape = (3, 4), Stride = (1, 3)

     col0  col1  col2  col3
row0: [0]   [3]   [6]   [9]
row1: [1]   [4]   [7]  [10]
row2: [2]   [5]   [8]  [11]

index = row * 1 + col * 3
```

### Layout 的操作函数

```cpp
auto L = make_layout(make_shape(3, 4), make_stride(1, 3));

rank(L);     // = 2        — 有 2 个模式（维度）
shape(L);    // = (3, 4)   — 形状
stride(L);   // = (1, 3)   — 步长
size(L);     // = 12       — 定义域大小（= 3 × 4）
cosize(L);   // = 12       — 值域大小（= L(11) + 1）
```

### Layout 的层次化访问

Layout 和 IntTuple 支持任意深度的嵌套访问：

```cpp
auto L = make_layout(make_shape (2, make_shape (3, 4)),
                     make_stride(1, make_stride(2, 6)));

// 多级 get
get<0>(L);      // 第 0 模式的子 Layout: Shape=2, Stride=1
get<1>(L);      // 第 1 模式的子 Layout: Shape=(3,4), Stride=(2,6)
get<1,0>(L);    // 第 1 模式的第 0 子模式: Shape=3, Stride=2
get<1,1>(L);    // 第 1 模式的第 1 子模式: Shape=4, Stride=6

// 层次化 size
size<0>(L);     // = 2        — 第 0 模式的大小
size<1>(L);     // = 12       — 第 1 模式的大小（= 3 × 4）
size<1,0>(L);   // = 3        — 第 1 模式的第 0 子模式大小
```

---

## Layout 代数

Layout 不仅仅是被动的数据结构，还支持一系列代数操作来创建更复杂的布局。

### Coalesce（合并）

`coalesce` 简化 Layout 而不改变其函数行为（即对所有坐标，映射结果不变）：

```cpp
// 原始 Layout：三个连续的模式
auto L = make_layout(make_shape(2, 3, 4), make_stride(1, 2, 6));
// L(x) 等价于一个 size=24 的连续 Layout

auto L_simple = coalesce(L);
// 结果：Shape=24, Stride=1  ← 合并为单一连续模式
```

合并规则：
- 删除 size=1 的模式
- 合并相邻的兼容模式（步长连续的）

### Composition（组合）

Layout 组合 `R = A ∘ B` 是**函数组合**：`R(c) = A(B(c))`。

```cpp
auto A = make_layout(make_shape(4, 8), make_stride(8, 1));  // 4×8 行主序
auto B = make_layout(make_shape(2, 4), make_stride(1, 2));  // 2×4 的一种排列

auto R = composition(A, B);
// R(c) = A(B(c))
// 用途：用 B 来重新索引 A，实现重排列和分块
```

组合是 CuTe 中最强大的操作之一，用于：
- **重新分块**数据布局
- **分配**线程到数据
- 构建复杂的内存访问模式

### Product 与 Divide（乘积与除法）

这些操作用于将数据布局**分配**给线程布局：

```cpp
// logical_divide: 用 TileShape 分割 Layout
auto data = make_layout(make_shape(128, 128));   // 128×128 数据
auto tile = make_shape(32, 32);                   // 32×32 分块

auto divided = logical_divide(data, tile);
// 结果：((32,4), (32,4))
// 内层 32 是分块内的坐标，外层 4 是分块的编号
```

**在 GEMM 中的应用**：
```
全局矩阵  →  logical_divide(tile_shape)  →  每个工作组的数据块
工作组数据  →  local_partition(thread_layout) → 每个线程的数据
```

---

## Tensor：数据 + 布局

**Tensor** 将数据指针（或数组）和 Layout 组合在一起，构成一个完整的多维数组抽象。

```
Tensor = Engine（数据源） + Layout（索引规则）
```

### 创建 Tensor

```cpp
using namespace cute;

// 从全局内存指针 + 形状创建
float* ptr = ...;  // 指向 GPU 全局内存
auto tensor_g = make_tensor(make_gmem_ptr(ptr),
                            make_shape(M, K),
                            make_stride(1, M));  // M×K 列主序

// 从共享内存创建
auto smem = compat::local_mem<float[128 * 32]>();
auto tensor_s = make_tensor(make_smem_ptr(smem),
                            make_layout(make_shape(128, 32)));

// 寄存器上的本地张量（数据存储在栈上）
auto tensor_r = make_tensor_like(tensor_g);  // 同形状，分配在寄存器
auto tensor_r2 = make_tensor<float>(make_shape(4, 8)); // 显式指定
```

### Tensor 的操作

```cpp
auto T = make_tensor(ptr, make_shape(3, 4), make_stride(1, 3));

// 访问元素
T(0, 0);       // 第 (0,0) 个元素
T(2, 3);       // 第 (2,3) 个元素
T(5);          // 线性索引第 5 个元素

// 查询信息
T.data();      // 底层数据指针
shape(T);      // (3, 4)
stride(T);     // (1, 3)
size(T);       // 12
rank(T);       // 2

// 切片
T(_, 0);       // 第 0 列（所有行）— 得到 rank-1 Tensor
T(1, _);       // 第 1 行（所有列）— 得到 rank-1 Tensor
T(_, _, ...);  // 使用 _ 保留该维度
```

### 内存空间标签

CuTe 使用 **tagged pointer** 来标记数据所在的内存空间，让算法自动选择最优实现：

```cpp
make_gmem_ptr(ptr);  // 全局内存（GPU 显存）
make_smem_ptr(ptr);  // 共享内存（工作组本地内存）
// 寄存器内存不需要标签（直接使用数组）
```

这种标签让 `copy()` 等算法自动调度到合适的硬件指令：
- 全局 → 共享：可使用异步拷贝指令
- 全局 → 寄存器：可使用 2D Block Load（Intel Xe）
- 寄存器 → 全局：可使用 2D Block Store（Intel Xe）

---

## 算法

### copy — 数据拷贝

```cpp
// 基本拷贝：自动选择实现
copy(src_tensor, dst_tensor);

// 使用特定的 Copy_Atom：调度到硬件指令
copy(copy_atom, src_tensor, dst_tensor);

// 条件拷贝：带掩码
copy_if(pred_tensor, src_tensor, dst_tensor);
```

`copy` 的强大之处在于**自动调度**：根据源和目标 Tensor 的内存空间标签和数据类型，自动选择最优的硬件拷贝指令。

### gemm — 矩阵乘累加

```cpp
// 基本 GEMM：C += A × B^T
// A: (M, K), B: (N, K), C: (M, N)
gemm(A_tensor, B_tensor, C_tensor);

// 使用 MMA_Atom 调度到硬件 MMA 指令
gemm(mma_atom, A_tensor, B_tensor, C_tensor);
```

CuTe 的 `gemm` 有多个调度变体：
- `(V) × (V) → (V)`：向量级
- `(M,K) × (N,K) → (M,N)`：矩阵级
- 批量版本

### 其他算法

```cpp
fill(tensor, value);    // 用 value 填充所有元素
clear(tensor);          // 清零
axpby(alpha, x, beta, y);  // y = alpha * x + beta * y
```

---

## MMA Atom 与 TiledMMA

CuTe 通过多层抽象封装硬件的矩阵乘累加（MMA）指令。

### Operation 结构体

最底层 — 直接封装硬件指令：

```cpp
// Intel Xe 的 DPAS 指令封装
struct XE_DPAS_TT<8, float, bfloat16_t> {
    using DRegisters = ...;  // D 矩阵的寄存器
    using ARegisters = ...;  // A 矩阵的寄存器
    using BRegisters = ...;  // B 矩阵的寄存器
    using CRegisters = ...;  // C 矩阵的寄存器

    static void fma(...);    // 执行 DPAS 指令
};
```

### MMA_Traits

为 Operation 提供**元信息**：逻辑形状、线程布局、值布局。

```cpp
// MMA_Traits 为每个 Operation 特化，提供：
// - Shape_MNK: 指令的逻辑 M×N×K 大小
// - ThrID:     线程 ID 到操作数位置的映射
// - ALayout, BLayout, CLayout: 值在线程间的分布
```

### MMA_Atom

组合 Operation + Traits，提供高级接口：

```cpp
using Atom = MMA_Atom<XE_DPAS_TT<8, float, bfloat16_t>>;
// Atom 隐藏了线程/数据布局的复杂性
// 对外提供：partition_fragment_A, partition_fragment_B, partition_fragment_C
```

### TiledMMA

将多个 Atom 组合成更大的分块操作：

```cpp
// 在 SYCL*TLA 中，TiledMMAHelper 简化了 TiledMMA 的创建
using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, bfloat16_t>>,  // 基本指令
    Layout<Shape<_256, _256, _32>>,               // 工作组分块大小
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>> // 子组排列
>::TiledMMA;
```

TiledMMA 提供的关键方法：

```cpp
auto thr_mma = mma.get_slice(thread_id);  // 获取线程级操作

// 分配寄存器碎片
auto frag_A = thr_mma.partition_fragment_A(A_tensor);
auto frag_B = thr_mma.partition_fragment_B(B_tensor);
auto frag_C = partition_fragment_C(mma, tile_shape);

// 执行 MMA
gemm(mma, frag_A, frag_B, frag_C);
```

### Intel Xe 的 DPAS 支持

Intel Xe GPU 使用 DPAS 指令进行矩阵运算。在 CuTe 中的命名格式：

```cpp
XE_DPAS_<LayoutA><LayoutB><M, AccType, InputType>
```

| 字段 | 含义 |
|---|---|
| `TT` | A 和 B 都是转置的（row-major） |
| `8` | M 维度为 8 |
| `float` | 累加器类型 |
| `bfloat16_t` | 输入类型 |

DPAS 指令的固有维度：
- M = 8（可选 1, 2, 4, 8）
- N = 16（固定）
- K = 取决于数据类型（bfloat16 → 16, int8 → 32, int4 → 64）

---

## Intel Xe 2D Block Copy

Intel Xe GPU 提供了硬件加速的 **2D 块拷贝指令**，可以高效地在全局内存和寄存器之间搬运 2D 数据块。

### 命名规则

```
XE_2D_[Packed_]<DataType>x<Rows>x<Cols>_<LD|ST>_<N|T|V>
```

| 部分 | 含义 |
|---|---|
| `XE_2D` | Intel Xe 2D 拷贝 |
| `Packed_` | 打包模式（仅用于 U8/U4 数据） |
| `DataType` | 寄存器中的数据宽度：U4, U8, U16, U32 |
| `Rows×Cols` | 2D 块的维度（元素数） |
| `LD / ST` | 加载 / 存储 |
| `N / T / V` | 行主序 / 列主序 / VNNI 格式 |

**示例：**

```cpp
XE_2D_U16x32x16_LD_N   // 加载 32×16 的 U16 数据，行主序
XE_2D_U16x32x32_LD_V   // 加载 32×32 的 U16 数据，VNNI 格式
XE_2D_U32x8x16_ST_N    // 存储 8×16 的 U32 数据，行主序
```

### 数据分发模式

所有 2D 拷贝以**子组（16 个工作项）** 为单位执行。

**非打包模式（Unpacked）：**

```
XE_2D_U16x32x16_LD_N → 32 行 × 16 列
  Work-item 0:  Column 0  (32 elements)
  Work-item 1:  Column 1  (32 elements)
  ...
  Work-item 15: Column 15 (32 elements)
```

如果列数 > 16，每个工作项获得多列：

```
XE_2D_U16x32x32_LD_N → 32 行 × 32 列
  Work-item 0:  Column 0 + Column 16  (各 32 elements)
  Work-item 1:  Column 1 + Column 17
  ...
  Work-item 15: Column 15 + Column 31
```

**VNNI 模式（V）：** VNNI 将多行的同一列元素打包到 32 位值中，不改变实际数据，只改变存储格式，使 DPAS 指令可以直接使用。

**打包模式（Packed）：** 用于 U8/U4 MMA 操作，每个工作项获得相邻列：

```
XE_2D_Packed_U8x8x32_LD_N → 8 行 × 32 列
  Work-item 0:  Column 0, 1    (各 8 elements)
  Work-item 1:  Column 2, 3
  ...
  Work-item 15: Column 30, 31
```

### 在 GEMM 中的使用

| 矩阵 | 行主序 | 列主序 |
|---|---|---|
| **A** | `LD_N` | `LD_T` |
| **B** | `LD_V`（或 `LD_N` for U32） | `LD_T` |
| **D**（输出） | `ST_N` | — |

---

## 实战代码解析

### 基础 GEMM（sgemm_1_sycl.cpp）

这个示例展示了不使用硬件 MMA 指令的基础 CuTe GEMM，帮助理解核心流程。

#### 1. 创建全局 Tensor 并分块

```cpp
// 创建全局矩阵的 Tensor
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

// 按工作组分块
auto cta_coord = make_coord(work_group_id_x, work_group_id_y, _);
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{}); // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1,  X>{}); // (BLK_M,BLK_N)
```

**解释：**
- `local_tile` 根据工作组坐标提取该工作组负责的数据块
- `Step<_1, X, _1>` 中 `_1` 表示参与选择的维度，`X` 表示跳过的维度
- 结果 `gA` 的形状是 `(BLK_M, BLK_K, k)` — 最后一维 `k` 是沿 K 维度的分块数

#### 2. 创建共享内存 Tensor

```cpp
auto smemA = compat::local_mem<float[BLK_M * BLK_K]>();
Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);  // (BLK_M, BLK_K)
```

#### 3. 按线程分配数据

```cpp
// 线程分配拷贝任务
Tensor tAgA = local_partition(gA, tA, thread_id);  // (THR_M, THR_K, k)
Tensor tAsA = local_partition(sA, tA, thread_id);  // (THR_M, THR_K)

// 线程分配计算任务（不同的分区方式）
Tensor tCsA = local_partition(sA, tC, thread_id, Step<_1, X>{});  // (THR_M, BLK_K)
Tensor tCsB = local_partition(sB, tC, thread_id, Step< X, _1>{}); // (THR_N, BLK_K)
Tensor tCgC = local_partition(gC, tC, thread_id, Step<_1, _1>{}); // (THR_M, THR_N)
```

**关键洞察：** 拷贝和计算可以使用**不同的线程布局**！
- `tA`/`tB` 定义了拷贝时线程如何分配到数据元素（优化内存访问合并）
- `tC` 定义了计算时线程如何分配到输出元素（优化计算并行度）

#### 4. 主循环

```cpp
Tensor tCrC = make_tensor_like(tCgC);  // 寄存器上的累加器
clear(tCrC);

for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    // 全局内存 → 共享内存
    copy(tAgA(_, _, k_tile), tAsA);
    copy(tBgB(_, _, k_tile), tBsB);

    cp_async_fence();
    cp_async_wait<0>();
    compat::wg_barrier();  // 等待所有线程完成写入

    // 共享内存上执行 GEMM
    gemm(tCsA, tCsB, tCrC);  // C += A × B^T

    compat::wg_barrier();  // 等待所有线程完成读取
}

// 写回结果
axpby(alpha, tCrC, beta, tCgC);  // C = alpha * acc + beta * C
```

### Intel Xe GEMM（xe_gemm.cpp）

这个示例使用 Intel Xe 的硬件指令（DPAS + 2D Block Copy），展示了高性能 GEMM 的写法。

#### 1. 设置 TiledMMA 和 Block Copy

```cpp
// 自动选择 MMA 操作（基于数据类型）
auto mma = choose_tiled_mma(A, B, C);

// 创建 Block 2D Copy 操作
auto copy_a = make_block_2d_copy_A(mma, A);
auto copy_b = make_block_2d_copy_B(mma, B);
auto copy_c = make_block_2d_copy_D(mma, C);
```

#### 2. 分配寄存器碎片

```cpp
auto thr_mma = mma.get_slice(local_id);

// MMA 碎片（用于计算）
auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
auto tCrC = partition_fragment_C(mma, select<0,1>(wg_tile));

// Copy 碎片（用于数据搬运）
auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
```

#### 3. 预取 + 主循环

```cpp
// 预热预取
for (int pf = 0; pf < prefetch_dist; pf++) {
    prefetch(prefetch_a, pAgA(_, _, _, pf));
    prefetch(prefetch_b, pBgB(_, _, _, pf));
}

// 主循环
for (int k = 0; k < k_tile_count; k++, k_pf++) {
    barrier_arrive(barrier_scope);

    // 2D Block Load: 全局内存 → 寄存器
    copy(copy_a, tAgA(_, _, _, k), tArA);
    copy(copy_b, tBgB(_, _, _, k), tBrB);

    // 继续预取
    prefetch(prefetch_a, pAgA(_, _, _, k_pf));
    prefetch(prefetch_b, pBgB(_, _, _, k_pf));

    // 重排数据以适配 MMA 格式
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    // DPAS: C += A × B
    gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
}

// 2D Block Store: 寄存器 → 全局内存
copy(copy_c, tCrC, tCgC);
```

**基础 GEMM vs Xe GEMM 的主要区别：**

| 方面 | 基础 GEMM | Xe GEMM |
|---|---|---|
| 数据搬运 | 逐元素拷贝到共享内存 | 2D Block Load 直接到寄存器 |
| 计算 | 标量 FMA 循环 | DPAS 硬件 MMA 指令 |
| 预取 | 无 | 硬件预取到 L1 缓存 |
| 数据格式 | 原始格式 | 需要 reorder 适配 VNNI |
| 性能 | 教学用途 | 接近峰值性能 |

---

## CUDA 与 SYCL 概念对照

如果你有 CUDA 背景，以下对照表会帮助你快速理解 SYCL/Intel GPU 中的对应概念：

| CUDA | SYCL / Intel Xe | CuTe 中的用法 |
|---|---|---|
| Thread | Work-item | `compat::local_id::x()` |
| Warp (32 threads) | Sub-group (16 work-items) | `mma.get_slice(local_id)` |
| Thread Block | Work-group | `compat::work_group_id::x()` |
| Grid | ND-Range | `sycl::nd_range<2>(global, local)` |
| Shared Memory | Local Memory (SLM) | `compat::local_mem<T[]>()` |
| `__syncthreads()` | `compat::wg_barrier()` | 工作组同步 |
| Tensor Core (MMA) | DPAS 指令 | `XE_DPAS_TT<8, float, bf16>` |
| `cp.async` | 2D Block Load/Store | `XE_2D_U16x32x16_LD_N` |
| `__shfl_sync` | Sub-group shuffle | 子组内通信 |

---

## 调试与可视化

### 打印 CuTe 对象

CuTe 提供了丰富的打印函数：

```cpp
// 打印 Layout
print(layout);          // 基本信息输出

// 打印 rank-2 Layout 的坐标-索引表
print_layout(layout);   // 可视化 2D 映射关系

// 打印 Tensor 的值
print_tensor(tensor);   // 显示多维数组的实际值

// 生成 LaTeX 可视化
print_latex(layout);    // 输出 LaTeX 代码用于生成彩色表格
```

### 仅在线程 0 打印

```cpp
if (thread0()) {
    print("mA : "); print(mA); print("\n");
    print("gA : "); print(gA); print("\n");
}
```

> **注意：** 设备端打印开销很大。即使打印代码不被执行（在未走到的分支中），也会生成更慢的代码。调试完毕后务必移除。

---

## 术语表

| 英文 | 中文 | 说明 |
|---|---|---|
| IntTuple | 整数元组 | 递归定义：整数或整数元组的元组 |
| Shape | 形状 | 描述各维度大小的 IntTuple |
| Stride | 步长 | 描述各维度内存间距的 IntTuple |
| Coord | 坐标 | 逻辑空间中的位置 |
| Layout | 布局 | (Shape, Stride) 对，坐标→索引的映射 |
| Rank | 秩 | IntTuple 的最外层元素数（维度数） |
| Depth | 深度 | IntTuple 的嵌套层数 |
| Size | 大小 | IntTuple 所有元素的乘积 |
| Cosize | 值域大小 | Layout 函数的值域大小 |
| Coalesce | 合并 | 简化 Layout 而不改变函数行为 |
| Composition | 组合 | 函数组合 R(c) = A(B(c)) |
| Tensor | 张量 | Engine（数据）+ Layout（索引）|
| Engine | 引擎 | 数据的存储后端（指针或数组） |
| Tagged Pointer | 标签指针 | 携带内存空间信息的指针 |
| Atom | 原子操作 | 硬件指令的最小抽象单元 |
| MMA_Atom | MMA 原子 | 矩阵乘累加指令抽象 |
| Copy_Atom | Copy 原子 | 数据拷贝指令抽象 |
| TiledMMA | 分块 MMA | 多个 Atom 组合的大分块操作 |
| TiledCopy | 分块 Copy | 多个 Copy Atom 组合的大分块拷贝 |
| DPAS | 点积累加脉动阵列 | Intel GPU 的矩阵运算指令 |
| VNNI | 垂直原生自然指令 | Intel 特有的数据打包格式 |
| Block 2D | 2D 块消息 | Intel GPU 的 2D 数据搬运指令 |
| Sub-group | 子组 | 16 个工作项的硬件执行单元 |
| Work-group | 工作组 | 协作执行的工作项集合 |
| SLM | 共享本地内存 | 工作组内共享的片上内存 |
| Prefetch | 预取 | 提前将数据加载到缓存 |

---

## 下一步学习

掌握 CuTe 基础后，建议按以下路径深入：

1. **[英文 Layout 文档](./cpp/cute/01_layout.md)** — Layout 的完整技术细节
2. **[英文 Layout 代数](./cpp/cute/02_layout_algebra.md)** — Coalesce、Composition、Product 的数学基础
3. **[英文 Tensor 文档](./cpp/cute/03_tensor.md)** — Tensor 引擎、所有权模型的完整说明
4. **[英文算法文档](./cpp/cute/04_algorithms.md)** — copy、gemm 等算法的完整调度逻辑
5. **[MMA Atom 文档](./cpp/cute/0t_mma_atom.md)** — Operation、Traits、Atom 的完整解释
6. **[Xe 2D Copy 文档](./cpp/cute/xe_2d_copy.md)** — Intel Xe 2D 拷贝操作的详细规格
7. **[GEMM 代码详解教程](./TUTORIAL_GEMM_CN.md)** — 从 CuTe 到完整 GEMM 内核的组装过程
8. **[GEMM 教程（英文）](./cpp/cute/0x_gemm_tutorial.md)** — 用 CuTe 从头构建 GEMM 的完整教程
