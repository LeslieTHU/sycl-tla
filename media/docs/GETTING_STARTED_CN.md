[README](../../README.md#documentation) > **SYCL\*TLA 中文入门指南**

# SYCL\*TLA 中文入门指南

本文档旨在帮助初学者快速了解 SYCL\*TLA 库的基本概念、项目结构、环境配置及使用方法。

---

## 目录

- [项目简介](#项目简介)
- [核心概念](#核心概念)
- [支持的硬件](#支持的硬件)
- [环境要求](#环境要求)
- [构建项目](#构建项目)
- [运行示例](#运行示例)
- [项目结构概览](#项目结构概览)
- [Intel GPU 示例一览](#intel-gpu-示例一览)
- [Python 接口](#python-接口)
- [常见问题排查](#常见问题排查)
- [进阶学习资源](#进阶学习资源)

---

## 项目简介

**SYCL\*TLA**（SYCL Templates for Linear Algebra）是一个高性能的 C++ 模板库，专注于在 Intel GPU 上通过 SYCL 实现高效的线性代数运算（如矩阵乘法 GEMM 及融合尾声操作）。

该项目基于 NVIDIA CUTLASS 开源仓库，将 CUTLASS 和 CuTe 的 API 扩展到 Intel GPU 平台。

### 主要特性

- **纯头文件库**：无需预编译，直接 `#include` 即可使用
- **混合精度支持**：FP64、FP32、FP16、BF16、FP8（E5M2/E4M3）、INT4/INT8
- **量化支持**：张量级、通道级、分组级量化
- **高性能 GEMM**：针对 Intel GPU 的 DPAS（Dot Product Accumulate Systolic）指令优化
- **尾声融合（Epilogue Fusion）**：通过 Epilogue Visitor Tree (EVT) 支持灵活的后处理操作
- **Flash Attention**：针对 Intel GPU 优化的 Flash Attention V2 实现

---

## 核心概念

### GEMM（通用矩阵乘法）

GEMM 是该库的核心运算：**D = α × A × B + β × C**。库中通过分层分块（hierarchical tiling）将大矩阵分解为适合 GPU 执行单元处理的小块，以实现高效并行计算。

### CuTe

CuTe 是 SYCL\*TLA 中用于描述和操作线程及数据张量的核心模板抽象库。它提供了：

- **Layout**：描述数据在内存中的排列方式
- **Tensor**：将数据数组和布局组合在一起
- **TiledMma / TiledCopy**：描述分块的矩阵乘累加和数据拷贝操作

详细文档：[CuTe 快速入门](./cpp/cute/00_quickstart.md)

### DPAS（Dot Product Accumulate Systolic）

DPAS 是 Intel GPU 上的核心矩阵运算指令，类似于 NVIDIA 的 Tensor Core。SYCL\*TLA 封装了 `XE_DPAS_TT` 模板来暴露和使用这些硬件指令。

### 尾声（Epilogue）

GEMM 计算完成后的后处理步骤。SYCL\*TLA 支持将多种后处理操作（如偏置加法、激活函数、缩放等）融合到一次 kernel 启动中，以减少内存带宽消耗。

### Block 2D 拷贝

Intel GPU 提供的 2D 块消息（Block 2D messages）可以在全局内存和寄存器之间高效移动固定大小的 2D 数据块，支持自动边界检查和布局变换。

---

## 支持的硬件

| GPU | Intel GPU 架构 | 构建目标 |
|---|---|---|
| Intel Data Center GPU Max 系列（Ponte Vecchio） | Xe-HPC | `intel_gpu_pvc` |
| Intel Arc B580 显卡（BattleMage） | Xe2 | `intel_gpu_bmg_g21` |

---

## 环境要求

- **操作系统**：仅支持 Linux
- **编译器**：Intel DPC++ 编译器（oneAPI 2025.1 及更新版本）
- **Intel 计算运行时**：25.13（含 Intel Graphics Compiler 2.10.10）
- **构建工具**：CMake 3.22+、Ninja
- **C++ 标准**：至少支持 C++17

---

## 构建项目

### 1. 设置 Intel oneAPI 环境

```bash
source /opt/intel/oneapi/setvars.sh
export CXX=icpx
export CC=icx
```

### 2. 创建构建目录并配置 CMake

**为 Intel Data Center GPU Max 系列（PVC）构建：**

```bash
mkdir build && cd build
CC=icx CXX=icpx cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_pvc
```

**为 Intel Arc B580（BMG）构建：**

```bash
mkdir build && cd build
CC=icx CXX=icpx cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21
```

**使用 G++ 作为宿主编译器（可选）：**

```bash
CC=icx CXX=icpx cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_HOST_COMPILER=g++-13 \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21
```

### 3. 构建

```bash
ninja          # 构建所有目标
ninja -j8      # 使用 8 个并行任务构建
```

### 4. 性能优化环境变量（可选）

设置这些环境变量以获得最佳性能：

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
export IGC_VISAOptions="-perfmodel"
export IGC_VectorAliasBBThreshold=100000000000
```

---

## 运行示例

### 构建并运行一个简单 GEMM 示例

```bash
# 从构建目录
ninja 00_bmg_gemm
./examples/00_bmg_gemm/00_bmg_gemm
```

预期输出：

```
Disposition: Passed
Problem Size: 5120x4096x4096x1
Cutlass GEMM Performance:     [247.159]TFlop/s  (0.6951)ms
```

### 运行单元测试

```bash
cmake --build . --target test_unit -j8
```

---

## 项目结构概览

```
sycl-tla/
├── include/                   # 头文件目录（使用本库时，将此目录加入 include 路径）
│   ├── cutlass/               # SYCL*TLA 核心模板库
│   │   ├── arch/              # Intel GPU 架构特性（包括 DPAS 指令级 GEMM）
│   │   ├── epilogue/          # 尾声操作（GEMM 后处理）
│   │   ├── gemm/              # GEMM 核心实现
│   │   ├── layout/            # 矩阵和张量的内存布局定义
│   │   └── ...                # 其他核心类型和基础运算
│   └── cute/                  # CuTe 布局代数、MMA/Copy 原子操作
│       ├── arch/              # Intel GPU 架构封装
│       ├── atom/              # MMA 和 Copy 原子操作定义
│       └── ...                # Shape、Stride、Layout、Tensor 等核心类型
│
├── examples/                  # 编程示例
│   ├── 00_bmg_gemm/           # 基础 GEMM 示例
│   ├── ...                    # 更多示例（见下文列表）
│   ├── cute/                  # CuTe 教程示例
│   └── python/                # Python 接口示例
│
├── test/                      # 测试
│   └── unit/                  # 单元测试（Google Test）
│
├── tools/                     # 工具
│   ├── library/               # SYCL*TLA 实例库
│   └── util/                  # 辅助工具（张量管理、参考实现等）
│
├── python/                    # Python 封装和生成器
├── media/docs/                # 文档目录
├── CMakeLists.txt             # 主 CMake 构建文件
├── SYCL.cmake                 # SYCL 构建规则
└── README.md                  # 项目主文档（英文）
```

---

## Intel GPU 示例一览

> **注意**：这些示例仅用于演示功能，不适用于性能基准测试。

| 示例 | 描述 |
|---|---|
| [00_bmg_gemm](../../examples/00_bmg_gemm/) | 基础 GEMM 实现，单精度输入输出 |
| [01_bmg_gemm_with_collective_builder](../../examples/01_bmg_gemm_with_collective_builder/) | 使用 CollectiveBuilder 构造 GEMM |
| [02_bmg_gemm_mixed_dtype](../../examples/02_bmg_gemm_mixed_dtype/) | 混合精度 GEMM（支持反量化） |
| [03_bmg_gemm_streamk](../../examples/03_bmg_gemm_streamk/) | 使用 Stream-K 调度器的 GEMM，改进负载均衡 |
| [04_bmg_grouped_gemm](../../examples/04_bmg_grouped_gemm/) | 分组 GEMM：一次处理多个不同大小的矩阵乘法 |
| [05_bmg_gemm_with_epilogues](../../examples/05_bmg_gemm_with_epilogues/) | 使用 Epilogue Visitor Tree (EVT) 的尾声融合示例 |
| [06_bmg_flash_attention](../../examples/06_bmg_flash_attention/) | Intel GPU 上的 Flash Attention V2 实现 |
| [07_bmg_dual_gemm](../../examples/07_bmg_dual_gemm/) | 双 GEMM 融合：共享 A 矩阵的两个 GEMM 合并到一个 kernel |
| [08_bmg_gemm_f8](../../examples/08_bmg_gemm_f8/) | FP8 浮点 GEMM |
| [09_bmg_grouped_gemm_f8](../../examples/09_bmg_grouped_gemm_f8/) | FP8 分组 GEMM |
| [10_bmg_grouped_gemm_mixed_dtype](../../examples/10_bmg_grouped_gemm_mixed_dtype/) | 混合精度分组 GEMM |
| [11_xe20_cutlass_library](../../examples/11_xe20_cutlass_library/) | CUTLASS 库接口示例 |
| [12_xe20_moe_gemm_cute_interface](../../examples/12_xe20_moe_gemm_cute_interface/) | MoE（混合专家）GEMM 与 CuTe 接口 |
| [13_bmg_gemm_bias](../../examples/13_bmg_gemm_bias/) | 带偏置加法的 GEMM |

---

## Python 接口

SYCL\*TLA 提供了 Python 接口，用于内核生成和测试。

### 安装

```bash
# 从项目根目录
pip install -e .
```

### 运行 Python 测试

```bash
export CUTLASS_USE_SYCL=1
cd python
python3 -m pytest -q
```

更多详情请参阅 [Python 文档](./python/xe_cutlass_library.md)。

---

## 常见问题排查

### 编译找不到编译器

确保已正确配置 Intel oneAPI 环境：

```bash
source /opt/intel/oneapi/setvars.sh
```

### CMake 缓存问题导致构建异常

删除构建目录后重新配置：

```bash
rm -rf build && mkdir build && cd build
# 然后重新运行 cmake 命令
```

### Python 测试导入错误

确保在项目根目录执行了安装：

```bash
pip install -e .
```

### 运行时找不到共享库

设置 `LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH=build/tools/library:$LD_LIBRARY_PATH
```

---

## 进阶学习资源

### 中文教程

- [GEMM 代码详解教程](./TUTORIAL_GEMM_CN.md) — 逐步讲解基础 GEMM 示例代码，涵盖数据类型、布局、TiledMMA、流水线、尾声等核心概念

### 英文文档

- [SYCL 构建指南](./cpp/build/building_with_sycl_support.md) — 详细的 SYCL 构建说明
- [功能列表](./cpp/functionality.md) — 各架构支持的完整功能列表
- [CUTLASS 3.x 设计](./cpp/cutlass_3x_design.md) — CUTLASS 3.x 架构设计思想
- [GEMM API 3.x](./cpp/gemm_api_3x.md) — GEMM 模板 API 详解
- [CuTe 快速入门](./cpp/cute/00_quickstart.md) — CuTe 核心概念教程
- [Xe 架构重设计](./cpp/xe_rearchitecture.md) — Intel Xe GPU 架构下的改进
- [高效 GEMM](./cpp/efficient_gemm.md) — GEMM 优化原理
- [代码组织](./cpp/code_organization.md) — 源码结构说明
- [术语表](./cpp/terminology.md) — 代码中使用的术语解释
- [变更日志](../../CHANGELOG-SYCL.md) — SYCL\*TLA 版本更新记录

### 外部资源

- [SYCL 规范（Khronos）](https://www.khronos.org/sycl/) — SYCL 标准文档
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) — Intel oneAPI 开发工具包
- [Intel DPC++ 编译器](https://github.com/intel/llvm) — 开源 DPC++ 编译器
