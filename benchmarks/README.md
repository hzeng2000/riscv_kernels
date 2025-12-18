# VectorWeaver Benchmarks

这是VectorWeaver生成的RISC-V向量算子的测试和基准测试目录。

## 目录结构

```
benchmarks/
├── common/                    # 通用头文件
│   └── common_defs.h         # 数据类型定义
├── rvv071/                   # RVV 0.7.1 (XTheadVector) 测试
│   ├── vec_dot_q8_0_q8_0/
│   ├── vec_dot_q4_0_q8_0/
│   ├── quantize_row_q8_0/
│   ├── rmsnorm/
│   ├── vec_silu_f32/
│   └── gemm/
├── rvv10/                    # RVV 1.0 标准 测试
│   ├── vec_dot_q8_0_q8_0/
│   ├── vec_dot_q4_0_q8_0/
│   ├── quantize_row_q8_0/
│   ├── rmsnorm/
│   ├── vec_silu_f32/
│   └── gemm/
├── patch_kernels.sh          # Kernel patch脚本
└── README.md                 # 本文件
```

## 快速开始

### 1. 生成Kernel

首先使用VectorWeaver生成优化的kernel：

```bash
cd vectorweaver

# 为RVV 0.7.1 (荔枝派4A / C910) 生成
python3 main.py -k vec_dot_q8_q8 --hw c910 --generate-all
python3 main.py -k quantize --hw c910 --generate-all
python3 main.py -k rmsnorm --hw c910 --generate-all

# 为RVV 1.0 (SG2044) 生成
python3 main.py -k vec_dot_q8_q8 --hw sg2044 --generate-all
python3 main.py -k vec_dot_q4_q8 --hw sg2044 --generate-all
python3 main.py -k quantize --hw sg2044 --generate-all
python3 main.py -k rmsnorm --hw sg2044 --generate-all

# 使用ranker模型并保存排名结果
python3 main.py -k vec_dot_q8_q8 --hw sg2044 --ranker-tune --save-ranking --dual-phase
```

### 2. Patch Kernels到Benchmarks

```bash
cd benchmarks

# Patch所有生成的kernel
./patch_kernels.sh

# 只patch RVV 1.0版本
./patch_kernels.sh rvv10

# 只patch特定算子
./patch_kernels.sh rvv071 rmsnorm
```

### 3. 编译和运行

详见各RVV版本目录下的 `how_to.md`。

## 编译环境要求

### RVV 0.7.1 (XTheadVector)
- 工具链: `riscv64-unknown-linux-gnu-gcc` (支持xtheadvector)
- 目标平台: 荔枝派4A, C906/C910/C920等

### RVV 1.0 标准
- 工具链: `riscv64-unknown-linux-gnu-gcc` (GCC 14+ 或支持RVV 1.0)
- 目标平台: SG2044, SpacemiT K1/M1等

## 测试说明

每个算子目录包含:
- `*_kernels.h` - 内核函数声明
- `*_kernels.cpp` 或 `*_kernels_*.cpp` - 内核实现
- `test_runner_*.cpp` - 测试和基准测试程序

测试程序支持的参数:
```bash
./test_runner -n <size>        # 向量/矩阵大小
              -r <runs>        # 测试运行次数
              --test-all       # 运行所有测试
              --help           # 显示帮助
```

