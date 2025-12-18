# RVV 1.0 编译和测试指南

本目录包含针对 RVV 1.0 标准的算子实现和测试。

适用平台：
- SG2044 (64核C920V2, VLEN=128)
- SpacemiT K1/M1 (8核X60, VLEN=256)
- 其他支持RVV 1.0的RISC-V平台

## 编译命令

### 通用编译选项
```bash
# 编译器: riscv64-unknown-linux-gnu-g++ (GCC 14+ 推荐)
# 架构: rv64gcv (包含RVV 1.0)
# ABI: lp64d
# 必须定义的宏:
#   -D__riscv_v         # 启用RVV intrinsics (通常由-march自动定义)
#   -D__RVV_ASM_STD     # 启用标准RVV 1.0 ASM代码
```

### vec_dot_q8_0_q8_0
```bash
cd vec_dot_q8_0_q8_0

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_vecdot.cpp \
    vec_dot_q8_0_q8_0_kernels.cpp \
    -march=rv64gcv_zfh \
    -mabi=lp64d \
    -D__RVV_ASM_STD \
    -I../common \
    -O3

# 运行测试
./test_runner.out -n 4096 -r 1000 --test-all
```

### vec_dot_q4_0_q8_0
```bash
cd vec_dot_q4_0_q8_0

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_vecdot_q4.cpp \
    vec_dot_q4_0_q8_0_kernels.cpp \
    -march=rv64gcv_zfh \
    -mabi=lp64d \
    -D__RVV_ASM_STD \
    -I../common \
    -O3
```

### quantize_row_q8_0
```bash
cd quantize_row_q8_0

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_quantize.cpp \
    quantize_row_q8_0_kernels.cpp \
    -march=rv64gcv_zfh \
    -mabi=lp64d \
    -D__RVV_ASM_STD \
    -I../common \
    -O3
```

### rmsnorm
```bash
cd rmsnorm

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_rmsnorm.cpp \
    rmsnorm_kernels.cpp \
    -march=rv64gcv_zfh \
    -mabi=lp64d \
    -D__RVV_ASM_STD \
    -I../common \
    -O3
```

### vec_silu_f32
```bash
cd vec_silu_f32

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_vec_silu_f32.cpp \
    vec_silu_f32_kernels.cpp \
    -march=rv64gcv_zfh \
    -mabi=lp64d \
    -D__RVV_ASM_STD \
    -I../common \
    -O3
```

### gemm (ggml_gemm_q4_0_8x8_q8_0)
```bash
cd gemm

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_gemm.cpp \
    gemm_kernels.cpp \
    -march=rv64gcv_zfh \
    -mabi=lp64d \
    -D__RVV_ASM_STD \
    -I../common \
    -O3
```

## 批量编译脚本

创建 `build_all.sh`:
```bash
#!/bin/bash
set -e

COMMON_FLAGS="-march=rv64gcv_zfh -mabi=lp64d -D__RVV_ASM_STD -I../common -O3"

for dir in */; do
    if [ -f "${dir}test_runner"*.cpp ]; then
        echo "Building ${dir}..."
        cd "$dir"
        riscv64-unknown-linux-gnu-g++ $COMMON_FLAGS -o test_runner.out *.cpp
        cd ..
    fi
done

echo "All builds complete!"
```

## SG2044 特定优化

SG2044有64个核心，可以使用多线程并行测试：

```bash
# 使用OpenMP并行
riscv64-unknown-linux-gnu-g++ -o test_runner_omp.out \
    test_runner.cpp \
    kernels.cpp \
    -march=rv64gcv_zfh \
    -mabi=lp64d \
    -D__RVV_ASM_STD \
    -I../common \
    -O3 \
    -fopenmp

# 运行时设置线程数
export OMP_NUM_THREADS=64
./test_runner_omp.out --test-all
```

## 运行测试

```bash
# 运行所有测试
for dir in */; do
    if [ -f "${dir}test_runner.out" ]; then
        echo "=== Testing ${dir} ==="
        cd "$dir"
        ./test_runner.out --test-all
        cd ..
    fi
done
```

## 与RVV 0.7.1的主要区别

| 方面 | RVV 0.7.1 (XTheadVector) | RVV 1.0 |
|-----|-------------------------|---------|
| 编译选项 | `-march=rv64gc_xtheadvector1p0_zfhmin` | `-march=rv64gcv_zfh` |
| ASM保护宏 | `-D__RVV_ASM_XTHEAD` | `-D__RVV_ASM_STD` |
| vsetvli | `th.vsetvli x0, t0, e8, m2` | `vsetvli x0, t0, e8, m2, ta, ma` |
| 内存访问 | `th.vle.v` / `th.vse.v` | `vle8.v` / `vse8.v` (需指定宽度) |
| 指令前缀 | `th.` 前缀 | 无前缀 |

## 注意事项

1. **工具链版本**: 推荐使用GCC 14+以获得最佳RVV 1.0支持
2. **VLEN**: 
   - SG2044 (C920V2): VLEN=128位
   - SpacemiT K1 (X60): VLEN=256位
   - 请确保代码正确处理不同VLEN
3. **Intrinsics**: RVV 1.0的intrinsics API与0.7.1类似，但底层指令不同
4. **多核**: SG2044有64核，考虑使用多线程测试以充分利用硬件

