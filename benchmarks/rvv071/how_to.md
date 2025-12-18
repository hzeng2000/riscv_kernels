# RVV 0.7.1 (XTheadVector) 编译和测试指南

本目录包含针对 RVV 0.7.1 (XTheadVector) 的算子实现和测试。

适用平台：
- 荔枝派4A (C910 x 4)
- 平头哥 C906 / C910 / C920 系列

## 编译命令

### 通用编译选项
```bash
# 编译器: riscv64-unknown-linux-gnu-g++
# 架构: rv64gc + xtheadvector + zfhmin
# ABI: lp64d
# 必须定义的宏:
#   -D__riscv_v              # 启用RVV intrinsics
#   -D__riscv_xtheadvector   # 启用XTheadVector intrinsics  
#   -D__RVV_ASM_XTHEAD       # 启用XTheadVector ASM代码
```

### vec_dot_q8_0_q8_0
```bash
cd vec_dot_q8_0_q8_0

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_vecdot.cpp \
    vec_dot_q8_0_q8_0_kernels.cpp \
    -march=rv64gc_xtheadvector1p0_zfhmin \
    -mabi=lp64d \
    -D__riscv_v \
    -D__riscv_xtheadvector \
    -D__RVV_ASM_XTHEAD \
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
    -march=rv64gc_xtheadvector1p0_zfhmin \
    -mabi=lp64d \
    -D__riscv_v \
    -D__riscv_xtheadvector \
    -D__RVV_ASM_XTHEAD \
    -I../common \
    -O3
```

### quantize_row_q8_0
```bash
cd quantize_row_q8_0

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_quantize.cpp \
    quantize_row_q8_0_kernels.cpp \
    -march=rv64gc_xtheadvector1p0_zfhmin \
    -mabi=lp64d \
    -D__riscv_v \
    -D__riscv_xtheadvector \
    -D__RVV_ASM_XTHEAD \
    -I../common \
    -O3
```

### rmsnorm
```bash
cd rmsnorm

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_rmsnorm.cpp \
    rmsnorm_kernels.cpp \
    -march=rv64gc_xtheadvector1p0_zfhmin \
    -mabi=lp64d \
    -D__riscv_v \
    -D__riscv_xtheadvector \
    -D__RVV_ASM_XTHEAD \
    -I../common \
    -O3
```

### vec_silu_f32
```bash
cd vec_silu_f32

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_vec_silu_f32.cpp \
    vec_silu_f32_kernels.cpp \
    -march=rv64gc_xtheadvector1p0_zfhmin \
    -mabi=lp64d \
    -D__riscv_v \
    -D__riscv_xtheadvector \
    -D__RVV_ASM_XTHEAD \
    -I../common \
    -O3
```

### gemm (ggml_gemm_q4_0_8x8_q8_0)
```bash
cd gemm

riscv64-unknown-linux-gnu-g++ -o test_runner.out \
    test_runner_gemm.cpp \
    gemm_kernels.cpp \
    -march=rv64gc_xtheadvector1p0_zfhmin \
    -mabi=lp64d \
    -D__riscv_v \
    -D__riscv_xtheadvector \
    -D__RVV_ASM_XTHEAD \
    -I../common \
    -O3
```

## 批量编译脚本

创建 `build_all.sh`:
```bash
#!/bin/bash
set -e

COMMON_FLAGS="-march=rv64gc_xtheadvector1p0_zfhmin -mabi=lp64d -D__riscv_v -D__riscv_xtheadvector -D__RVV_ASM_XTHEAD -I../common -O3"

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

## 注意事项

1. **工具链版本**: 确保使用支持XTheadVector的工具链
2. **头文件**: 需要 `<riscv_vector.h>` 支持 XTheadVector intrinsics
3. **运行环境**: 需要在支持XTheadVector的RISC-V平台上运行

