# GEMM Benchmark for SG2044 (RVV 1.0)

## 编译

```bash
# 标准版本（不支持 FP16）
./build_gemm.sh

# 或者使用支持 FP16 的版本
g++ -march=rv64gcv_zfh -mabi=lp64d -O3 -D__RVV_ASM_STD -I../common \
    -o test_runner_gemm.out \
    test_runner_gemm.cpp \
    ggml_gemm_q4_0_8x8_q8_0_kernels_sg2044.cpp
```

## 运行

```bash
# 默认参数 (M=128, N=256, K=512, runs=10, 运行所有测试)
./test_runner_gemm.out

# 自定义参数
./test_runner_gemm.out -M 256 -N 512 -K 1024 -r 100

# 只运行 intrinsics 版本
./test_runner_gemm.out --test-intrinsics

# 只运行汇编版本
./test_runner_gemm.out --test-asm

# 运行所有测试
./test_runner_gemm.out --test-all
```

## 参数说明

- `-M <rows>`: 矩阵 A 的行数（必须是 4 的倍数）
- `-N <cols>`: 矩阵 B 的列数（必须是 8 的倍数）
- `-K <inner>`: 内积维度（必须是 32 的倍数）
- `-r <runs>`: 测试重复次数
- `--test-all`: 运行所有可用测试
- `--test-intrinsics`: 只运行 RVV intrinsics 版本
- `--test-asm`: 只运行汇编优化版本

## 生成的 Kernel

- `ggml_gemm_q4_0_8x8_q8_0_scalar`: 标量 C++ 基准实现
- `ggml_gemm_q4_0_8x8_q8_0_rvv_intrinsics`: RVV intrinsics 实现
- `ggml_gemm_q4_0_8x8_q8_0_baseline`: 汇编基准实现 (unroll=1)
- `ggml_gemm_q4_0_8x8_q8_0_asm_unroll2`: 汇编优化 (unroll=2, serial)
- `ggml_gemm_q4_0_8x8_q8_0_asm_unroll2_interleaved`: 汇编优化 (unroll=2, interleaved)

## 输出格式

测试会显示每个kernel的：
- 平均执行时间（毫秒）
- 与标量baseline的平均差异（用于正确性验证）

示例输出：
```
M=128, N=256, K=512, Runs=10
Preparing test data...
Data quantized.

--- GEMM Q4_0_8x8 * Q8_0 Correctness & Performance Check ---
Kernel Name              Avg Time (ms)     Avg Diff
------------------------------------------------------------
Scalar C++               2.345678          0
RVV Intrinsics           0.456789          0.000012
Asm Unroll=1 (baseline)  0.523456          0.000015
Asm Unroll=2 Serial      0.398765          0.000011
Asm Unroll=2 Interleaved 0.387654          0.000010
```

