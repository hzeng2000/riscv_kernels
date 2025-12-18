#!/bin/bash
# Auto-generated build script for ggml_gemm_q4_0_8x8_q8_0 on sg2044
# RVV Version: rvv_1.0

set -e

# CXX="g++"
CXX="riscv64-unknown-linux-gnu-g++ -static"
CXXFLAGS="-march=rv64gcv -mabi=lp64d -O3"
DEFINES="-D__RVV_ASM_STD"
INCLUDES="-I../common"

echo "Building ggml_gemm_q4_0_8x8_q8_0 benchmark for sg2044..."

$CXX $CXXFLAGS $DEFINES $INCLUDES -o test_runner_gemm.out \
    test_runner_gemm.cpp \
    ggml_gemm_q4_0_8x8_q8_0_kernels_sg2044.cpp
    
# CXX="g++"
CXX="riscv64-unknown-linux-gnu-g++ -static"
CXXFLAGS="-march=rv64gcv_zfh -mabi=lp64d -O3"
DEFINES="-D__RVV_ASM_STD"
INCLUDES="-I../common"

echo "Building ggml_gemm_q4_0_8x8_q8_0 benchmark for sg2044..."

$CXX $CXXFLAGS $DEFINES $INCLUDES -o test_runner_gemm_zfh.out \
    test_runner_gemm.cpp \
    ggml_gemm_q4_0_8x8_q8_0_kernels_sg2044.cpp

echo "Done! Run with: ./test_runner_gemm.out -n 4096 -r 1000"
echo "Done! Run with: ./test_runner_gemm_zfh.out -n 4096 -r 1000"
