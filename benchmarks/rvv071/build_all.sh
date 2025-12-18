#!/bin/bash
# build_all.sh - 编译所有RVV 0.7.1 (XTheadVector) 算子测试
#
# 用法: ./build_all.sh [kernel_name]
#   kernel_name: 可选，指定只编译某个算子

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 编译选项
CXX="riscv64-unknown-linux-gnu-g++"
CXXFLAGS="-march=rv64gc_xtheadvector1p0_zfhmin -mabi=lp64d -O3"
DEFINES="-D__riscv_v -D__riscv_xtheadvector -D__RVV_ASM_XTHEAD"
INCLUDES="-I../common"

KERNEL_FILTER="${1:-all}"

build_kernel() {
    local dir="$1"
    local test_runner="$2"
    local kernels="$3"
    
    if [ "$KERNEL_FILTER" != "all" ] && [ "$KERNEL_FILTER" != "$dir" ]; then
        return
    fi
    
    if [ -d "$dir" ]; then
        echo "=== Building $dir ==="
        cd "$dir"
        
        if [ -f "$test_runner" ] && [ -f "$kernels" ]; then
            $CXX $CXXFLAGS $DEFINES $INCLUDES -o test_runner.out "$test_runner" "$kernels"
            echo "    -> Created: $dir/test_runner.out"
        else
            echo "    [SKIP] Missing files: $test_runner or $kernels"
        fi
        
        cd ..
    fi
}

echo "=============================================="
echo "  Building RVV 0.7.1 (XTheadVector) Kernels"
echo "=============================================="
echo "Compiler: $CXX"
echo "Flags: $CXXFLAGS $DEFINES"
echo ""

build_kernel "rmsnorm" "test_runner_rmsnorm.cpp" "rmsnorm_kernels.cpp"
build_kernel "vec_dot_q8_0_q8_0" "test_runner_vecdot.cpp" "vec_dot_kernels.cpp"
build_kernel "quantize_row_q8_0" "test_runner_quantize.cpp" "quantize_kernels.cpp"
build_kernel "vec_silu_f32" "test_runner_vec_silu_f32.cpp" "vec_silu_f32_kernels.cpp"
build_kernel "gemm" "test_runner_gemm.cpp" "gemm_kernels.cpp"
# vec_dot_q4_0_q8_0 需要单独的文件名
if [ -d "vec_dot_q4_0_q8_0" ]; then
    cd vec_dot_q4_0_q8_0
    if [ -f "test_runner_vecdot_q4.cpp" ] 2>/dev/null; then
        build_kernel "vec_dot_q4_0_q8_0" "test_runner_vecdot_q4.cpp" "vec_dot_q4_0_q8_0_kernels.cpp"
    fi
    cd ..
fi

echo ""
echo "=============================================="
echo "  Build Complete!"
echo "=============================================="

