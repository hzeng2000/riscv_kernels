#!/bin/bash
# Auto-generated build script for ggml_vec_silu_f32 on th1520
# RVV Version: rvv_0.7.1
# Supports: O1/O3 optimization levels, local/cross compilation

set -e

KERNEL_NAME="ggml_vec_silu_f32"
KERNEL_KEY="vec_silu_f32"
HW_NAME="th1520"
MARCH="rv64gc_xtheadvector1p0_zfhmin"
MARCH_ZFH="rv64gc_xtheadvector1p0_zfhmin"
DEFINES="-D__riscv_v -D__riscv_xtheadvector -D__RVV_ASM_XTHEAD"
INCLUDES="-I../common"

# Source and output files
SRC_TEST="test_runner_${KERNEL_KEY}.cpp"
SRC_KERNEL="${KERNEL_NAME}_kernels_${HW_NAME}.cpp"

# Function to build with given compiler and optimization level
build_variant() {
    local CXX="$1"
    local OPT_LEVEL="$2"
    local SUFFIX="$3"
    local USE_ZFH="$4"
    
    local MARCH_USED="$MARCH"
    local ZFH_SUFFIX=""
    if [ "$USE_ZFH" = "true" ]; then
        MARCH_USED="$MARCH_ZFH"
        ZFH_SUFFIX="_zfh"
    fi
    
    local OUTPUT="test_runner_${KERNEL_KEY}_${SUFFIX}${ZFH_SUFFIX}.out"
    local CXXFLAGS="-march=${MARCH_USED} -mabi=lp64d -${OPT_LEVEL} -static"
    
    echo "Building: $OUTPUT (CXX=$CXX, OPT=$OPT_LEVEL, ZFH=$USE_ZFH)"
    $CXX $CXXFLAGS $DEFINES $INCLUDES -o "$OUTPUT" "$SRC_TEST" "$SRC_KERNEL" || {
        echo "  [SKIP] Build failed (compiler may not be available)"
        return 1
    }
    echo "  [OK] $OUTPUT"
}

echo "=========================================="
echo "Building ggml_vec_silu_f32 benchmark for th1520"
echo "=========================================="
echo ""

# Cross-compiler builds (for running on RISC-V hardware)
CROSS_CXX="riscv64-unknown-linux-gnu-g++"
if command -v $CROSS_CXX &> /dev/null; then
    echo "--- Cross-compiler builds ---"
    build_variant "$CROSS_CXX" "O1" "cross_O1" "false"
    build_variant "$CROSS_CXX" "O3" "cross_O3" "false"
    build_variant "$CROSS_CXX" "O1" "cross_O1" "true"
    build_variant "$CROSS_CXX" "O3" "cross_O3" "true"
    echo ""
else
    echo "[SKIP] Cross-compiler not found: $CROSS_CXX"
    echo ""
fi

# Local compiler builds (for running on local RISC-V machine)
LOCAL_CXX="g++"
# Check if we're on a RISC-V machine
if [ "$(uname -m)" = "riscv64" ]; then
    echo "--- Local builds (native RISC-V) ---"
    build_variant "$LOCAL_CXX" "O1" "local_O1" "false"
    build_variant "$LOCAL_CXX" "O3" "local_O3" "false"
    build_variant "$LOCAL_CXX" "O1" "local_O1" "true"
    build_variant "$LOCAL_CXX" "O3" "local_O3" "true"
    echo ""
else
    echo "[SKIP] Not on RISC-V machine, skipping local builds"
    echo ""
fi

echo "=========================================="
echo "Build complete!"
echo ""
echo "Available executables:"
ls -la test_runner_*.out 2>/dev/null || echo "  (none)"
echo ""
echo "Run examples:"
echo "  ./test_runner_vec_silu_f32_cross_O3.out -n 4096 -r 1000"
echo "  ./test_runner_vec_silu_f32_local_O3.out -n 4096 -r 1000"
