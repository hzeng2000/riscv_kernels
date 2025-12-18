#!/bin/bash
# Auto-generated build script for ggml_vec_silu_f32 on th1520
# RVV Version: rvv_0.7.1
# Supports: O1/O3 optimization levels, local/cross compilation, static/dynamic linking

set -e

KERNEL_NAME="ggml_vec_silu_f32"
KERNEL_KEY="vec_silu_f32"
HW_NAME="th1520"
MARCH="rv64gc_xtheadvector1p0"
MARCH_ZFH="rv64gc_xtheadvector1p0_zfhmin"
DEFINES="-D__RVV_ASM_XTHEAD"
INCLUDES="-I../common"

# Source and output files
SRC_TEST="test_runner_${KERNEL_KEY}.cpp"
SRC_KERNEL="${KERNEL_NAME}_kernels_${HW_NAME}.cpp"

# Function to build with given compiler and optimization level
# Args: CXX, OPT_LEVEL, SUFFIX, USE_ZFH, STATIC
build_variant() {
    local CXX="$1"
    local OPT_LEVEL="$2"
    local SUFFIX="$3"
    local USE_ZFH="$4"
    local STATIC="$5"
    
    local MARCH_USED="$MARCH"
    local ZFH_SUFFIX=""
    if [ "$USE_ZFH" = "true" ]; then
        MARCH_USED="$MARCH_ZFH"
        ZFH_SUFFIX="_zfh"
    fi
    
    local STATIC_PREFIX=""
    local STATIC_FLAG=""
    if [ "$STATIC" = "true" ]; then
        STATIC_PREFIX="static_"
        STATIC_FLAG="-static"
    fi
    
    local OUTPUT="${STATIC_PREFIX}test_runner_${KERNEL_KEY}_${SUFFIX}${ZFH_SUFFIX}.out"
    local CXXFLAGS="-march=${MARCH_USED} -mabi=lp64d -${OPT_LEVEL} ${STATIC_FLAG}"
    
    echo "Building: $OUTPUT (CXX=$CXX, OPT=$OPT_LEVEL, ZFH=$USE_ZFH, STATIC=$STATIC)"
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
    echo "--- Cross-compiler builds (static) ---"
    build_variant "$CROSS_CXX" "O1" "cross_O1" "false" "true"
    build_variant "$CROSS_CXX" "O3" "cross_O3" "false" "true"
    build_variant "$CROSS_CXX" "O1" "cross_O1" "true" "true"
    build_variant "$CROSS_CXX" "O3" "cross_O3" "true" "true"
    echo ""
    echo "--- Cross-compiler builds (dynamic) ---"
    build_variant "$CROSS_CXX" "O1" "cross_O1" "false" "false"
    build_variant "$CROSS_CXX" "O3" "cross_O3" "false" "false"
    build_variant "$CROSS_CXX" "O1" "cross_O1" "true" "false"
    build_variant "$CROSS_CXX" "O3" "cross_O3" "true" "false"
    echo ""
else
    echo "[SKIP] Cross-compiler not found: $CROSS_CXX"
    echo ""
fi

# Local compiler builds (for running on local RISC-V machine)
LOCAL_CXX="g++"
# Check if we're on a RISC-V machine
if [ "$(uname -m)" = "riscv64" ]; then
    echo "--- Local builds (static) ---"
    build_variant "$LOCAL_CXX" "O1" "local_O1" "false" "true"
    build_variant "$LOCAL_CXX" "O3" "local_O3" "false" "true"
    build_variant "$LOCAL_CXX" "O1" "local_O1" "true" "true"
    build_variant "$LOCAL_CXX" "O3" "local_O3" "true" "true"
    echo ""
    echo "--- Local builds (dynamic) ---"
    build_variant "$LOCAL_CXX" "O1" "local_O1" "false" "false"
    build_variant "$LOCAL_CXX" "O3" "local_O3" "false" "false"
    build_variant "$LOCAL_CXX" "O1" "local_O1" "true" "false"
    build_variant "$LOCAL_CXX" "O3" "local_O3" "true" "false"
    echo ""
else
    echo "[SKIP] Not on RISC-V machine, skipping local builds"
    echo ""
fi

echo "=========================================="
echo "Build complete!"
echo ""
echo "Available executables:"
ls -la *test_runner_*.out 2>/dev/null || echo "  (none)"
echo ""
echo "Run examples:"
echo "  ./static_test_runner_vec_silu_f32_cross_O3.out -n 4096 -r 1000    # static linked"
echo "  ./test_runner_vec_silu_f32_cross_O3.out -n 4096 -r 1000           # dynamic linked"
