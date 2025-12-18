#pragma once

#include "../common_defs.h"

// --- Function Pointer Type for all vec_silu implementations ---
using vec_silu_f32_t = void (*)(const int n, float * y, const float * x);

// --- Kernel Declarations ---

// 1. Baseline Scalar Implementation
void ggml_vec_silu_f32_scalar(const int n, float * y, const float * x);

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
// 2. RVV Intrinsics Implementation (Fast Approximation)
void ggml_vec_silu_f32_rvv_intrinsics_fast(const int n, float * y, const float * x);
#endif

#if defined(__RVV_ASM_XTHEAD)
// 3. RVV Assembly, Unroll=2, Serial (Fast Approximation)
void ggml_vec_silu_f32_rvv_asm_unroll2_serial_fast(const int n, float * y, const float * x);

// 4. RVV Assembly, Unroll=2, Pipelined (Fast Approximation)
void ggml_vec_silu_f32_rvv_asm_unroll2_pipelined_fast(const int n, float * y, const float * x);
void ggml_vec_silu_f32_asm_auto_generated(const int n, float * y, const float * x);
#endif
