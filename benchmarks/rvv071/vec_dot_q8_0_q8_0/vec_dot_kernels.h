#pragma once

#include "../common/common_defs.h"

// --- Function Pointer Type for all vec_dot implementations ---
using vec_dot_q8_0_q8_0_t = void (*)(int n, float *s, const void *vx, const void *vy);

// --- Kernel Declarations ---

// 1. Baseline Scalar Implementation
void ggml_vec_dot_q8_0_q8_0_scalar(int n, float *s, const void *vx, const void *vy);

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
// 2. RVV Intrinsics
void ggml_vec_dot_q8_0_q8_0_rvv_intrinsics(int n, float *s, const void *vx, const void *vy);
#endif

#if defined(__RVV_ASM_XTHEAD)
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll1(int n, float *s, const void *vx, const void *vy);
// 3. ILP Axis: Unroll=2 (Serial) - Simple and direct unrolling
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll2(int n, float *s, const void *vx, const void *vy);

// 4. ILP Axis: Unroll=2 (Fused) - Your "大乘法" idea, corrected and implemented
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll2_fused(int n, float *s, const void *vx, const void *vy);

// 5. ILP Axis: Unroll=4 (User's original version)
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll4(int n, float *s, const void *vx, const void *vy);

// 6. Memory & ILP Axis: Unroll=4 (Pipelined) - Reorders instructions to hide latency
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll4_pipelined(int n, float *s, const void *vx, const void *vy);

// 7. Computation Axis: FP64 Accumulator - Uses double for final accumulation
void ggml_vec_dot_q8_0_q8_0_rvv_asm_fp64_accum(int n, float *s, const void *vx, const void *vy);
// 8. Accumulation Axis: Blocked Accumulation
void ggml_vec_dot_q8_0_q8_0_rvv_asm_blocked_accum(int n, float *s, const void *vx, const void *vy);
// 9. Memory Axis: Unroll=4 with Prefetching
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll4_prefetch(int n, float *s, const void *vx, const void *vy);
void ggml_vec_dot_q8_0_q8_0_rvv_asm_auto_generated(int n, float *s, const void *vx, const void *vy);
#endif