#pragma once

#include "../common/common_defs.h"

// 函数指针类型
using ggml_rmsnorm_f32_t = void (*)(int n, float *y, const float *x, float eps);

// --- 内核函数声明 ---

// 1. 基准纯 C++ 实现
void ggml_rmsnorm_f32_scalar(int n, float *y, const float *x, float eps);

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
// 2. RVV Intrinsics 实现
void ggml_rmsnorm_f32_rvv_intrinsics(int n, float *y, const float *x, float eps);
void ggml_rmsnorm_f32_rvv_asm_unroll2_serial(int n, float *y, const float *x, float eps);
void ggml_rmsnorm_f32_rvv_asm_unroll2_pipelined(int n, float *y, const float *x, float eps);
void ggml_rmsnorm_f32_rvv_asm_unroll4_pipelined(int n, float *y, const float *x, float eps);
void ggml_rmsnorm_f32_auto_generated(int n, float *y, const float *x, float eps);
#endif