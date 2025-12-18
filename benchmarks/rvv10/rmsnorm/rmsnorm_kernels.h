#pragma once

#include "../common/common_defs.h"

// 函数指针类型
using ggml_rmsnorm_f32_t = void (*)(int n, float *y, const float *x, float eps);

// --- 内核函数声明 ---
// RVV 1.0 版本

// 1. 基准纯 C++ 实现
void ggml_rmsnorm_f32_scalar(int n, float *y, const float *x, float eps);

#if defined(__riscv_v)
// 2. RVV 1.0 Intrinsics 实现
void ggml_rmsnorm_f32_rvv_intrinsics(int n, float *y, const float *x, float eps);

// 3. RVV 1.0 ASM 实现 (由VectorWeaver生成)
// TODO: 根据实际生成的kernel添加声明
// void ggml_rmsnorm_f32_baseline(int n, float *y, const float *x, float eps);
// void ggml_rmsnorm_f32_asm_unroll2(int n, float *y, const float *x, float eps);
// void ggml_rmsnorm_f32_asm_unroll4_interleaved(int n, float *y, const float *x, float eps);
#endif