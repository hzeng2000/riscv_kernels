#pragma once

#include "../common/common_defs.h" // Includes block_q8_0, QK8_0, etc.

// A unified function pointer type for all implementations
using quantize_row_q8_0_t = void (*)(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);

// --- Kernel Declarations ---

// 1. Reference C++ implementation
void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
// 2. RVV Intrinsics version (your correct baseline)
void quantize_row_q8_0_rvv(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0_rvv_asm_unroll1(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0_rvv_asm_one_block(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0_rvv_asm_unroll2(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0_rvv_asm_unroll4(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0_rvv_asm_unroll4_interleaved(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
#endif
