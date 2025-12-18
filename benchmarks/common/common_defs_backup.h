#pragma once

#include <cstdint>
#include <cassert>

// from ggml
typedef uint16_t ggml_half;
#define GGML_RESTRICT __restrict__
#define QK8_0 32
typedef struct
{
    ggml_half d;      // delta (缩放因子)
    int8_t qs[QK8_0]; // quants (量化后的权重)
} block_q8_0;

#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

template <int K> constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}
template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

// control size
static_assert(sizeof(block<4, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(sizeof(block<4, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(sizeof(block<8, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(sizeof(block<8, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

typedef uint16_t ggml_fp16_t;
static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h)
{
    float f;
    __asm__(
        "fmv.h.x %[f], %[h]\n\t"
        "fcvt.s.h %[f], %[f]"
        : [f] "=&f"(f)
        : [h] "r"(h));
    return f;
}
static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f)
{
    ggml_fp16_t res;
    __asm__(
        "fcvt.h.s %[f], %[f]\n\t"
        "fmv.x.h %[h], %[f]"
        : [h] "=&r"(res)
        : [f] "f"(f));
    return res;
}
#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)
#define GGML_CPU_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_CPU_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)