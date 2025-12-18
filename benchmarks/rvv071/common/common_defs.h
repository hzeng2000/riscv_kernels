#pragma once

#include <cstdint>
#include <cassert>
#include <math.h>   // fabsf
#include <string.h> // memcpy
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
// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#if defined(__ARM_NEON) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
    #define GGML_CPU_COMPUTE_FP16_TO_FP32(x) neon_compute_fp16_to_fp32(x)
    #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) neon_compute_fp32_to_fp16(x)

    #define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)

    static inline float neon_compute_fp16_to_fp32(ggml_fp16_t h) {
        __fp16 tmp;
        memcpy(&tmp, &h, sizeof(ggml_fp16_t));
        return (float)tmp;
    }

    static inline ggml_fp16_t neon_compute_fp32_to_fp16(float f) {
        ggml_fp16_t res;
        __fp16 tmp = f;
        memcpy(&res, &tmp, sizeof(ggml_fp16_t));
        return res;
    }
#elif defined(__riscv_xtheadvector) && defined(__riscv_zfhmin)
    static inline float riscv_compute_fp16_to_fp32(ggml_fp16_t h) {
        float f;
        __asm__(
            "fmv.h.x %[f], %[h]\n\t"
            "fcvt.s.h %[f], %[f]"
            : [f] "=&f" (f)
            : [h] "r" (h)
        );
        return f;
    }

    static inline ggml_fp16_t riscv_compute_fp32_to_fp16(float f) {
        ggml_fp16_t res;
        __asm__(
            "fcvt.h.s %[f], %[f]\n\t"
            "fmv.x.h %[h], %[f]"
            : [h] "=&r" (res)
            : [f] "f" (f)
        );
        return res;
    }

    #define GGML_CPU_COMPUTE_FP16_TO_FP32(x) riscv_compute_fp16_to_fp32(x)
    #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) riscv_compute_fp32_to_fp16(x)
    #define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
    #define GGML_CPU_FP32_TO_FP16(x) GGML_CPU_COMPUTE_FP32_TO_FP16(x)
#endif

// precomputed f32 table for f16 (256 KB)
// Simple standalone definition and initialization
inline float ggml_table_f32_f16[1 << 16];

// Simple initializer that runs at startup
struct ggml_table_f32_f16_initializer {
    ggml_table_f32_f16_initializer() {
        for (int i = 0; i < (1 << 16); ++i) {
            union {
                uint16_t u16;
                ggml_fp16_t fp16;
            } u = {(uint16_t)i};
            ggml_table_f32_f16[i] = ggml_compute_fp16_to_fp32(u.fp16);
        }
    }
};
inline ggml_table_f32_f16_initializer ggml_table_f32_f16_init_instance;

// On ARM NEON, it's quicker to directly convert x -> x instead of calling into ggml_lookup_fp16_to_fp32,
// so we define GGML_CPU_FP16_TO_FP32 and GGML_CPU_FP32_TO_FP16 elsewhere for NEON.
// This is also true for POWER9.
#if !defined(GGML_CPU_FP16_TO_FP32)
inline static float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return ggml_table_f32_f16[s];
}

#define GGML_CPU_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#endif

#if !defined(GGML_CPU_FP32_TO_FP16)
#define GGML_CPU_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)
#endif