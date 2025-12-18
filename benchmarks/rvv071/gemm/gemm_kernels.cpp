#include "ggml_gemm_q4_0_8x8_q8_0_kernels.h"
#include "../common/common_defs.h"
#include <cassert>

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
#include <riscv_vector.h>
#endif

// 1. Baseline Scalar Implementation
void ggml_gemm_q4_0_8x8_q8_0_scalar(
    int M, int N, int K,
    const void * A, const void * B, float * C
) {
    const int qk = QK8_0;
    const int nb = K / qk;
    const int ncols_interleaved = 8;
    const int M_div_4 = M / 4;

    assert (K % qk == 0);
    assert (M % 4 == 0);
    assert (N % ncols_interleaved == 0);

    for (int y = 0; y < M_div_4; y++) {
        const block_q8_0x4 * a_ptr_base = (const block_q8_0x4 *) A + (y * nb);
        for (int x = 0; x < N / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr_base = (const block_q4_0x8 *) B + (x * nb);
            float sumf[4][8] = {{0.0f}};

            for (int l = 0; l < nb; l++) {
                const block_q8_0x4 * a_ptr = a_ptr_base + l;
                const block_q4_0x8 * b_ptr = b_ptr_base + l;
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        int32_t sumi = 0;
                        for (int k = 0; k < qk/2; k++) {
                             const uint8_t b_packed = b_ptr->qs[k*ncols_interleaved + j];
                             const int8_t v0 = (b_packed & 0x0F) << 4;
                             const int8_t v1 = (b_packed & 0xF0);

                             const int8_t a0 = a_ptr->qs[m*qk + k];
                             const int8_t a1 = a_ptr->qs[m*qk + k + qk/2];

                             sumi += (v0 * a0 + v1 * a1);
                        }
                        sumf[m][j] += (float)(sumi >> 4) * GGML_FP16_TO_FP32(b_ptr->d[j]) * GGML_FP16_TO_FP32(a_ptr->d[m]);
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    C[(y * 4 + m) * N + (x * ncols_interleaved + j)] = sumf[m][j];
                }
            }
        }
    }
}


#if defined(__riscv_v) || defined(__riscv_xtheadvector)
// 2. RVV Intrinsics Implementation
void ggml_gemm_q4_0_8x8_q8_0_rvv_intrinsics(
    int M, int N, int K,
    const void * A, const void * B, float * C
) {
    const int qk = QK8_0;
    const int nb = K / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    const int M_div_4 = M / 4;
    assert (K % qk == 0);
    assert (M % 4 == 0);
    assert (N % ncols_interleaved == 0);
    for (int y = 0; y < M_div_4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) A + (y * nb);
        for (int x = 0; x < N / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) B + (x * nb);
            float sumf[4][8] = {{0.0f}};
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            const size_t vl = blocklen;
                            const int8_t * b_qs_ptr = &b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen];
                            const int8_t * a_qs_ptr_0 = &a_ptr[l].qs[k * 4 * blocklen + m * blocklen];
                            const int8_t * a_qs_ptr_1 = &a_ptr[l].qs[k * 4 * blocklen + m * blocklen + qk / 2 * 4];
                            vuint8m1_t vb_packed = __riscv_vle8_v_u8m1(reinterpret_cast<const uint8_t *>(b_qs_ptr), vl);
                            vint8m1_t va_0 = __riscv_vle8_v_i8m1(a_qs_ptr_0, vl);
                            vint8m1_t va_1 = __riscv_vle8_v_i8m1(a_qs_ptr_1, vl);
                            vint8m1_t vb_packed_signed = __riscv_vreinterpret_v_u8m1_i8m1(vb_packed);
                            vint8m1_t vb_0 = __riscv_vsll_vx_i8m1(__riscv_vand_vx_i8m1(vb_packed_signed, 0x0F, vl), 4, vl);
                            vint8m1_t vb_1 = __riscv_vand_vx_i8m1(vb_packed_signed, -16, vl);
                            vint16m2_t vprod_0 = __riscv_vwmul_vv_i16m2(vb_0, va_0, vl);
                            vint16m2_t vprod_1 = __riscv_vwmul_vv_i16m2(vb_1, va_1, vl);
                            vint16m2_t vsum16 = __riscv_vadd_vv_i16m2(vprod_0, vprod_1, vl);
                            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
                            vint32m1_t vsum32 = __riscv_vwredsum_vs_i16m2_i32m1(vsum16, v_zero, vl);
                            int sumi = __riscv_vmv_x_s_i32m1_i32(vsum32);
                            sumf[m][j] += (float)(sumi >> 4) * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    C[(y * 4 + m) * N + (x * ncols_interleaved + j)] = sumf[m][j];
                }
            }
        }
    }
}
#endif

#if defined(__RVV_ASM_XTHEAD)

// Helper function for processing the tail-end blocks (ASM version)
static inline void ggml_gemm_q4_0_8x8_q8_0_rvv_tail_processing_asm(
    const block_q8_0x4* a_ptr,
    const block_q4_0x8* b_ptr,
    float* sumf_mj,
    int m,
    int j
) {
    const int qk = QK8_0;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    int32_t sumi = 0;

    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
        int32_t psum;
        const int8_t * b_qs_ptr = &b_ptr->qs[k * ncols_interleaved * blocklen + j * blocklen];
        const int8_t * a_qs_ptr_0 = &a_ptr->qs[k * 4 * blocklen + m * blocklen];
        const int8_t * a_qs_ptr_1 = &a_ptr->qs[k * 4 * blocklen + m * blocklen + qk / 2 * 4];
        
        asm volatile (
            "li t0, 8\n\t"
            "th.vsetvli x0, t0, e8, m1\n\t"
            "th.vle.v v8, (%[b_ptr])\n\t"
            "th.vle.v v10, (%[a0_ptr])\n\t"
            "th.vle.v v11, (%[a1_ptr])\n\t"
            "th.vand.vi v9, v8, 15\n\t"
            "th.vsll.vi v9, v9, 4\n\t"
            "th.vand.vi v8, v8, -16\n\t"
            "th.vwmul.vv v16, v9, v10\n\t"
            "th.vwmacc.vv v16, v8, v11\n\t"
            "th.vsetvli x0, t0, e16, m2\n\t"
            "th.vmv.s.x v24, x0\n\t"
            "th.vwredsum.vs v24, v16, v24\n\t"
            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.x.s %[p_out], v24\n\t"
            : [p_out] "=r"(psum)
            : [b_ptr]"r"(b_qs_ptr), [a0_ptr]"r"(a_qs_ptr_0), [a1_ptr]"r"(a_qs_ptr_1)
            : "t0", "memory", "v8", "v9", "v10", "v11", "v16", "v17", "v24"
        );
        sumi += psum;
    }
    *sumf_mj += (float)(sumi >> 4) * GGML_FP16_TO_FP32(b_ptr->d[j]) * GGML_FP16_TO_FP32(a_ptr->d[m]);
}

// --- UNROLL = 1 ---

void ggml_gemm_q4_0_8x8_q8_0_rvv_asm_unroll1_serial(
    int M, int N, int K,
    const void * A, const void * B, float * C
) {
    const int nb = K / QK8_0;
    const int M_div_4 = M / 4;
    const int N_div_8 = N / 8;

    for (int y = 0; y < M_div_4; y++) {
        const block_q8_0x4 * a_ptr_base = (const block_q8_0x4 *) A + (y * nb);
        for (int x = 0; x < N_div_8; x++) {
            const block_q4_0x8 * b_ptr_base = (const block_q4_0x8 *) B + (x * nb);
            float sumf[4][8] = {{0.0f}};
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    for (int l = 0; l < nb; l++) {
                        ggml_gemm_q4_0_8x8_q8_0_rvv_tail_processing_asm(a_ptr_base + l, b_ptr_base + l, &sumf[m][j], m, j);
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    C[(y * 4 + m) * N + (x * 8 + j)] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q4_0_8x8_q8_0_rvv_asm_unroll1_pipelined(
    int M, int N, int K,
    const void * A, const void * B, float * C
) {
    ggml_gemm_q4_0_8x8_q8_0_rvv_asm_unroll1_serial(M, N, K, A, B, C);
}

void ggml_gemm_q4_0_8x8_q8_0_rvv_asm_unroll1_fused(
    int M, int N, int K,
    const void * A, const void * B, float * C
) {
    const int qk = QK8_0;
    const int nb = K / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    const int M_div_4 = M / 4;

    for (int y = 0; y < M_div_4; y++) {
        const block_q8_0x4 * a_ptr_base = (const block_q8_0x4 *) A + (y * nb);
        for (int x = 0; x < N / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr_base = (const block_q4_0x8 *) B + (x * nb);
            float sumf[4][8] = {{0.0f}};

            for (int l = 0; l < nb; l++) {
                 const block_q8_0x4 * a_ptr = a_ptr_base + l;
                 const block_q4_0x8 * b_ptr = b_ptr_base + l;
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        int32_t sumi_block = 0;
                        const int8_t * b_qs_ptr = &b_ptr->qs[j * blocklen];
                        const int8_t * a_qs_ptr_0 = &a_ptr->qs[m * blocklen];
                        const int8_t * a_qs_ptr_1 = &a_ptr->qs[m * blocklen + qk/2*4];

                        asm volatile (
                            "li t0, 16\n\t"
                            "th.vsetvli x0, t0, e8, m1\n\t"
                            "th.vle.v v9, (%[b_ptr])\n\t"
                            "th.vand.vi v8, v9, 15\n\t"
                            "th.vsll.vi v8, v8, 4\n\t"
                            "th.vand.vi v9, v9, -16\n\t"

                            "li t0, 32\n\t"
                            "th.vsetvli x0, t0, e8, m2\n\t"
                            "th.vle.v v10, (%[a_ptr])\n\t"
                            "th.vwmul.vv v12, v8, v10\n\t"

                            "th.vsetvli x0, t0, e16, m4\n\t"
                            "th.vmv.s.x v16, x0\n\t"
                            "th.vwredsum.vs v16, v12, v16\n\t"
                            "th.vsetvli x0, t0, e32, m1\n\t"
                            "th.vmv.x.s %[p_out], v16\n\t"
                            : [p_out] "=r"(sumi_block)
                            : [b_ptr]"r"(b_qs_ptr), [a_ptr]"r"(a_qs_ptr_0)
                            : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"
                        );
                        sumf[m][j] += (float)(sumi_block >> 4) * GGML_FP16_TO_FP32(b_ptr->d[j]) * GGML_FP16_TO_FP32(a_ptr->d[m]);
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    C[(y * 4 + m) * N + (x * ncols_interleaved + j)] = sumf[m][j];
                }
            }
        }
    }
}


// --- UNROLL = 2 ---

void ggml_gemm_q4_0_8x8_q8_0_rvv_asm_unroll2_serial(
    int M, int N, int K,
    const void * A, const void * B, float * C
) {
    const int qk = QK8_0;
    const int nb = K / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    const int M_div_4 = M / 4;

    for (int y = 0; y < M_div_4; y++) {
        const block_q8_0x4 * a_ptr_base = (const block_q8_0x4 *) A + (y * nb);
        for (int x = 0; x < N / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr_base = (const block_q4_0x8 *) B + (x * nb);
            float sumf[4][8] = {{0.0f}};

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    int l = 0;
                    for (; l + 1 < nb; l += 2) {
                        const block_q8_0x4 * a_ptr0 = a_ptr_base + l;
                        const block_q8_0x4 * a_ptr1 = a_ptr_base + l + 1;
                        const block_q4_0x8 * b_ptr0 = b_ptr_base + l;
                        const block_q4_0x8 * b_ptr1 = b_ptr_base + l + 1;

                        int32_t sumi0_block = 0;
                        int32_t sumi1_block = 0;

                        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                            int32_t psum0, psum1;
                            const int8_t * b_qs_ptr0 = &b_ptr0->qs[k * ncols_interleaved * blocklen + j * blocklen];
                            const int8_t * a_qs_ptr0_0 = &a_ptr0->qs[k * 4 * blocklen + m * blocklen];
                            const int8_t * a_qs_ptr0_1 = &a_ptr0->qs[k * 4 * blocklen + m * blocklen + qk / 2 * 4];
                            const int8_t * b_qs_ptr1 = &b_ptr1->qs[k * ncols_interleaved * blocklen + j * blocklen];
                            const int8_t * a_qs_ptr1_0 = &a_ptr1->qs[k * 4 * blocklen + m * blocklen];
                            const int8_t * a_qs_ptr1_1 = &a_ptr1->qs[k * 4 * blocklen + m * blocklen + qk / 2 * 4];

                            asm volatile (
                                "li t0, 8\n\t"
                                "th.vsetvli x0, t0, e8, m1\n\t"
                                "th.vle.v v8, (%[b0_ptr])\n\t"
                                "th.vle.v v10, (%[a00_ptr])\n\t"
                                "th.vle.v v11, (%[a01_ptr])\n\t"
                                "th.vand.vi v9, v8, 15\n\t"
                                "th.vsll.vi v9, v9, 4\n\t"
                                "th.vand.vi v8, v8, -16\n\t"
                                "th.vwmul.vv v16, v9, v10\n\t"
                                "th.vwmacc.vv v16, v8, v11\n\t"
                                "th.vle.v v8, (%[b1_ptr])\n\t"
                                "th.vle.v v10, (%[a10_ptr])\n\t"
                                "th.vle.v v11, (%[a11_ptr])\n\t"
                                "th.vand.vi v9, v8, 15\n\t"
                                "th.vsll.vi v9, v9, 4\n\t"
                                "th.vand.vi v8, v8, -16\n\t"
                                "th.vwmul.vv v20, v9, v10\n\t"
                                "th.vwmacc.vv v20, v8, v11\n\t"
                                "th.vsetvli x0, t0, e16, m2\n\t"
                                "th.vmv.s.x v24, x0\n\t"
                                "th.vwredsum.vs v24, v16, v24\n\t"
                                "th.vmv.s.x v28, x0\n\t"
                                "th.vwredsum.vs v28, v20, v28\n\t"
                                "th.vsetvli x0, t0, e32, m1\n\t"
                                "th.vmv.x.s %[psum0], v24\n\t"
                                "th.vmv.x.s %[psum1], v28\n\t"
                                : [psum0] "=r"(psum0), [psum1] "=r"(psum1)
                                : [b0_ptr] "r"(b_qs_ptr0), [a00_ptr] "r"(a_qs_ptr0_0), [a01_ptr] "r"(a_qs_ptr0_1),
                                  [b1_ptr] "r"(b_qs_ptr1), [a10_ptr] "r"(a_qs_ptr1_0), [a11_ptr] "r"(a_qs_ptr1_1)
                                : "t0", "memory", "v8", "v9", "v10", "v11", "v16", "v17", "v20", "v21", "v24", "v28"
                            );
                            sumi0_block += psum0;
                            sumi1_block += psum1;
                        }
                        sumf[m][j] += (float)(sumi0_block >> 4) * GGML_FP16_TO_FP32(b_ptr0->d[j]) * GGML_FP16_TO_FP32(a_ptr0->d[m]);
                        sumf[m][j] += (float)(sumi1_block >> 4) * GGML_FP16_TO_FP32(b_ptr1->d[j]) * GGML_FP16_TO_FP32(a_ptr1->d[m]);
                    }
                    for (; l < nb; l++) {
                        ggml_gemm_q4_0_8x8_q8_0_rvv_tail_processing_asm(a_ptr_base + l, b_ptr_base + l, &sumf[m][j], m, j);
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    C[(y * 4 + m) * N + (x * ncols_interleaved + j)] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q4_0_8x8_q8_0_rvv_asm_unroll2_pipelined(
    int M, int N, int K,
    const void * A, const void * B, float * C
) {
    const int qk = QK8_0;
    const int nb = K / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    const int M_div_4 = M / 4;

    for (int y = 0; y < M_div_4; y++) {
        const block_q8_0x4 * a_ptr_base = (const block_q8_0x4 *) A + (y * nb);
        for (int x = 0; x < N / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr_base = (const block_q4_0x8 *) B + (x * nb);
            float sumf[4][8] = {{0.0f}};

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    int l = 0;
                    for (; l + 1 < nb; l += 2) {
                        const block_q8_0x4 * a_ptr0 = a_ptr_base + l;
                        const block_q8_0x4 * a_ptr1 = a_ptr_base + l + 1;
                        const block_q4_0x8 * b_ptr0 = b_ptr_base + l;
                        const block_q4_0x8 * b_ptr1 = b_ptr_base + l + 1;

                        int32_t sumi0_block = 0;
                        int32_t sumi1_block = 0;

                        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                            int32_t psum0, psum1;
                            const int8_t * b_qs_ptr0 = &b_ptr0->qs[k * ncols_interleaved * blocklen + j * blocklen];
                            const int8_t * a_qs_ptr0_0 = &a_ptr0->qs[k * 4 * blocklen + m * blocklen];
                            const int8_t * a_qs_ptr0_1 = &a_ptr0->qs[k * 4 * blocklen + m * blocklen + qk / 2 * 4];
                            const int8_t * b_qs_ptr1 = &b_ptr1->qs[k * ncols_interleaved * blocklen + j * blocklen];
                            const int8_t * a_qs_ptr1_0 = &a_ptr1->qs[k * 4 * blocklen + m * blocklen];
                            const int8_t * a_qs_ptr1_1 = &a_ptr1->qs[k * 4 * blocklen + m * blocklen + qk / 2 * 4];

                            asm volatile (
                                "li t0, 8\n\t"
                                "th.vsetvli x0, t0, e8, m1\n\t"
                                "th.vle.v v8, (%[b0_ptr])\n\t"
                                "th.vle.v v12, (%[b1_ptr])\n\t"
                                "th.vle.v v10, (%[a00_ptr])\n\t"
                                "th.vle.v v11, (%[a01_ptr])\n\t"
                                "th.vand.vi v9, v8, 15\n\t"
                                "th.vsll.vi v9, v9, 4\n\t"
                                "th.vand.vi v8, v8, -16\n\t"
                                "th.vle.v v14, (%[a10_ptr])\n\t"
                                "th.vle.v v15, (%[a11_ptr])\n\t"
                                "th.vwmul.vv v16, v9, v10\n\t"
                                "th.vwmacc.vv v16, v8, v11\n\t"
                                "th.vand.vi v13, v12, 15\n\t"
                                "th.vsll.vi v13, v13, 4\n\t"
                                "th.vand.vi v12, v12, -16\n\t"
                                "th.vwmul.vv v20, v13, v14\n\t"
                                "th.vwmacc.vv v20, v12, v15\n\t"
                                "th.vsetvli x0, t0, e16, m2\n\t"
                                "th.vmv.s.x v24, x0\n\t"
                                "th.vwredsum.vs v24, v16, v24\n\t"
                                "th.vmv.s.x v28, x0\n\t"
                                "th.vwredsum.vs v28, v20, v28\n\t"
                                "th.vsetvli x0, t0, e32, m1\n\t"
                                "th.vmv.x.s %[psum0], v24\n\t"
                                "th.vmv.x.s %[psum1], v28\n\t"
                                : [psum0] "=r"(psum0), [psum1] "=r"(psum1)
                                : [b0_ptr] "r"(b_qs_ptr0), [a00_ptr] "r"(a_qs_ptr0_0), [a01_ptr] "r"(a_qs_ptr0_1),
                                  [b1_ptr] "r"(b_qs_ptr1), [a10_ptr] "r"(a_qs_ptr1_0), [a11_ptr] "r"(a_qs_ptr1_1)
                                : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", 
                                  "v16", "v17", "v20", "v21", "v24", "v28"
                            );
                            sumi0_block += psum0;
                            sumi1_block += psum1;
                        }
                        sumf[m][j] += (float)(sumi0_block >> 4) * GGML_FP16_TO_FP32(b_ptr0->d[j]) * GGML_FP16_TO_FP32(a_ptr0->d[m]);
                        sumf[m][j] += (float)(sumi1_block >> 4) * GGML_FP16_TO_FP32(b_ptr1->d[j]) * GGML_FP16_TO_FP32(a_ptr1->d[m]);
                    }
                    for (; l < nb; l++) {
                        ggml_gemm_q4_0_8x8_q8_0_rvv_tail_processing_asm(a_ptr_base + l, b_ptr_base + l, &sumf[m][j], m, j);
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    C[(y * 4 + m) * N + (x * ncols_interleaved + j)] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q4_0_8x8_q8_0_rvv_asm_unroll2_fused(
    int M, int N, int K,
    const void * A, const void * B, float * C
) {
    const int qk = QK8_0;
    const int nb = K / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    const int M_div_4 = M / 4;

    for (int y = 0; y < M_div_4; y++) {
        const block_q8_0x4 * a_ptr_base = (const block_q8_0x4 *) A + (y * nb);
        for (int x = 0; x < N / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr_base = (const block_q4_0x8 *) B + (x * nb);
            float sumf[4][8] = {{0.0f}};

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    int l = 0;
                    for (; l + 1 < nb; l += 2) {
                        const block_q8_0x4 * a_ptr0 = a_ptr_base + l;
                        const block_q8_0x4 * a_ptr1 = a_ptr_base + l + 1;
                        const block_q4_0x8 * b_ptr0 = b_ptr_base + l;
                        const block_q4_0x8 * b_ptr1 = b_ptr_base + l + 1;

                        int32_t sumi0_block = 0;
                        int32_t sumi1_block = 0;

                                                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                            int32_t psum0, psum1;
                            const int8_t * b_qs_ptr0 = &b_ptr0->qs[k * ncols_interleaved * blocklen + j * blocklen];
                            const int8_t * a_qs_ptr0_0 = &a_ptr0->qs[k * 4 * blocklen + m * blocklen];
                            const int8_t * a_qs_ptr0_1 = &a_ptr0->qs[k * 4 * blocklen + m * blocklen + qk / 2 * 4];
                            const int8_t * b_qs_ptr1 = &b_ptr1->qs[k * ncols_interleaved * blocklen + j * blocklen];
                            const int8_t * a_qs_ptr1_0 = &a_ptr1->qs[k * 4 * blocklen + m * blocklen];
                            const int8_t * a_qs_ptr1_1 = &a_ptr1->qs[k * 4 * blocklen + m * blocklen + qk / 2 * 4];

                            asm volatile (
                                "li t0, 8\n\t"
                                "th.vsetvli x0, t0, e8, m1\n\t"
                                "th.vle.v v9, (%[b0_ptr])\n\t"
                                "th.vand.vi v8, v9, 15\n\t"
                                "th.vsll.vi v8, v8, 4\n\t"
                                "th.vand.vi v9, v9, -16\n\t"
                                "th.vle.v v11, (%[b1_ptr])\n\t"
                                "th.vand.vi v10, v11, 15\n\t"
                                "th.vsll.vi v10, v10, 4\n\t"
                                "th.vand.vi v11, v11, -16\n\t"

                                "li t0, 32\n\t"
                                "th.vsetvli x0, t0, e8, m2\n\t"
                                "th.vle.v v12, (%[a0_ptr])\n\t"
                                "th.vle.v v14, (%[a1_ptr])\n\t"

                                "li t0, 64\n\t"
                                "th.vsetvli x0, t0, e8, m4\n\t"
                                "th.vwmul.vv v16, v8, v12\n\t"

                                "li t0, 32\n\t"
                                "th.vsetvli x0, t0, e16, m4\n\t"
                                "th.vmv.s.x v24, x0\n\t"
                                "th.vmv.s.x v28, x0\n\t"
                                "th.vwredsum.vs v24, v16, v24\n\t"
                                "th.vwredsum.vs v28, v20, v28\n\t"

                                "th.vsetvli x0, t0, e32, m1\n\t"
                                "th.vmv.x.s %[psum0], v24\n\t"
                                "th.vmv.x.s %[psum1], v28\n\t"
                                : [psum0] "=r"(psum0), [psum1] "=r"(psum1)
                                : [b0_ptr] "r"(b_qs_ptr0), [a0_ptr] "r"(a_qs_ptr0_0), 
                                  [b1_ptr] "r"(b_qs_ptr1), [a1_ptr] "r"(a_qs_ptr1_0)
                                : "t0", "memory", "v8", "v9", "v10", "v11", "v16", "v17", "v20", "v21", "v24", "v28"
                            );
                            sumi0_block += psum0;
                            sumi1_block += psum1;
                        }
                        sumf[m][j] += (float)(sumi0_block >> 4) * GGML_FP16_TO_FP32(b_ptr0->d[j]) * GGML_FP16_TO_FP32(a_ptr0->d[m]);
                        sumf[m][j] += (float)(sumi1_block >> 4) * GGML_FP16_TO_FP32(b_ptr1->d[j]) * GGML_FP16_TO_FP32(a_ptr1->d[m]);
                    }
                    for (; l < nb; l++) {
                        ggml_gemm_q4_0_8x8_q8_0_rvv_tail_processing_asm(a_ptr_base + l, b_ptr_base + l, &sumf[m][j], m, j);
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    C[(y * 4 + m) * N + (x * ncols_interleaved + j)] = sumf[m][j];
                }
            }
        }
    }
}
void ggml_gemm_q4_0_8x8_q8_0_asm_auto_generated(int M, int N, int K, const void * A, const void * B, float * C) {
    const int qk = QK8_0;
    const int nb = K / qk;
    const int blocklen = 8;
    const int M_div_4 = M / 4;
    const int N_div_8 = N / 8;

    for (int y = 0; y < M_div_4; y++) {
        const block_q8_0x4 * a_ptr_base = (const block_q8_0x4 *) A + (y * nb);
        for (int x = 0; x < N_div_8; x++) {
            const block_q4_0x8 * b_ptr_base = (const block_q4_0x8 *) B + (x * nb);
float sumf[4][8] = {{0.0f}};
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    int l = 0;
                    // --- Main loop over K blocks (unrolled) ---
                    for (; l + 1 < nb; l += 2) {
                        const block_q8_0x4 * a_ptr[2];
                        const block_q4_0x8 * b_ptr[2];
                        a_ptr[0] = a_ptr_base + l + 0;
                        b_ptr[0] = b_ptr_base + l + 0;
                        a_ptr[1] = a_ptr_base + l + 1;
                        b_ptr[1] = b_ptr_base + l + 1;
                        
                        int32_t sumi_block[2] = {0};
                        
                        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                            int32_t psum[2];
                            const int8_t * b_qs_ptr0 = &b_ptr[0]->qs[k * 8 * blocklen + j * blocklen];
                            const int8_t * a_qs_ptr0_0 = &a_ptr[0]->qs[m * qk + k * blocklen];
                            const int8_t * a_qs_ptr0_1 = &a_ptr[0]->qs[m * qk + k * blocklen + qk/2];
                            const int8_t * b_qs_ptr1 = &b_ptr[1]->qs[k * 8 * blocklen + j * blocklen];
                            const int8_t * a_qs_ptr1_0 = &a_ptr[1]->qs[m * qk + k * blocklen];
                            const int8_t * a_qs_ptr1_1 = &a_ptr[1]->qs[m * qk + k * blocklen + qk/2];

                            asm volatile (
                                "li t0, 8\n\t" "th.vsetvli x0, t0, e8, m1\n\t"
                                // --- Block 0 ---
                                "th.vle.v v8, (%[b0_ptr])\n\t"
                                "th.vle.v v10, (%[a00_ptr])\n\t"
                                "th.vle.v v11, (%[a01_ptr])\n\t"
                                "th.vand.vi v9, v8, 15\n\t"
                                "th.vsll.vi v9, v9, 4\n\t"
                                "th.vand.vi v8, v8, -16\n\t"
                                "th.vwmul.vv v16, v9, v10\n\t"
                                "th.vwmacc.vv v16, v8, v11\n\t"
                                // --- Block 1 ---
                                "th.vle.v v8, (%[b1_ptr])\n\t"
                                "th.vle.v v10, (%[a10_ptr])\n\t"
                                "th.vle.v v11, (%[a11_ptr])\n\t"
                                "th.vand.vi v9, v8, 15\n\t"
                                "th.vsll.vi v9, v9, 4\n\t"
                                "th.vand.vi v8, v8, -16\n\t"
                                "th.vwmul.vv v20, v9, v10\n\t"
                                "th.vwmacc.vv v20, v8, v11\n\t"

                                "th.vsetvli x0, t0, e16, m2\n\t"
                                "th.vmv.s.x v24, x0\n\t"
                                "th.vwredsum.vs v24, v16, v24\n\t"
                                "th.vmv.s.x v28, x0\n\t"
                                "th.vwredsum.vs v28, v20, v28\n\t"
                                "th.vsetvli x0, t0, e32, m1\n\t"
                                "th.vmv.x.s %[psum0], v24\n\t"
                                "th.vmv.x.s %[psum1], v28\n\t"
                                :  [psum0] "=r"(psum[0]),  [psum1] "=r"(psum[1])                                 :                                   [b0_ptr] "r"(b_qs_ptr0), 
                                  [a00_ptr] "r"(a_qs_ptr0_0), 
                                  [a01_ptr] "r"(a_qs_ptr0_1),                                  [b1_ptr] "r"(b_qs_ptr1), 
                                  [a10_ptr] "r"(a_qs_ptr1_0), 
                                  [a11_ptr] "r"(a_qs_ptr1_1)                                : "t0", "memory", "v8", "v9", "v10", "v11", "v16", "v17", "v20", "v21", "v24", "v28"                            );
sumi_block[0] += psum[0];sumi_block[1] += psum[1];                        }
                        sumf[m][j] += (float)(sumi_block[0] >> 4) * GGML_FP16_TO_FP32(b_ptr[0]->d[j]) * GGML_FP16_TO_FP32(a_ptr[0]->d[m]);
                        sumf[m][j] += (float)(sumi_block[1] >> 4) * GGML_FP16_TO_FP32(b_ptr[1]->d[j]) * GGML_FP16_TO_FP32(a_ptr[1]->d[m]);
                    }
                    // --- Tail loop over K blocks ---
                    for (; l < nb; l++) {
                        ggml_gemm_q4_0_8x8_q8_0_rvv_tail_processing_asm(a_ptr_base + l, b_ptr_base + l, &sumf[m][j], m, j);
                    }
                }
            }
            // --- Store results ---
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    C[(y * 4 + m) * N + (x * 8 + j)] = sumf[m][j];
                }
            }
        }
    }
}
#endif
