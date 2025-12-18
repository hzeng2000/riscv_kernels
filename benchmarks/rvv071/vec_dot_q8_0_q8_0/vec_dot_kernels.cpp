#include "vec_dot_kernels.h"
#include "../common/common_defs.h" 
#include <cassert>

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
#include <riscv_vector.h>
#endif

// 1. Scalar (no change)
void ggml_vec_dot_q8_0_q8_0_scalar(int n, float *s, const void *vx, const void *vy) {
    const int nb = n / QK8_0;
    const auto *x = static_cast<const block_q8_0 *>(vx);
    const auto *y = static_cast<const block_q8_0 *>(vy);
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        int32_t sumi = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }
        sumf += (float)sumi * (GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
    }
    *s = sumf;
}


#if defined(__riscv_v) || defined(__riscv_xtheadvector)
// 2. RVV Intrinsics (no change)
void ggml_vec_dot_q8_0_q8_0_rvv_intrinsics(int n, float *s, const void *vx, const void *vy) {
    const int nb = n / QK8_0;
    const block_q8_0 *GGML_RESTRICT x = (const block_q8_0 *)vx;
    const block_q8_0 *GGML_RESTRICT y = (const block_q8_0 *)vy;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        size_t vl = QK8_0;
        vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[ib].qs, vl);
        vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
        vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);
        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);
        int32_t sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);
        sumf += (float)sumi * (GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
    }
    *s = sumf;
}
#endif


#if defined(__RVV_ASM_XTHEAD)
// Reusable ASM helper for tail loops
static inline void process_single_block_asm(const block_q8_0* x, const block_q8_0* y, float& sumf) {
    int32_t sumi;
    asm volatile(
        "li t0, 32\n\t"
        "th.vsetvli x0, t0, e8,m2\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"
        "th.vle.v v10, (%[y_ptr])\n\t"
        "th.vwmul.vv v12, v8, v10\n\t"
        "th.vsetvli x0, t0, e16,m4\n\t"
        "th.vmv.s.x v24, x0\n\t"
        "th.vwredsum.vs v24, v12, v24\n\t"
        "th.vsetvli x0, t0, e32,m1\n\t"
        "th.vmv.x.s %[sumi], v24\n\t"
        : [sumi] "=r"(sumi)
        : [x_ptr] "r"(x->qs), [y_ptr] "r"(y->qs)
        : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v24"
    );
    sumf += (float)sumi * GGML_FP16_TO_FP32(x->d) * GGML_FP16_TO_FP32(y->d);
}

// 3. ILP Axis: Unroll=2 (Serial)
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll1(int n, float *s, const void *vx, const void *vy) {
    const int nb = n / QK8_0;
    const auto *x = static_cast<const block_q8_0 *>(vx);
    const auto *y = static_cast<const block_q8_0 *>(vy);
    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        int32_t sumi;
        asm volatile(
            "li t0, 32\n\t"
            "th.vsetvli x0, t0, e8,m2\n\t"
            "th.vle.v v8, (%[x_ptr])\n\t"
            "th.vle.v v10, (%[y_ptr])\n\t"
            "th.vwmul.vv v12, v8, v10\n\t"
            "th.vsetvli x0, t0, e16,m4\n\t"
            "th.vmv.s.x v24, x0\n\t"
            "th.vwredsum.vs v24, v12, v24\n\t"
            "th.vsetvli x0, t0, e32,m1\n\t"
            "th.vmv.x.s %[sumi], v24\n\t"
            : [sumi] "=r"(sumi)
            : [x_ptr] "r"(x[ib].qs), [y_ptr] "r"(y[ib].qs)
            : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v24"
        );
        sumf += (float)sumi * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
    }
    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll2(int n, float *s, const void *vx, const void *vy) {
    const int nb = n / QK8_0;
    const auto *x = static_cast<const block_q8_0 *>(vx);
    const auto *y = static_cast<const block_q8_0 *>(vy);
    float sumf = 0.0f;
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        int32_t sumi[2];
        asm volatile(
            // Set VL and load data for two blocks
            "li t0, 32\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vle.v v8, (%[x0_ptr])\n\t"
            "th.vle.v v10, (%[y0_ptr])\n\t"
            "th.vle.v v12, (%[x1_ptr])\n\t"
            "th.vle.v v14, (%[y1_ptr])\n\t"
            // Block 0 computation
            "th.vwmul.vv v16, v8, v10\n\t"   // v16(m4) = v8(m2) * v10(m2)
            // Block 1 computation
            "th.vwmul.vv v20, v12, v14\n\t"  // v20(m4) = v12(m2) * v14(m2)
            // Reduction
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v24, x0\n\t"         // zero accumulator v24
            "th.vmv.s.x v28, x0\n\t"         // zero accumulator v28
            "th.vwredsum.vs v24, v16, v24\n\t"
            "th.vwredsum.vs v28, v20, v28\n\t"
            // Store results
            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.x.s %[sumi0], v24\n\t"
            "th.vmv.x.s %[sumi1], v28\n\t"
            : [sumi0] "=r"(sumi[0]), [sumi1] "=r"(sumi[1])
            : [x0_ptr] "r"(x[ib].qs),   [y0_ptr] "r"(y[ib].qs),
              [x1_ptr] "r"(x[ib+1].qs), [y1_ptr] "r"(y[ib+1].qs)
            : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v28"
        );
        sumf += (float)sumi[0] * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
        sumf += (float)sumi[1] * GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib+1].d);
    }
    for (; ib < nb; ++ib) { process_single_block_asm(&x[ib], &y[ib], sumf); }
    *s = sumf;
}

// 4. ILP Axis: Unroll=2 (Fused - "大乘法")
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll2_fused(int n, float *s, const void *vx, const void *vy) {
    const int nb = n / QK8_0;
    const auto *x = static_cast<const block_q8_0 *>(vx);
    const auto *y = static_cast<const block_q8_0 *>(vy);
    float sumf = 0.0f;
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        int32_t sumi[2];
        asm volatile(
            // Load vectors for two blocks using LMUL=2
            "li t0, 32\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vle.v v8, (%[x0_ptr])\n\t" // v8,v9  <- x[ib]
            "th.vle.v v10, (%[x1_ptr])\n\t"// v10,v11 <- x[ib+1]
            "th.vle.v v12, (%[y0_ptr])\n\t"// v12,v13 <- y[ib]
            "th.vle.v v14, (%[y1_ptr])\n\t"// v14,v15 <- y[ib+1]
            // "Fuse" into a single wider multiply using LMUL=4
            "li t0, 64\n\t"
            "th.vsetvli x0, t0, e8, m4\n\t"
            "th.vwmul.vv v16, v8, v12\n\t" // v16-v23(m8) = {v8-v11} * {v12-v15}
            // Reduce the two halves of the m8 result vector
            "li t0, 32\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v24, x0\n\t"
            "th.vmv.s.x v28, x0\n\t"
            "th.vwredsum.vs v24, v16, v24\n\t" // Reduce first half {v16-v19}
            "th.vwredsum.vs v28, v20, v28\n\t" // Reduce second half {v20-v23}
            // Store results
            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.x.s %[sumi0], v24\n\t"
            "th.vmv.x.s %[sumi1], v28\n\t"
            : [sumi0] "=r"(sumi[0]), [sumi1] "=r"(sumi[1])
            : [x0_ptr] "r"(x[ib].qs),   [y0_ptr] "r"(y[ib].qs),
              [x1_ptr] "r"(x[ib+1].qs), [y1_ptr] "r"(y[ib+1].qs)
            : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v28"
        );
        sumf += (float)sumi[0] * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
        sumf += (float)sumi[1] * GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib+1].d);
    }
    for (; ib < nb; ++ib) { process_single_block_asm(&x[ib], &y[ib], sumf); }
    *s = sumf;
}


// 5. ILP Axis: Unroll=4 (User's original version)
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll4(int n, float * s, const void * vx, const void * vy) {
    const int nb = n / QK8_0;
    const auto *x = static_cast<const block_q8_0 *>(vx);
    const auto *y = static_cast<const block_q8_0 *>(vy);
    float sumf = 0.0f;
    int ib = 0;
    for (; ib + 3 < nb; ib += 4) {
        int32_t sumi[4];
        asm volatile(
            "li t0, 32\n\t"
            // Set VL for loads
            "th.vsetvli x0, t0, e8, m2\n\t"
            // Load all data first
            "th.vle.v v8,  (%[x0_ptr])\n\t"
            "th.vle.v v10, (%[x1_ptr])\n\t"
            "th.vle.v v12, (%[x2_ptr])\n\t"
            "th.vle.v v14, (%[x3_ptr])\n\t"
            "th.vle.v v16, (%[y0_ptr])\n\t"
            "th.vle.v v18, (%[y1_ptr])\n\t"
            "th.vle.v v20, (%[y2_ptr])\n\t"
            "th.vle.v v22, (%[y3_ptr])\n\t"

            // Perform Mul->Reduce for each block sequentially
            // Block 0
            "th.vwmul.vv v24, v8, v16\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v28, x0\n\t"
            "th.vwredsum.vs v28, v24, v28\n\t"

            // Block 1
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v10, v18\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v29, x0\n\t"
            "th.vwredsum.vs v29, v24, v29\n\t"

            // Block 2
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v12, v20\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v30, x0\n\t"
            "th.vwredsum.vs v30, v24, v30\n\t"

            // Block 3
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v14, v22\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v31, x0\n\t"
            "th.vwredsum.vs v31, v24, v31\n\t"

            // Store All
            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.x.s %[sumi0], v28\n\t"
            "th.vmv.x.s %[sumi1], v29\n\t"
            "th.vmv.x.s %[sumi2], v30\n\t"
            "th.vmv.x.s %[sumi3], v31\n\t"
            : [sumi0] "=r"(sumi[0]), [sumi1] "=r"(sumi[1]), [sumi2] "=r"(sumi[2]), [sumi3] "=r"(sumi[3])
            : [x0_ptr] "r"(x[ib].qs), [y0_ptr] "r"(y[ib].qs), [x1_ptr] "r"(x[ib+1].qs), [y1_ptr] "r"(y[ib+1].qs),
              [x2_ptr] "r"(x[ib+2].qs), [y2_ptr] "r"(y[ib+2].qs), [x3_ptr] "r"(x[ib+3].qs), [y3_ptr] "r"(y[ib+3].qs)
            : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf += (float)sumi[0] * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
        sumf += (float)sumi[1] * GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib+1].d);
        sumf += (float)sumi[2] * GGML_FP16_TO_FP32(x[ib+2].d) * GGML_FP16_TO_FP32(y[ib+2].d);
        sumf += (float)sumi[3] * GGML_FP16_TO_FP32(x[ib+3].d) * GGML_FP16_TO_FP32(y[ib+3].d);
    }
    for (; ib < nb; ++ib) { process_single_block_asm(&x[ib], &y[ib], sumf); }
    *s = sumf;
}

// 6. Memory & ILP Axis: Unroll=4 (Pipelined - Corrected)
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll4_pipelined(int n, float *s, const void *vx, const void *vy) {
    const int nb = n / QK8_0;
    const auto *x = static_cast<const block_q8_0 *>(vx);
    const auto *y = static_cast<const block_q8_0 *>(vy);
    float sumf = 0.0f;
    int ib = 0;
    for (; ib + 3 < nb; ib += 4) {
        int32_t sumi[4];
        asm volatile(
            "li t0, 32\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            // --- Software Pipeline Stage 1: Load a block ---
            "th.vle.v v8, (%[x0_ptr])\n\t"
            "th.vle.v v16, (%[y0_ptr])\n\t"
            // --- Stage 2: Load next, Compute previous ---
            "th.vle.v v10, (%[x1_ptr])\n\t"
            "th.vle.v v18, (%[y1_ptr])\n\t"
            "th.vwmul.vv v24, v8, v16\n\t"
            // --- Stage 3: Load next, Compute previous, Reduce oldest ---
            "th.vle.v v12, (%[x2_ptr])\n\t"
            "th.vle.v v20, (%[y2_ptr])\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v28, x0\n\t"
            "th.vwredsum.vs v28, v24, v28\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v10, v18\n\t"
            // --- Stage 4: Load last, Compute previous, Reduce oldest ---
            "th.vle.v v14, (%[x3_ptr])\n\t"
            "th.vle.v v22, (%[y3_ptr])\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v29, x0\n\t"
            "th.vwredsum.vs v29, v24, v29\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v12, v20\n\t"
            // --- Pipeline Drain: Compute and Reduce remaining blocks ---
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v30, x0\n\t"
            "th.vwredsum.vs v30, v24, v30\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v14, v22\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v31, x0\n\t"
            "th.vwredsum.vs v31, v24, v31\n\t"
            // Store All Results
            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.x.s %[sumi0], v28\n\t"
            "th.vmv.x.s %[sumi1], v29\n\t"
            "th.vmv.x.s %[sumi2], v30\n\t"
            "th.vmv.x.s %[sumi3], v31\n\t"
            : [sumi0] "=r"(sumi[0]), [sumi1] "=r"(sumi[1]), [sumi2] "=r"(sumi[2]), [sumi3] "=r"(sumi[3])
            : [x0_ptr] "r"(x[ib].qs), [y0_ptr] "r"(y[ib].qs), [x1_ptr] "r"(x[ib+1].qs), [y1_ptr] "r"(y[ib+1].qs),
              [x2_ptr] "r"(x[ib+2].qs), [y2_ptr] "r"(y[ib+2].qs), [x3_ptr] "r"(x[ib+3].qs), [y3_ptr] "r"(y[ib+3].qs)
            : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf += (float)sumi[0] * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
        sumf += (float)sumi[1] * GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib+1].d);
        sumf += (float)sumi[2] * GGML_FP16_TO_FP32(x[ib+2].d) * GGML_FP16_TO_FP32(y[ib+2].d);
        sumf += (float)sumi[3] * GGML_FP16_TO_FP32(x[ib+3].d) * GGML_FP16_TO_FP32(y[ib+3].d);
    }
    for (; ib < nb; ++ib) { process_single_block_asm(&x[ib], &y[ib], sumf); }
    *s = sumf;
}
// 7. Computation Axis: FP64 Accumulator (Corrected Assembly)
void ggml_vec_dot_q8_0_q8_0_rvv_asm_fp64_accum(int n, float *s, const void *vx, const void *vy) {
    const int nb = n / QK8_0;
    const auto *x = static_cast<const block_q8_0 *>(vx);
    const auto *y = static_cast<const block_q8_0 *>(vy);
    double sum_d = 0.0; // Use double for higher precision accumulation
    int ib = 0;

    // Use the CORRECTED pipelined unroll-4 logic for the main loop
    for (; ib + 3 < nb; ib += 4) {
        int32_t sumi[4];
        // THIS ASM BLOCK IS NOW FIXED
        asm volatile(
            "li t0, 32\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            // --- Software Pipeline Stage 1: Load a block ---
            "th.vle.v v8, (%[x0_ptr])\n\t"
            "th.vle.v v16, (%[y0_ptr])\n\t"
            // --- Stage 2: Load next, Compute previous ---
            "th.vle.v v10, (%[x1_ptr])\n\t"
            "th.vle.v v18, (%[y1_ptr])\n\t"
            "th.vwmul.vv v24, v8, v16\n\t"
            // --- Stage 3: Load next, Compute previous, Reduce oldest ---
            "th.vle.v v12, (%[x2_ptr])\n\t"
            "th.vle.v v20, (%[y2_ptr])\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v28, x0\n\t"
            "th.vwredsum.vs v28, v24, v28\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v10, v18\n\t"
            // --- Stage 4: Load last, Compute previous, Reduce oldest ---
            "th.vle.v v14, (%[x3_ptr])\n\t"
            "th.vle.v v22, (%[y3_ptr])\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v29, x0\n\t"
            "th.vwredsum.vs v29, v24, v29\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v12, v20\n\t"
            // --- Pipeline Drain: Compute and Reduce remaining blocks ---
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v30, x0\n\t"
            "th.vwredsum.vs v30, v24, v30\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v14, v22\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v31, x0\n\t"
            "th.vwredsum.vs v31, v24, v31\n\t"
            // Store All Results
            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.x.s %[sumi0], v28\n\t"
            "th.vmv.x.s %[sumi1], v29\n\t"
            "th.vmv.x.s %[sumi2], v30\n\t"
            "th.vmv.x.s %[sumi3], v31\n\t"
            : [sumi0] "=r"(sumi[0]), [sumi1] "=r"(sumi[1]), [sumi2] "=r"(sumi[2]), [sumi3] "=r"(sumi[3])
            : [x0_ptr] "r"(x[ib].qs), [y0_ptr] "r"(y[ib].qs), [x1_ptr] "r"(x[ib+1].qs), [y1_ptr] "r"(y[ib+1].qs),
              [x2_ptr] "r"(x[ib+2].qs), [y2_ptr] "r"(y[ib+2].qs), [x3_ptr] "r"(x[ib+3].qs), [y3_ptr] "r"(y[ib+3].qs)
            : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sum_d += (double)sumi[0] * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
        sum_d += (double)sumi[1] * GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib+1].d);
        sum_d += (double)sumi[2] * GGML_FP16_TO_FP32(x[ib+2].d) * GGML_FP16_TO_FP32(y[ib+2].d);
        sum_d += (double)sumi[3] * GGML_FP16_TO_FP32(x[ib+3].d) * GGML_FP16_TO_FP32(y[ib+3].d);
    }
    
    // Tail loop must also use the double accumulator
    float temp_sumf = 0;
    // We need a local float accumulator for the helper function
    for (; ib < nb; ++ib) {
        process_single_block_asm(&x[ib], &y[ib], temp_sumf);
    }
    sum_d += temp_sumf;

    *s = (float)sum_d;
}

// 8. Accumulation Axis: Blocked Accumulation (Corrected Assembly)
void ggml_vec_dot_q8_0_q8_0_rvv_asm_blocked_accum(int n, float *s, const void *vx, const void *vy) {
    const int nb = n / QK8_0;
    const auto *x = static_cast<const block_q8_0 *>(vx);
    const auto *y = static_cast<const block_q8_0 *>(vy);

    const int ACCUM_BLOCKS = 4;
    size_t vl_accum = ACCUM_BLOCKS;
    vfloat32m1_t v_sum_lanes = __riscv_vfmv_v_f_f32m1(0.0f, vl_accum);

    int ib = 0;
    for (; ib + (ACCUM_BLOCKS - 1) < nb; ib += ACCUM_BLOCKS) {
        int32_t sumi[ACCUM_BLOCKS];

        // 使用已验证过的、正确的 unroll=4 汇编代码块
        asm volatile(
            "li t0, 32\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vle.v v8,  (%[x0_ptr])\n\t"
            "th.vle.v v10, (%[x1_ptr])\n\t"
            "th.vle.v v12, (%[x2_ptr])\n\t"
            "th.vle.v v14, (%[x3_ptr])\n\t"
            "th.vle.v v16, (%[y0_ptr])\n\t"
            "th.vle.v v18, (%[y1_ptr])\n\t"
            "th.vle.v v20, (%[y2_ptr])\n\t"
            "th.vle.v v22, (%[y3_ptr])\n\t"

            // Block 0
            "th.vwmul.vv v24, v8, v16\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v28, x0\n\t"
            "th.vwredsum.vs v28, v24, v28\n\t"

            // Block 1
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v10, v18\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v29, x0\n\t"
            "th.vwredsum.vs v29, v24, v29\n\t"

            // Block 2
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v12, v20\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v30, x0\n\t"
            "th.vwredsum.vs v30, v24, v30\n\t"

            // Block 3
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v14, v22\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v31, x0\n\t"
            "th.vwredsum.vs v31, v24, v31\n\t"

            // Store All
            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.x.s %[sumi0], v28\n\t"
            "th.vmv.x.s %[sumi1], v29\n\t"
            "th.vmv.x.s %[sumi2], v30\n\t"
            "th.vmv.x.s %[sumi3], v31\n\t"
            : [sumi0] "=r"(sumi[0]), [sumi1] "=r"(sumi[1]), [sumi2] "=r"(sumi[2]), [sumi3] "=r"(sumi[3])
            : [x0_ptr] "r"(x[ib].qs), [y0_ptr] "r"(y[ib].qs), [x1_ptr] "r"(x[ib+1].qs), [y1_ptr] "r"(y[ib+1].qs),
              [x2_ptr] "r"(x[ib+2].qs), [y2_ptr] "r"(y[ib+2].qs), [x3_ptr] "r"(x[ib+3].qs), [y3_ptr] "r"(y[ib+3].qs)
            : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31"
        );

        float partial_sums_fp[ACCUM_BLOCKS];
        for (int i = 0; i < ACCUM_BLOCKS; ++i) {
            float d_x = GGML_FP16_TO_FP32(x[ib + i].d);
            float d_y = GGML_FP16_TO_FP32(y[ib + i].d);
            partial_sums_fp[i] = (float)sumi[i] * d_x * d_y;
        }

        vfloat32m1_t v_partials = __riscv_vle32_v_f32m1(partial_sums_fp, vl_accum);
        v_sum_lanes = __riscv_vfadd_vv_f32m1(v_sum_lanes, v_partials, vl_accum);
    }

    // --- 最终规约 ---
    vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl_accum);
    vfloat32m1_t v_final_scalar_sum = __riscv_vfredusum_vs_f32m1_f32m1(v_sum_lanes, v_zero, vl_accum);
    float sumf = __riscv_vfmv_f_s_f32m1_f32(v_final_scalar_sum);
    
    // 处理剩余的尾部块
    for (; ib < nb; ++ib) {
        int32_t sumi;
        asm volatile(
            "li t0, 32\n\t"
            "th.vsetvli x0, t0, e8,m2\n\t"
            "th.vle.v v8, (%[x_ptr])\n\t" "th.vle.v v10, (%[y_ptr])\n\t"
            "th.vwmul.vv v12, v8, v10\n\t"
            "th.vsetvli x0, t0, e16,m4\n\t"
            "th.vmv.s.x v24, x0\n\t" "th.vwredsum.vs v24, v12, v24\n\t"
            "th.vsetvli x0, t0, e32,m1\n\t" "th.vmv.x.s %[sumi], v24\n\t"
            : [sumi] "=r"(sumi)
            : [x_ptr] "r"(x[ib].qs), [y_ptr] "r"(y[ib].qs)
            : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v24"
        );
        sumf += (float)sumi * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
    }
    
    *s = sumf;
}

// 9. Memory Axis: Unroll=4 with Prefetching (Corrected)
void ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll4_prefetch(int n, float *s, const void *vx, const void *vy) {
    const int nb = n / QK8_0;
    const auto *x = static_cast<const block_q8_0 *>(vx);
    const auto *y = static_cast<const block_q8_0 *>(vy);
    float sumf = 0.0f;
    int ib = 0;

    // 主循环，每次处理4个块，但需要预取未来的2个块，所以总共需要6个块
    for (; ib + 5 < nb; ib += 4) {
        int32_t sumi[4];
        asm volatile(
            "li t0, 32\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"

            // --- 阶段1: 加载计算数据 & 预取未来数据 ---
            // 加载当前计算所需的前2个块
            "th.vle.v v8,  (%[x0_ptr])\n\t"
            "th.vle.v v16, (%[y0_ptr])\n\t"
            "th.vle.v v10, (%[x1_ptr])\n\t"
            "th.vle.v v18, (%[y1_ptr])\n\t"
            
            // **预取**未来第 ib+4 和 ib+5 块的数据到空闲寄存器
            "th.vle.v v0, (%[x4_ptr])\n\t"
            "th.vle.v v2, (%[y4_ptr])\n\t"
            "th.vle.v v4, (%[x5_ptr])\n\t"
            "th.vle.v v6, (%[y5_ptr])\n\t"

            // 加载当前计算所需的后2个块
            "th.vle.v v12, (%[x2_ptr])\n\t"
            "th.vle.v v20, (%[y2_ptr])\n\t"
            "th.vle.v v14, (%[x3_ptr])\n\t"
            "th.vle.v v22, (%[y3_ptr])\n\t"

            // --- 阶段2: 串行执行4次“乘法-规约” ---
            // 计算第0块
            "th.vwmul.vv v24, v8, v16\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v28, x0\n\t"
            "th.vwredsum.vs v28, v24, v28\n\t"

            // 计算第1块
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v10, v18\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v29, x0\n\t"
            "th.vwredsum.vs v29, v24, v29\n\t"

            // 计算第2块
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v12, v20\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v30, x0\n\t"
            "th.vwredsum.vs v30, v24, v30\n\t"
            
            // 计算第3块
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vwmul.vv v24, v14, v22\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v31, x0\n\t"
            "th.vwredsum.vs v31, v24, v31\n\t"

            // --- 阶段3: 存储结果 ---
            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.x.s %[sumi0], v28\n\t"
            "th.vmv.x.s %[sumi1], v29\n\t"
            "th.vmv.x.s %[sumi2], v30\n\t"
            "th.vmv.x.s %[sumi3], v31\n\t"

            : [sumi0] "=r"(sumi[0]), [sumi1] "=r"(sumi[1]), [sumi2] "=r"(sumi[2]), [sumi3] "=r"(sumi[3])
            : [x0_ptr] "r"(x[ib].qs),     [y0_ptr] "r"(y[ib].qs),
              [x1_ptr] "r"(x[ib+1].qs), [y1_ptr] "r"(y[ib+1].qs),
              [x2_ptr] "r"(x[ib+2].qs), [y2_ptr] "r"(y[ib+2].qs),
              [x3_ptr] "r"(x[ib+3].qs), [y3_ptr] "r"(y[ib+3].qs),
              // 预取指针
              [x4_ptr] "r"(x[ib+4].qs), [y4_ptr] "r"(y[ib+4].qs),
              [x5_ptr] "r"(x[ib+5].qs), [y5_ptr] "r"(y[ib+5].qs)
            : "t0", "memory", 
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", // Clobbered prefetch registers
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf += (float)sumi[0] * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
        sumf += (float)sumi[1] * GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib+1].d);
        sumf += (float)sumi[2] * GGML_FP16_TO_FP32(x[ib+2].d) * GGML_FP16_TO_FP32(y[ib+2].d);
        sumf += (float)sumi[3] * GGML_FP16_TO_FP32(x[ib+3].d) * GGML_FP16_TO_FP32(y[ib+3].d);
    }

    // 常规的尾部循环处理所有剩余的块
    for (; ib < nb; ++ib) {
        process_single_block_asm(&x[ib], &y[ib], sumf);
    }
    
    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0_rvv_asm_auto_generated(int n, float *s, const void *vx, const void *vy) {
    const int nb = n / QK8_0;
    const auto *x = static_cast<const block_q8_0 *>(vx);
    const auto *y = static_cast<const block_q8_0 *>(vy);
    
    float sumf = 0.0f;
    int ib = 0;

 
    for (; ib + 1 < nb; ib += 2) {
        int32_t sumi[2];
        asm volatile(
            "li t0, 32\n\t"
            /******************* CODE BLOCK START *******************/


            // 1. Load all data for current compute window
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vle.v v8, (%[x0_ptr])\n\t"
            "th.vle.v v10, (%[y0_ptr])\n\t"
            "th.vle.v v12, (%[x1_ptr])\n\t"
            "th.vle.v v14, (%[y1_ptr])\n\t"
            // 2. Perform all multiplications
            "th.vwmul.vv v16, v8, v10\n\t"
            "th.vwmul.vv v16, v12, v14\n\t"
            // 3. Perform all reductions
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vmv.s.x v20, x0\n\t"
            "th.vwredsum.vs v20, v16, v20\n\t"
            "th.vmv.s.x v21, x0\n\t"
            "th.vwredsum.vs v21, v16, v21\n\t"

            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.x.s %[sumi0], v20\n\t"
            "th.vmv.x.s %[sumi1], v21\n\t"
            /******************** CODE BLOCK END ********************/
            :  [sumi0] "=r"(sumi[0]),  [sumi1] "=r"(sumi[1])             :  [x0_ptr] "r"(x[ib+0].qs), [y0_ptr] "r"(y[ib+0].qs),  [x1_ptr] "r"(x[ib+1].qs), [y1_ptr] "r"(y[ib+1].qs)             :               "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21");
        
        sumf += (float)sumi[0] * GGML_FP16_TO_FP32(x[ib+0].d) * GGML_FP16_TO_FP32(y[ib+0].d);
        sumf += (float)sumi[1] * GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib+1].d);
    }

    // Tail loop
    
    for (; ib < nb; ++ib) {
        process_single_block_asm(&x[ib], &y[ib], sumf);
    }
    *s = sumf;
    
}
#endif