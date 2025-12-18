#include "quantize_kernels.h"
#include <cmath>
#include <cassert>
#include <algorithm> // for std::max
#include <cstdio>

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
#include <riscv_vector.h>
#endif

#if !defined(MAX)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// 1. Reference C++ implementation (unchanged)
void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int64_t nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);
        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;
            y[i].qs[j] = roundf(x0);
        }
    }
}

#if defined(__riscv_v) || defined(__riscv_xtheadvector)

// 2. RVV Intrinsics version (your correct baseline, unchanged)
void quantize_row_q8_0_rvv(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    size_t vl = QK8_0;
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        vfloat32m8_t v_x   = __riscv_vle32_v_f32m8(x+i*QK8_0, vl);
        vfloat32m8_t vfabs = __riscv_vfabs_v_f32m8(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m8_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d = amax / 127.0f;
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);

        vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);
        vint16m4_t   vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
        vint8m2_t    vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);
        __riscv_vse8_v_i8m2(y[i].qs , vs, vl);
    }
}

void quantize_row_q8_0_rvv_asm_unroll1(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int64_t nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        const float *x_ptr = x + i * QK8_0;
        int8_t *y_qs_ptr = y[i].qs;
        float amax;
        float d, id;

        // --- Part 1: Calculate amax using RVV ---
        // This block is self-contained and correct.
        // The key is "th.vmv.v.i v1, 0" which resets the reduction accumulator in each iteration.
        asm volatile (
            "li t0, 32\n\t"
            // Set vtype for reduction accumulator initialization
            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.v.i v1, 0\n\t"

            // Set vtype for main computation
            "th.vsetvli x0, t0, e32, m8\n\t"
            "th.vle.v v8, (%[x_ptr])\n\t"   // Load float32 data
            "th.vfabs.v v16, v8\n\t"       // Absolute values

            // Reduce to find the maximum value
            "th.vfredmax.vs v16, v16, v1\n\t"

            // Move the scalar result to the 'amax' C variable
            "th.vfmv.f.s %[amax], v16\n\t"
            : [amax] "=f" (amax)
            : [x_ptr] "r" (x_ptr)
            : "t0", "memory", "v1",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
        );

        // --- C++ part: Calculate scaling factor ---
        d = amax / 127.0f;
        id = d ? 1.0f / d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);

        // --- Part 2: Scale, convert, and store using RVV ---
        // This block is now clean, without intermediate arrays.
        // The data flow happens entirely within vector registers.
        asm volatile(
            "li t0, 32\n\t"
            // Set vtype for float operations
            "th.vsetvli x0, t0, e32, m8\n\t"
            // CRITICAL: Reload the data. Do not assume v8 from the previous block is valid.
            "th.vle.v v8, (%[x_ptr])\n\t"
            // Scale the floats by 'id'
            "th.vfmul.vf v8, v8, %[id]\n\t"

            // Convert f32 to i16 (with truncation).
            // Note: v8 holds m8e32, after conversion v4 will hold m4e16.
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vfncvt.x.f.v v4, v8\n\t"

            // Narrow i16 to i8.
            // Note: v4 holds m4e16, after conversion v2 will hold m2e8.
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vnsrl.vi v2, v4, 0\n\t"

            // Store the final int8 results directly to the destination.
            "th.vse.v v2, (%[y_ptr])\n\t"
            : // No C variable outputs from this block
            : [x_ptr] "r" (x_ptr), [y_ptr] "r" (y_qs_ptr), [id] "f" (id)
            : "t0", "memory",
              "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
        );
    }
}

static inline void quantize_row_q8_0_rvv_asm_one_block(const float * GGML_RESTRICT x_ptr, block_q8_0 * GGML_RESTRICT y) {
    float amax;
    // --- Part 1: Calculate amax using RVV ---
    asm volatile (
        "li t0, 32\n\t"
        "th.vsetvli x0, t0, e32, m1\n\t"
        "th.vmv.v.i v1, 0\n\t"
        "th.vsetvli x0, t0, e32, m8\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"
        "th.vfabs.v v8, v8\n\t"
        "th.vfredmax.vs v8, v8, v1\n\t"
        "th.vfmv.f.s %[amax], v8\n\t"
        : [amax] "=f" (amax)
        : [x_ptr] "r" (x_ptr)
        : "t0", "memory", "v1",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
    );

    // --- C++ part: Calculate scaling factor ---
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    y->d = GGML_FP32_TO_FP16(d);

    // --- Part 2: Scale, convert, and store using RVV ---
    asm volatile(
        "li t0, 32\n\t"
        "th.vsetvli x0, t0, e32, m8\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"
        "th.vfmul.vf v8, v8, %[id]\n\t"
        "th.vsetvli x0, t0, e16, m4\n\t"
        "th.vfncvt.x.f.v v4, v8\n\t"
        "th.vsetvli x0, t0, e8, m2\n\t"
        "th.vnsrl.vi v2, v4, 0\n\t"
        "th.vse.v v2, (%[y_ptr])\n\t"
        :
        : [x_ptr] "r" (x_ptr), [y_ptr] "r" (y->qs), [id] "f" (id)
        : "t0", "memory",
          "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
    );
}

void quantize_row_q8_0_rvv_asm_unroll2(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int64_t nb = k / QK8_0;
    int64_t i = 0;

    for (; i + 1 < nb; i += 2) {
        const float *x_ptr0 = x + (i + 0) * QK8_0;
        const float *x_ptr1 = x + (i + 1) * QK8_0;
        float amax[2];

        // --- Part 1: Find amax for 2 blocks (serially) ---
        asm volatile (
            "li t0, 32\n\t"
            // Set vtype for reduction accumulator
            "th.vsetvli x0, t0, e32, m1\n\t"
            "th.vmv.v.i v1, 0\n\t"

            // Process Block 0
            "th.vsetvli x0, t0, e32, m8\n\t"
            "th.vle.v v8, (%[x_ptr0])\n\t"
            "th.vfabs.v v8, v8\n\t"
            "th.vfredmax.vs v8, v8, v1\n\t"
            "th.vfmv.f.s %[amax0], v8\n\t"

            // Process Block 1
            "th.vle.v v16, (%[x_ptr1])\n\t"
            "th.vfabs.v v16, v16\n\t"
            "th.vfredmax.vs v16, v16, v1\n\t"
            "th.vfmv.f.s %[amax1], v16\n\t"
            : [amax0] "=f" (amax[0]), [amax1] "=f" (amax[1])
            : [x_ptr0] "r" (x_ptr0), [x_ptr1] "r" (x_ptr1)
            : "t0", "memory", "v1",
              "v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
        );

        // --- C++ part: Calculate scaling factors ---
        const float d0 = amax[0] / 127.0f;
        const float id0 = d0 ? 1.0f/d0 : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d0);

        const float d1 = amax[1] / 127.0f;
        const float id1 = d1 ? 1.0f/d1 : 0.0f;
        y[i + 1].d = GGML_FP32_TO_FP16(d1);

        // --- Part 2: Scale, convert, and store for 2 blocks (serially) ---
        asm volatile(
            "li t0, 32\n\t"
            // Process Block 0
            "th.vsetvli x0, t0, e32, m8\n\t"
            "th.vle.v v8, (%[x_ptr0])\n\t"
            "th.vfmul.vf v8, v8, %[id0]\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vfncvt.x.f.v v4, v8\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vnsrl.vi v2, v4, 0\n\t"
            "th.vse.v v2, (%[y_ptr0])\n\t"

            // Process Block 1
            "th.vsetvli x0, t0, e32, m8\n\t"
            "th.vle.v v8, (%[x_ptr1])\n\t"
            "th.vfmul.vf v8, v8, %[id1]\n\t"
            "th.vsetvli x0, t0, e16, m4\n\t"
            "th.vfncvt.x.f.v v4, v8\n\t"
            "th.vsetvli x0, t0, e8, m2\n\t"
            "th.vnsrl.vi v2, v4, 0\n\t"
            "th.vse.v v2, (%[y_ptr1])\n\t"
            :
            : [x_ptr0] "r" (x_ptr0), [y_ptr0] "r" (y[i].qs),   [id0] "f" (id0),
              [x_ptr1] "r" (x_ptr1), [y_ptr1] "r" (y[i+1].qs), [id1] "f" (id1)
            : "t0", "memory",
              "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
        );
    }

    // Tail loop for remaining blocks
    for (; i < nb; i++) {
        quantize_row_q8_0_rvv_asm_one_block(x + i * QK8_0, &y[i]);
    }
}
#endif
