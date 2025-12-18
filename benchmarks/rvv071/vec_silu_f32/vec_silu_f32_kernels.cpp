#include "vec_silu_f32_kernels.h"
#include "../common/common_defs.h"
#include <cmath>
#include <cassert>

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
#include <riscv_vector.h>
#endif

// Helper for scalar implementation
inline static float ggml_silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// 1. Baseline Scalar Implementation
void ggml_vec_silu_f32_scalar(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
// 2. RVV Intrinsics Implementation (Fast Approximation)
void ggml_vec_silu_f32_rvv_intrinsics_fast(const int n, float * y, const float * x) {
    int i = 0;
    for (; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);

        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + i, vl);
        
        // --- Fast exp(-x) Approximation ---
        vfloat32m1_t vneg_x = __riscv_vfneg_v_f32m1(vx, vl);
        vfloat32m1_t vz_clamped = __riscv_vfmax_vf_f32m1(vneg_x, -87.3f, vl);
        vz_clamped = __riscv_vfmin_vf_f32m1(vz_clamped, 88.7f, vl);

        const float exp_alpha_f = 12102203.0f; // (1 << 23) / log(2)
        const int32_t exp_bias_i  = 1065353216;   // 127 << 23

        vfloat32m1_t vscaled_z = __riscv_vfmul_vf_f32m1(vz_clamped, exp_alpha_f, vl);
        vint32m1_t vint_z = __riscv_vfcvt_x_f_v_i32m1(vscaled_z, vl);
        vint32m1_t vexp_int = __riscv_vadd_vx_i32m1(vint_z, exp_bias_i, vl);
        vfloat32m1_t vexp_val = __riscv_vreinterpret_v_i32m1_f32m1(vexp_int);

        // --- Final SiLU Calculation ---
        const float one_f = 1.0f;
        vfloat32m1_t vden = __riscv_vfadd_vf_f32m1(vexp_val, one_f, vl);
        vfloat32m1_t vy = __riscv_vfdiv_vv_f32m1(vx, vden, vl);

        __riscv_vse32_v_f32m1(y + i, vy, vl);
        i += vl;
    }
}
#endif

#if defined(__RVV_ASM_XTHEAD)

// --- Helper for tail processing ---
static inline void process_silu_tail_asm_fast(float* y, const float* x, size_t vl) {
    const float exp_alpha_f = 12102203.0f;
    const int32_t exp_bias_i  = 1065353216;
    const float one_f = 1.0f;
    asm volatile (
        "th.vsetvli x0, %[vl], e32, m1\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"
        "th.vfneg.v v9, v8\n\t"
        "th.vfmax.vf v9, v9, %[clamp_min]\n\t"
        "th.vfmin.vf v9, v9, %[clamp_max]\n\t"
        "th.vfmul.vf v9, v9, %[exp_alpha]\n\t"
        "th.vfcvt.x.f.v v10, v9\n\t"
        "th.vadd.vx v10, v10, %[exp_bias]\n\t"
        "th.vfadd.vf v9, v10, %[one]\n\t"
        "th.vfdiv.vv v8, v8, v9\n\t"
        "th.vse.v v8, (%[y_ptr])\n\t"
        :
        : [x_ptr] "r"(x), [y_ptr] "r"(y), [vl] "r"(vl),
          [clamp_min] "f"(-87.3f), [clamp_max] "f"(88.7f),
          [exp_alpha] "f"(exp_alpha_f), [exp_bias] "r"(exp_bias_i), [one] "f"(one_f)
        : "t0", "memory", "v8", "v9", "v10"
    );
}

// 3. RVV Assembly, Unroll=2, Serial (Fast Approximation)
void ggml_vec_silu_f32_rvv_asm_unroll2_serial_fast(const int n, float * y, const float * x) {
    int i = 0;
    size_t vl;
    while (i < n) {
        vl = __riscv_vsetvl_e32m1(n - i);
        if (vl * 2 <= (size_t)(n - i)) {
            // Process first chunk
            process_silu_tail_asm_fast(y + i, x + i, vl);
            // Process second chunk
            process_silu_tail_asm_fast(y + i + vl, x + i + vl, vl);
            i += vl * 2;
        } else {
            // Process remaining tail
            process_silu_tail_asm_fast(y + i, x + i, vl);
            i += vl;
        }
    }
}


// 4. RVV Assembly, Unroll=2, Pipelined (Fast Approximation)
void ggml_vec_silu_f32_rvv_asm_unroll2_pipelined_fast(const int n, float * y, const float * x) {
    int i = 0;
    const float exp_alpha_f = 12102203.0f;
    const int32_t exp_bias_i  = 1065353216;
    const float one_f = 1.0f;

    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        size_t vl2 = (n - i - vl > 0) ? __riscv_vsetvl_e32m1(n - i - vl) : 0;
        
        if (vl2 > 0) {
            asm volatile (
                "th.vsetvli x0, %[vl], e32, m1\n\t"
                "th.vle.v v8, (%[x_ptr])\n\t"
                "add %[x_ptr], %[x_ptr], %[vl_bytes]\n\t"
                "th.vsetvli x0, %[vl2], e32, m1\n\t"
                "th.vle.v v12, (%[x_ptr])\n\t"
                "add %[x_ptr], %[x_ptr], %[vl2_bytes]\n\t"
                "th.vsetvli x0, %[vl], e32, m1\n\t"
                "th.vfneg.v v9, v8\n\t"
                "th.vfmax.vf v9, v9, %[clamp_min]\n\t"
                "th.vfmin.vf v9, v9, %[clamp_max]\n\t"
                "th.vfmul.vf v9, v9, %[exp_alpha]\n\t"
                "th.vfcvt.x.f.v v10, v9\n\t"
                "th.vadd.vx v10, v10, %[exp_bias]\n\t"
                "th.vfadd.vf v9, v10, %[one]\n\t"
                "th.vfdiv.vv v8, v8, v9\n\t"
                "th.vsetvli x0, %[vl2], e32, m1\n\t"
                "th.vfneg.v v13, v12\n\t"
                "th.vfmax.vf v13, v13, %[clamp_min]\n\t"
                "th.vfmin.vf v13, v13, %[clamp_max]\n\t"
                "th.vfmul.vf v13, v13, %[exp_alpha]\n\t"
                "th.vfcvt.x.f.v v14, v13\n\t"
                "th.vadd.vx v14, v14, %[exp_bias]\n\t"
                "th.vfadd.vf v13, v14, %[one]\n\t"
                "th.vfdiv.vv v12, v12, v13\n\t"
                "th.vsetvli x0, %[vl], e32, m1\n\t"
                "th.vse.v v8, (%[y_ptr])\n\t"
                "add %[y_ptr], %[y_ptr], %[vl_bytes]\n\t"
                "th.vsetvli x0, %[vl2], e32, m1\n\t"
                "th.vse.v v12, (%[y_ptr])\n\t"
                "add %[y_ptr], %[y_ptr], %[vl2_bytes]\n\t"
                : [x_ptr] "+r"(x), [y_ptr] "+r"(y)
                : [vl] "r"(vl), [vl2] "r"(vl2),
                  [vl_bytes] "r"(vl * 4), [vl2_bytes] "r"(vl2 * 4),
                  [clamp_min] "f"(-87.3f), [clamp_max] "f"(88.7f),
                  [exp_alpha] "f"(exp_alpha_f), [exp_bias] "r"(exp_bias_i), [one] "f"(one_f)
                : "t0", "memory", "v8", "v9", "v10", "v12", "v13", "v14"
            );
            i += vl + vl2;
        } else {
            process_silu_tail_asm_fast(y + i, x + i, vl);
            i += vl;
        }
    }
}
void ggml_vec_silu_f32_asm_auto_generated(const int n, float * y, const float * x) {
    int i = 0;
    const float exp_alpha_f = 12102203.0f;
    const int32_t exp_bias_i  = 1065353216;
    const float one_f = 1.0f;
    
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        
        // Check if we can process `ur` full chunks
        bool can_unroll = true;
        for (int j=1; j < 2; ++j) {
            if (vl * j >= (size_t)(n-i)) {
                can_unroll = false;
                break;
            }
        }
        
        if (can_unroll) {
            size_t vl_bytes = vl * 4;
            asm volatile (
                    // --- Load all chunks ---
                    "th.vsetvli x0, %[vl], e32, m1\n\t"
                    "th.vle.v v8, (%[x_ptr])\n\t"
                    "add %[x_ptr], %[x_ptr], %[vl_bytes]\n\t"
                    "th.vsetvli x0, %[vl], e32, m1\n\t"
                    "th.vle.v v12, (%[x_ptr])\n\t"
                    "add %[x_ptr], %[x_ptr], %[vl_bytes]\n\t"
                    
                    // --- Compute all chunks ---
                    "th.vsetvli x0, %[vl], e32, m1\n\t"
                    "th.vfneg.v v9, v8\n\t"
                    "th.vfmax.vf v9, v9, %[clamp_min]\n\t"
                    "th.vfmin.vf v9, v9, %[clamp_max]\n\t"
                    "th.vfmul.vf v9, v9, %[exp_alpha]\n\t"
                    "th.vfcvt.x.f.v v10, v9\n\t"
                    "th.vadd.vx v10, v10, %[exp_bias]\n\t"
                    "th.vfadd.vf v9, v10, %[one]\n\t"
                    "th.vfdiv.vv v8, v8, v9\n\t"
                    "th.vsetvli x0, %[vl], e32, m1\n\t"
                    "th.vfneg.v v13, v12\n\t"
                    "th.vfmax.vf v13, v13, %[clamp_min]\n\t"
                    "th.vfmin.vf v13, v13, %[clamp_max]\n\t"
                    "th.vfmul.vf v13, v13, %[exp_alpha]\n\t"
                    "th.vfcvt.x.f.v v14, v13\n\t"
                    "th.vadd.vx v14, v14, %[exp_bias]\n\t"
                    "th.vfadd.vf v13, v14, %[one]\n\t"
                    "th.vfdiv.vv v12, v12, v13\n\t"
                    
                    // --- Store all chunks ---
                    "th.vsetvli x0, %[vl], e32, m1\n\t"
                    "th.vse.v v8, (%[y_ptr])\n\t"
                    "add %[y_ptr], %[y_ptr], %[vl_bytes]\n\t"
                    "th.vsetvli x0, %[vl], e32, m1\n\t"
                    "th.vse.v v12, (%[y_ptr])\n\t"
                    "add %[y_ptr], %[y_ptr], %[vl_bytes]\n\t"

                : [x_ptr] "+r"(x), [y_ptr] "+r"(y)
                : [vl] "r"(vl), [vl_bytes] "r"(vl_bytes),
                  [clamp_min] "f"(-87.3f), [clamp_max] "f"(88.7f),
                  [exp_alpha] "f"(exp_alpha_f), [exp_bias] "r"(exp_bias_i), [one] "f"(one_f)
                : "memory", "t0", "v10", "v12", "v13", "v14", "v8", "v9"            );
            i += vl * 2;
        } else {
            // Process remaining tail or single chunk
            process_silu_tail_asm_fast(y + i, x + i, vl);
            i += vl;
        }
    }
}
#endif
