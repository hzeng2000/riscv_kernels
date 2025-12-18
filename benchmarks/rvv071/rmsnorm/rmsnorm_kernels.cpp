#include "rmsnorm_kernels.h"
#include <cmath>
#include <cassert>
#include <cstdio>

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
#include <riscv_vector.h>
#endif

// 1. 基准纯 C++ 实现
void ggml_rmsnorm_f32_scalar(int n, float *y, const float *x, float eps) {
    assert(n > 0);
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_sq += (double)x[i] * x[i];
    }
    const float mean = (float)(sum_sq / n);
    const float scale = 1.0f / sqrtf(mean + eps);
    printf("sum_sq = %f, mean = %f, scale = %f\n", sum_sq, mean, scale);
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] * scale;
    }
}

#if defined(__riscv_v) || defined(__riscv_xtheadvector)
// 2. RVV Intrinsics 实现
void ggml_rmsnorm_f32_rvv_intrinsics(int n, float *y, const float *x, float eps) {
    assert(n > 0);
    double sum_sq = 0.0;
    size_t vl;

    // Pass 1: Vectorized sum of squares
    // Use a vector accumulator (m8) for intermediate sums
    vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t v_acc_sq = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v_x = __riscv_vle32_v_f32m8(x + i, vl);
        v_acc_sq = __riscv_vfmacc_vv_f32m8(v_acc_sq, v_x, v_x, vl);
    }

    // Reduce the vector accumulator to a scalar
    vl = __riscv_vsetvl_e32m8(n);
    vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t v_sum_sq = __riscv_vfredosum_vs_f32m8_f32m1(v_acc_sq, v_zero, vl);
    sum_sq = (double)__riscv_vfmv_f_s_f32m1_f32(v_sum_sq);

    // Scalar part: calculate the final scale
    const float mean = (float)(sum_sq / n);
    const float scale = 1.0f / sqrtf(mean + eps);

    // Pass 2: Vectorized scaling
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v_x = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t v_y = __riscv_vfmul_vf_f32m8(v_x, scale, vl);
        __riscv_vse32_v_f32m8(y + i, v_y, vl);
    }
}

void ggml_rmsnorm_f32_rvv_asm_unroll2_serial(int n, float *y, const float *x, float eps) {
    assert(n > 0);
    float sum_sq_f;

    // == Pass 1: Sum of Squares (Unroll 2x, Serial) ==
    asm volatile (
        "th.vsetvli x0, x0, e32, m2, d1\n\t" // 设置向量类型用于累加器
        "th.vmv.v.i v8, 0\n\t"              // v_acc0 = 0
        "th.vmv.v.i v12, 0\n\t"             // v_acc1 = 0
        "1:\n\t"
        // 设置向量长度 (vl)，并检查剩余元素数量 n 是否足够处理2个块
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "slli t1, t0, 1\n\t"                 // t1 = vl * 2
        "slt t3, %[n_in], t1\n\t"            // t3 = (n < vl*2) ? 1 : 0
        "bnez t3, 2f\n\t"                    // 如果 t3 非零, 跳转到收尾循环

        // --- 主循环体: 处理2个块 ---
        "slli t2, t0, 2\n\t"                 // t2 = vl_bytes
        "th.vle.v v4, (%[x_ptr])\n\t"         // 加载块0
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vfmacc.vv v8, v4, v4\n\t"         // 累加 v_acc0 += v4*v4

        "th.vle.v v6, (%[x_ptr])\n\t"         // 加载块1
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vfmacc.vv v12, v6, v6\n\t"        // 累加 v_acc1 += v6*v6

        "sub %[n_in], %[n_in], t1\n\t"        // n -= vl*2
        "bnez %[n_in], 1b\n\t"               // 如果 n > 0, 继续循环
        "j 3f\n\t"

        // --- 收尾循环: 处理剩余的 (< vl*2) 元素 ---
        "2:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "beqz t0, 3f\n\t"                    // 如果 vl == 0, 结束
        "th.vle.v v4, (%[x_ptr])\n\t"
        "th.vfmacc.vv v8, v4, v4\n\t"
        "slli t2, t0, 2\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "sub %[n_in], %[n_in], t0\n\t"
        "bnez %[n_in], 2b\n\t"

        // --- 归约: 合并所有累加器 ---
        "3:\n\t"
        "th.vsetvli x0, %[n_total], e32, m2\n\t"
        "th.vfadd.vv v8, v8, v12\n\t"         // v_acc0 += v_acc1
        "th.vsetvli x0, x0, e32, m1, d1\n\t"
        "th.vmv.v.i v24, 0\n\t"
        "th.vfredosum.vs v24, v8, v24\n\t"    // 归约最终的向量和
        "th.vfmv.f.s %[sum_out], v24\n\t"     // 将结果移出到浮点寄存器
        : [sum_out] "=f"(sum_sq_f), [x_ptr] "+r"(x), [n_in] "+r"(n)
        : [n_total] "r"(n)
        : "t0", "t1", "t2", "t3", "memory",
          "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v24"
    );

    const float mean = sum_sq_f / (float)n;
    const float scale = 1.0f / sqrtf(mean + eps);

    // == Pass 2: Apply Scale (Unroll 2x, Serial) ==
    // 为保持指针正确，重新赋值
    int remaining_n_pass2 = n;
    const float *x_ptr_pass2 = x;
    asm volatile(
        "flw fa0, 0(%[scale_ptr])\n\t"       // 加载 scale 值到浮点寄存器 fa0
        "1:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "slli t1, t0, 1\n\t"
        "slt t3, %[n_in], t1\n\t"
        "bnez t3, 2f\n\t"

        "slli t2, t0, 2\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"
        "th.vfmul.vf v8, v8, fa0\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vse.v v8, (%[y_ptr])\n\t"
        "add %[y_ptr], %[y_ptr], t2\n\t"

        "th.vle.v v10, (%[x_ptr])\n\t"
        "th.vfmul.vf v10, v10, fa0\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vse.v v10, (%[y_ptr])\n\t"
        "add %[y_ptr], %[y_ptr], t2\n\t"

        "sub %[n_in], %[n_in], t1\n\t"
        "bnez %[n_in], 1b\n\t"
        "j 3f\n\t"

        "2:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "beqz t0, 3f\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"
        "th.vfmul.vf v8, v8, fa0\n\t"
        "slli t2, t0, 2\n\t"
        "th.vse.v v8, (%[y_ptr])\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "sub %[n_in], %[n_in], t0\n\t"
        "bnez %[n_in], 2b\n\t"
        "3:\n\t"
        : [x_ptr] "+r"(x_ptr_pass2), [y_ptr] "+r"(y), [n_in] "+r"(remaining_n_pass2)
        : [scale_ptr] "r"(&scale)
        : "t0", "t1", "t2", "t3", "fa0", "memory", "v8", "v9", "v10", "v11"
    );
}

void ggml_rmsnorm_f32_rvv_asm_unroll2_pipelined(int n, float *y, const float *x, float eps) {
    assert(n > 0);
    float sum_sq_f;

    // == Pass 1: Sum of Squares (Unroll 2x, Pipelined) ==
    asm volatile (
        "th.vsetvli x0, x0, e32, m2, d1\n\t"
        "th.vmv.v.i v8, 0\n\t" // 单个累加器即可，因为计算是交错的
        "1:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "slli t1, t0, 1\n\t"
        "slt t3, %[n_in], t1\n\t"
        "bnez t3, 2f\n\t"

        // --- 流水线化加载/计算 ---
        "slli t2, t0, 2\n\t"
        "th.vle.v v4, (%[x_ptr])\n\t"         // 加载块0
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vle.v v6, (%[x_ptr])\n\t"         // 加载块1
        "th.vfmacc.vv v8, v4, v4\n\t"         // 计算块0的同时加载块1
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vfmacc.vv v8, v6, v6\n\t"         // 计算块1
        "sub %[n_in], %[n_in], t1\n\t"
        "bnez %[n_in], 1b\n\t"
        "j 3f\n\t"

        // --- 收尾循环 ---
        "2:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "beqz t0, 3f\n\t"
        "th.vle.v v4, (%[x_ptr])\n\t"
        "th.vfmacc.vv v8, v4, v4\n\t"
        "slli t2, t0, 2\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "sub %[n_in], %[n_in], t0\n\t"
        "bnez %[n_in], 2b\n\t"

        // --- 归约 ---
        "3:\n\t"
        "th.vsetvli x0, x0, e32, m1, d1\n\t"
        "th.vmv.v.i v24, 0\n\t"
        "th.vfredosum.vs v24, v8, v24\n\t"
        "th.vfmv.f.s %[sum_out], v24\n\t"
        : [sum_out] "=f"(sum_sq_f), [x_ptr] "+r"(x), [n_in] "+r"(n)
        :
        : "t0", "t1", "t2", "t3", "memory",
          "v4", "v5", "v6", "v7", "v8", "v9", "v24"
    );

    const float mean = sum_sq_f / (float)n;
    const float scale = 1.0f / sqrtf(mean + eps);

    // == Pass 2: Apply Scale (Unroll 2x, Pipelined) ==
    int remaining_n_pass2 = n;
    const float *x_ptr_pass2 = x;
    asm volatile(
        "flw fa0, 0(%[scale_ptr])\n\t"
        "1:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "slli t1, t0, 1\n\t"
        "slt t3, %[n_in], t1\n\t"
        "bnez t3, 2f\n\t"

        // --- 流水线化 加载/计算/存储 ---
        "slli t2, t0, 2\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"       // 加载块0
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vle.v v10, (%[x_ptr])\n\t"      // 加载块1
        "th.vfmul.vf v8, v8, fa0\n\t"       // 计算块0
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vse.v v8, (%[y_ptr])\n\t"       // 存储块0
        "th.vfmul.vf v10, v10, fa0\n\t"     // 计算块1
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "th.vse.v v10, (%[y_ptr])\n\t"      // 存储块1
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "sub %[n_in], %[n_in], t1\n\t"
        "bnez %[n_in], 1b\n\t"
        "j 3f\n\t"

        "2:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "beqz t0, 3f\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"
        "th.vfmul.vf v8, v8, fa0\n\t"
        "slli t2, t0, 2\n\t"
        "th.vse.v v8, (%[y_ptr])\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "sub %[n_in], %[n_in], t0\n\t"
        "bnez %[n_in], 2b\n\t"
        "3:\n\t"
        : [x_ptr] "+r"(x_ptr_pass2), [y_ptr] "+r"(y), [n_in] "+r"(remaining_n_pass2)
        : [scale_ptr] "r"(&scale)
        : "t0", "t1", "t2", "t3", "fa0", "memory", "v8", "v9", "v10", "v11"
    );
}

void ggml_rmsnorm_f32_rvv_asm_unroll4_pipelined(int n, float *y, const float *x, float eps) {
    assert(n > 0);
    float sum_sq_f;

    // == Pass 1: Sum of Squares (Unroll 4x, Pipelined) ==
    asm volatile (
        "th.vsetvli x0, x0, e32, m2, d1\n\t"
        "th.vmv.v.i v8,  0\n\t"      // v_acc0
        "th.vmv.v.i v12, 0\n\t"      // v_acc1
        "th.vmv.v.i v16, 0\n\t"      // v_acc2
        "th.vmv.v.i v20, 0\n\t"      // v_acc3
        "1:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "slli t1, t0, 2\n\t"         // t1 = vl * 4
        "slt t3, %[n_in], t1\n\t"    // t3 = (n < vl*4)
        "bnez t3, 2f\n\t"            // if true, jump to tail

        // --- 主循环: 4路展开流水线 ---
        "slli t2, t0, 2\n\t"         // t2 = vl_bytes
        "th.vle.v v4, (%[x_ptr])\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vle.v v6, (%[x_ptr])\n\t"
        "th.vfmacc.vv v8, v4, v4\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vle.v v10, (%[x_ptr])\n\t"
        "th.vfmacc.vv v12, v6, v6\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vle.v v14, (%[x_ptr])\n\t"
        "th.vfmacc.vv v16, v10, v10\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vfmacc.vv v20, v14, v14\n\t"
        "sub %[n_in], %[n_in], t1\n\t"
        "bnez %[n_in], 1b\n\t"
        "j 3f\n\t"

        // --- 收尾循环 ---
        "2:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "beqz t0, 3f\n\t"
        "th.vle.v v4, (%[x_ptr])\n\t"
        "th.vfmacc.vv v8, v4, v4\n\t" // 累加到第一个累加器
        "slli t2, t0, 2\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "sub %[n_in], %[n_in], t0\n\t"
        "bnez %[n_in], 2b\n\t"

        // --- 归约 ---
        "3:\n\t"
        "th.vsetvli x0, %[n_total], e32, m2\n\t"
        "th.vfadd.vv v8, v8, v12\n\t"
        "th.vfadd.vv v16, v16, v20\n\t"
        "th.vfadd.vv v8, v8, v16\n\t"
        "th.vsetvli x0, x0, e32, m1, d1\n\t"
        "th.vmv.v.i v24, 0\n\t"
        "th.vfredosum.vs v24, v8, v24\n\t"
        "th.vfmv.f.s %[sum_out], v24\n\t"
        : [sum_out] "=f"(sum_sq_f), [x_ptr] "+r"(x), [n_in] "+r"(n)
        : [n_total] "r"(n)
        : "t0", "t1", "t2", "t3", "memory",
          "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24"
    );

    const float mean = sum_sq_f / (float)n;
    const float scale = 1.0f / sqrtf(mean + eps);

    // == Pass 2: Apply Scale (Unroll 4x, Pipelined) ==
    int remaining_n_pass2 = n;
    const float *x_ptr_pass2 = x;
    asm volatile(
        "flw fa0, 0(%[scale_ptr])\n\t"
        "1:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "slli t1, t0, 2\n\t"
        "slt t3, %[n_in], t1\n\t"
        "bnez t3, 2f\n\t"

        // --- 流水线: L-L-C-L-C-S-L-C-S-C-S-S
        "slli t2, t0, 2\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"       // Load 0
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vle.v v10, (%[x_ptr])\n\t"      // Load 1
        "th.vfmul.vf v8, v8, fa0\n\t"       // Comp 0
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vle.v v12, (%[x_ptr])\n\t"      // Load 2
        "th.vfmul.vf v10, v10, fa0\n\t"     // Comp 1
        "th.vse.v v8, (%[y_ptr])\n\t"       // Store 0
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vle.v v14, (%[x_ptr])\n\t"      // Load 3
        "th.vfmul.vf v12, v12, fa0\n\t"     // Comp 2
        "th.vse.v v10, (%[y_ptr])\n\t"      // Store 1
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vfmul.vf v14, v14, fa0\n\t"     // Comp 3
        "th.vse.v v12, (%[y_ptr])\n\t"      // Store 2
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "th.vse.v v14, (%[y_ptr])\n\t"      // Store 3
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "sub %[n_in], %[n_in], t1\n\t"
        "bnez %[n_in], 1b\n\t"
        "j 3f\n\t"

        // --- 收尾循环 ---
        "2:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "beqz t0, 3f\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"
        "th.vfmul.vf v8, v8, fa0\n\t"
        "slli t2, t0, 2\n\t"
        "th.vse.v v8, (%[y_ptr])\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "sub %[n_in], %[n_in], t0\n\t"
        "bnez %[n_in], 2b\n\t"
        "3:\n\t"
        : [x_ptr] "+r"(x_ptr_pass2), [y_ptr] "+r"(y), [n_in] "+r"(remaining_n_pass2)
        : [scale_ptr] "r"(&scale)
        : "t0", "t1", "t2", "t3", "fa0", "memory",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
    );
}

void ggml_rmsnorm_f32_auto_generated(int n, float *y, const float *x, float eps) {
    assert(n > 0);
    float sum_sq_f;
    const int n_total = n; // Keep original n for mean calculation

    // ================== Pass 1: Sum of Squares ==================
    asm volatile (
        "th.vsetvli x0, x0, e32, m2, d1\n\t"
"th.vmv.v.i v8, 0\n\t""th.vmv.v.i v12, 0\n\t"        "1:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "slli t1, t0, 1\n\t"
        "slt t3, %[n_in], t1\n\t"
        "bnez t3, 2f\n\t"
        "slli t2, t0, 2\n\t"
        // --- Main Loop Body ---
            "th.vle.v v4, (%[x_ptr])\n\t"
            "add %[x_ptr], %[x_ptr], t2\n\t"
            "th.vfmacc.vv v8, v4, v4\n\t"
            "th.vle.v v6, (%[x_ptr])\n\t"
            "add %[x_ptr], %[x_ptr], t2\n\t"
            "th.vfmacc.vv v12, v6, v6\n\t"
        "sub %[n_in], %[n_in], t1\n\t"
        "bnez %[n_in], 1b\n\t"
        "j 3f\n\t"
        // --- Tail Loop ---
        "2:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "beqz t0, 3f\n\t"
        "th.vle.v v4, (%[x_ptr])\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vfmacc.vv v8, v4, v4\n\t"
        "sub %[n_in], %[n_in], t0\n\t"
        "bnez %[n_in], 2b\n\t"
        // --- Reduction ---
        "3:\n\t"
        "th.vsetvli x0, %[n_in], e32, m2\n\t"
"th.vfadd.vv v8, v8, v12\n\t"        "th.vsetvli x0, x0, e32, m1, d1\n\t"
        "th.vmv.v.i v24, 0\n\t"
        "th.vfredosum.vs v24, v8, v24\n\t"
        "th.vfmv.f.s %[sum_out], v24\n\t"
        : [sum_out] "=f"(sum_sq_f), [x_ptr] "+r"(x), [n_in] "+r"(n)
        : 
        : "t0", "t1", "t2", "t3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "fa0", "memory"    );

    const float mean = sum_sq_f / (float)n;
    const float scale = 1.0f / sqrtf(mean + eps);

    // ================== Pass 2: Apply Scale ==================
    // CRITICAL: We need new local copies for pass 2, initialized with original values.
    const float *x_ptr_pass2 = x;
    float *y_ptr_pass2 = y;
    int n_pass2 = n; // Use the saved n
    asm volatile(
        "flw fa0, 0(%[scale_ptr])\n\t"
        "1:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "slli t1, t0, 1\n\t"
        "slt t3, %[n_in], t1\n\t"
        "bnez t3, 2f\n\t"
        "slli t2, t0, 2\n\t"
        // --- Main Loop Body ---
        "th.vle.v v8, (%[x_ptr])\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vfmul.vf v8, v8, fa0\n\t"
        "th.vse.v v8, (%[y_ptr])\n\t"
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "th.vle.v v10, (%[x_ptr])\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vfmul.vf v10, v10, fa0\n\t"
        "th.vse.v v10, (%[y_ptr])\n\t"
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "sub %[n_in], %[n_in], t1\n\t"
        "bnez %[n_in], 1b\n\t"
        "j 3f\n\t"
        // --- Tail Loop ---
        "2:\n\t"
        "th.vsetvli t0, %[n_in], e32, m2\n\t"
        "beqz t0, 3f\n\t"
        "th.vle.v v8, (%[x_ptr])\n\t"
        "slli t2, t0, 2\n\t"
        "add %[x_ptr], %[x_ptr], t2\n\t"
        "th.vfmul.vf v8, v8, fa0\n\t"
        "th.vse.v v8, (%[y_ptr])\n\t"
        "add %[y_ptr], %[y_ptr], t2\n\t"
        "sub %[n_in], %[n_in], t0\n\t"
        "bnez %[n_in], 2b\n\t"
        "3:\n\t"
        : [x_ptr] "+r"(x_ptr_pass2), [y_ptr] "+r"(y_ptr_pass2), [n_in] "+r"(n_pass2)
        : [scale_ptr] "r"(&scale)
        : "t0", "t1", "t2", "t3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "fa0", "memory"    );
}
#endif