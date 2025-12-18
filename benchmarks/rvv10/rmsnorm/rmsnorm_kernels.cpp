// RVV 1.0 RMSNorm Kernels
// 
// 这个文件将由 patch_kernels.sh 脚本自动填充
// 或者手动从 vectorweaver/generated_kernels 复制
//
// 请使用以下命令生成kernel：
// cd vectorweaver
// python3 main.py -k rmsnorm --hw sg2044 --generate-all

#include "rmsnorm_kernels.h"
#include "../common/common_defs.h"
#include <cmath>
#include <cassert>

#if defined(__riscv_v)
#include <riscv_vector.h>
#endif

// 基准纯 C++ 实现 (RVV版本无关)
void ggml_rmsnorm_f32_scalar(int n, float *y, const float *x, float eps) {
    assert(n > 0);
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_sq += (double)x[i] * x[i];
    }
    const float mean = (float)(sum_sq / n);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] * scale;
    }
}

// ============================================
// 以下是RVV 1.0版本的kernel实现
// 由VectorWeaver自动生成
// ============================================

// TODO: 使用 patch_kernels.sh 或手动复制生成的kernel到这里

#if defined(__riscv_v)
// RVV 1.0 Intrinsics 实现
void ggml_rmsnorm_f32_rvv_intrinsics(int n, float *y, const float *x, float eps) {
    assert(n > 0);
    double sum_sq = 0.0;
    size_t vl;

    vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t v_acc_sq = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v_x = __riscv_vle32_v_f32m8(x + i, vl);
        v_acc_sq = __riscv_vfmacc_vv_f32m8(v_acc_sq, v_x, v_x, vl);
    }

    vl = __riscv_vsetvl_e32m8(n);
    vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t v_sum_sq = __riscv_vfredosum_vs_f32m8_f32m1(v_acc_sq, v_zero, vl);
    sum_sq = (double)__riscv_vfmv_f_s_f32m1_f32(v_sum_sq);

    const float mean = (float)(sum_sq / n);
    const float scale = 1.0f / sqrtf(mean + eps);

    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v_x = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t v_y = __riscv_vfmul_vf_f32m8(v_x, scale, vl);
        __riscv_vse32_v_f32m8(y + i, v_y, vl);
    }
}
#endif

