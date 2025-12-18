 ``` bash
#  -D__riscv_xtheadvector is definded in -march=rv64gc_xtheadvector1p0_zfhmin
riscv64-unknown-linux-gnu-g++ -o \
    test_runner_vec_silu_f32.out \
    test_runner_vec_silu_f32.cpp \
    vec_silu_f32_kernels.cpp \
    -march=rv64gc_xtheadvector1p0_zfhmin \
    -mabi=lp64d \
    -D__riscv_v \
    -D__RVV_ASM_XTHEAD \
    -O3


# -s
riscv64-unknown-linux-gnu-g++ -S -o kernels.s vec_dot_q4_0_q8_0_kernels.cpp -march=rv64gc_xtheadvector1p0_zfhmin -mabi=lp64d -D__riscv_v -O3
```