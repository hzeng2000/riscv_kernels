#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <map>
#include <functional>
#include <iomanip>
#include <cassert>
#include <algorithm>

#include "../common/common_defs.h"
#include "ggml_gemm_q4_0_8x8_q8_0_kernels.h"

// --- Test Configuration ---
struct TestConfig {
    int M = 128;
    int N = 256;
    int K = 512;
    int runs = 10;
    std::map<std::string, bool> tests_to_run;
};

// --- Test Entry ---
struct TestEntry {
    std::string name;
    gemm_q4_0_8x8_q8_0_t func;
};

// --- Helper Functions ---
void generate_random_floats(std::vector<float>& vec, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (float& val : vec) {
        val = dis(gen);
    }
}

// Quantizer for Matrix A -> block_q8_0x4
void quantize_to_q8_0x4(int M, int K, const std::vector<float>& float_vec, std::vector<block_q8_0x4>& q_vec) {
    const int qk = QK8_0;
    const int nb_k = K / qk;
    const int M_div_4 = M / 4;
    q_vec.resize(M_div_4 * nb_k);

    for (int y = 0; y < M_div_4; ++y) {
        for (int l = 0; l < nb_k; ++l) {
            block_q8_0x4& block = q_vec[y * nb_k + l];
            for (int m = 0; m < 4; ++m) {
                float amax = 0.0f;
                const int row_offset = (y * 4 + m) * K;
                const int col_offset = l * qk;

                for (int k = 0; k < qk; ++k) {
                    amax = std::max(amax, std::abs(float_vec[row_offset + col_offset + k]));
                }
                const float d = amax / 127.0f;
                block.d[m] = GGML_FP32_TO_FP16(d);
                const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

                for (int k = 0; k < qk; ++k) {
                    block.qs[m * qk + k] = static_cast<int8_t>(roundf(float_vec[row_offset + col_offset + k] * id));
                }
            }
        }
    }
}

// Quantizer for Matrix B -> block_q4_0x8 (Corrected)
void quantize_to_q4_0x8(int K, int N, const std::vector<float>& float_vec, std::vector<block_q4_0x8>& q_vec) {
    const int qk = QK8_0;
    const int nb_k = K / qk;
    const int ncols_interleaved = 8;
    const int N_div_8 = N / ncols_interleaved;
    q_vec.resize(N_div_8 * nb_k);

    for (int x = 0; x < N_div_8; ++x) {
        for (int l = 0; l < nb_k; ++l) {
            block_q4_0x8& block = q_vec[x * nb_k + l];
            for (int j = 0; j < ncols_interleaved; ++j) {
                float amax = 0.0f;
                const int col_idx = x * ncols_interleaved + j;

                for (int k_block = 0; k_block < qk; ++k_block) {
                    amax = std::max(amax, std::abs(float_vec[(l*qk + k_block) * N + col_idx]));
                }
                const float d = amax / 7.0f; // Q4 is symmetric -8..7, max abs value is 7
                block.d[j] = GGML_FP32_TO_FP16(d);
                const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

                for (int k_half = 0; k_half < qk / 2; ++k_half) {
                    const float f0 = float_vec[(l*qk + k_half) * N + col_idx];
                    const float f1 = float_vec[(l*qk + k_half + qk/2) * N + col_idx];

                    const uint8_t i0 = static_cast<uint8_t>(roundf(f0 * id) + 8);
                    const uint8_t i1 = static_cast<uint8_t>(roundf(f1 * id) + 8);
                    
                    block.qs[k_half * ncols_interleaved + j] = (i0 & 0x0F) | ((i1 & 0x0F) << 4);
                }
            }
        }
    }
}

// ========== Scalar Baseline Implementation ==========

// Generic scalar implementation (original signature)
void ggml_gemm_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

// Adapter to match the auto-generated header signature
void ggml_gemm_q4_0_8x8_q8_0_scalar(int M, int N, int K, const void * A, const void * B, float * C) {
    // Parameter mapping:
    // M -> nr (number of rows in output)
    // N -> nc (number of cols in output)
    // K -> n (inner dimension)
    // A -> vy (block_q8_0x4, left matrix)
    // B -> vx (block_q4_0x8, right matrix)
    // C -> s (output matrix)
    // bs = N (stride for output matrix)
    ggml_gemm_q4_0_8x8_q8_0_generic(K, C, N, B, A, M, N);
}


void print_help(char* app_name) {
    std::cout << "Usage: " << app_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -M <rows>         Matrix A rows (default: 128)\n";
    std::cout << "  -N <cols>         Matrix B cols (default: 256)\n";
    std::cout << "  -K <inner_dim>    Matrix A cols / B rows (default: 512)\n";
    std::cout << "  -r <runs>         Number of test runs (default: 10)\n";
    std::cout << "  --test-all        Run all available tests\n";
    std::cout << "  --test-intrinsics Run the RVV intrinsics test\n";
    std::cout << "  --test-asm        Run all hand-tuned assembly tests\n";
    std::cout << "  --help            Show this help message\n";
}

void parse_args(int argc, char** argv, TestConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-M") config.M = std::stoi(argv[++i]);
        else if (arg == "-N") config.N = std::stoi(argv[++i]);
        else if (arg == "-K") config.K = std::stoi(argv[++i]);
        else if (arg == "-r") config.runs = std::stoi(argv[++i]);
        else if (arg == "--test-all") config.tests_to_run["all"] = true;
        else if (arg == "--test-intrinsics") config.tests_to_run["intrinsics"] = true;
        else if (arg == "--test-asm") config.tests_to_run["asm"] = true;
        else if (arg == "--help") { print_help(argv[0]); exit(0); }
        else { std::cerr << "Unknown argument: " << arg << std::endl; print_help(argv[0]); exit(1); }
    }
}

double check_correctness(int M, int N, const std::vector<float>& base, const std::vector<float>& test) {
    double diff_sum = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        double current_diff = std::abs(base[i] - test[i]);
        diff_sum += current_diff;
        max_diff = std::max(max_diff, current_diff);
    }
    // Return average difference for better scaling with matrix size
    return diff_sum / (M*N);
}

int main(int argc, char** argv) {
    TestConfig config;
    parse_args(argc, argv, config);

    if (config.M % 4 != 0 || config.N % 8 != 0 || config.K % QK8_0 != 0) {
        std::cerr << "Error: M must be multiple of 4, N multiple of 8, K multiple of " << QK8_0 << std::endl;
        return 1;
    }

    std::cout << "M=" << config.M << ", N=" << config.N << ", K=" << config.K << ", Runs=" << config.runs << std::endl;
    std::cout << "Preparing test data..." << std::endl;

    std::vector<float> A_f32(config.M * config.K);
    std::vector<float> B_f32(config.K * config.N);
    generate_random_floats(A_f32, 1337);
    generate_random_floats(B_f32, 42);

    std::vector<block_q8_0x4> A_q;
    std::vector<block_q4_0x8> B_q;
    quantize_to_q8_0x4(config.M, config.K, A_f32, A_q);
    quantize_to_q4_0x8(config.K, config.N, B_f32, B_q);
    
    std::cout << "Data quantized." << std::endl;

    // --- Define all tests ---
    std::vector<TestEntry> all_tests;
    all_tests.push_back({"Scalar C++", ggml_gemm_q4_0_8x8_q8_0_scalar});
#if defined(__riscv_v)
    if (config.tests_to_run["all"] || config.tests_to_run["intrinsics"]) {
        all_tests.push_back({"RVV Intrinsics", ggml_gemm_q4_0_8x8_q8_0_rvv_intrinsics});
    }
#endif
#if defined(__RVV_ASM_STD)
    if (config.tests_to_run["all"] || config.tests_to_run["asm"]) {
        all_tests.push_back({"Asm Unroll=1 (baseline)", ggml_gemm_q4_0_8x8_q8_0_baseline});
        all_tests.push_back({"Asm Unroll=2 Serial", ggml_gemm_q4_0_8x8_q8_0_asm_unroll2});
        all_tests.push_back({"Asm Unroll=2 Interleaved", ggml_gemm_q4_0_8x8_q8_0_asm_unroll2_interleaved});
    }
#endif

    // --- Run Tests ---
    std::vector<float> base_C(config.M * config.N);
    std::cout << "\n--- GEMM Q4_0_8x8 * Q8_0 Correctness & Performance Check ---\n";
    std::cout << std::left << std::setw(25) << "Kernel Name"
              << std::setw(18) << "Avg Time (ms)"
              << "Avg Diff\n";
    std::cout << std::string(60, '-') << std::endl;

    for (size_t i = 0; i < all_tests.size(); ++i) {
        auto const& test = all_tests[i];
        std::vector<float> C(config.M * config.N, 0.0f);

        auto start_time = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < config.runs; ++k) {
            test.func(config.M, config.N, config.K, A_q.data(), B_q.data(), C.data());
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_time = end_time - start_time;
        
        double avg_time = total_time.count() / config.runs;
        double diff = 0.0;

        if (i == 0) {
            base_C = C;
        } else {
            diff = check_correctness(config.M, config.N, base_C, C);
        }
        
        std::cout << std::left << std::setw(25) << test.name
                  << std::fixed << std::setprecision(6) << std::setw(18) << avg_time
                  << diff << "\n";
    }

    return 0;
}
