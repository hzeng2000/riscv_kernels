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

#include "vec_dot_kernels.h"
#include "../common/common_defs.h" 

// --- Test Configuration ---
struct TestConfig {
    int n = 64 * 32;
    int runs = 1000;
    std::map<std::string, bool> tests_to_run;
};

// --- Test Entry ---
struct TestEntry {
    std::string name;
    vec_dot_q8_0_q8_0_t func;
};

// --- Helper Functions ---
void generate_random_floats(std::vector<float>& vec) {
    std::mt19937 gen(1337); // Use fixed seed for reproducibility
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (float& val : vec) {
        val = dis(gen);
    }
}

void quantize_to_q8_0(const std::vector<float>& float_vec, std::vector<block_q8_0>& q_vec) {
    const int n = float_vec.size();
    const int nb = n / QK8_0;
    assert(n % QK8_0 == 0);
    q_vec.resize(nb);
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        const int offset = i * QK8_0;
        for (int j = 0; j < QK8_0; ++j) {
            amax = std::max(amax, std::abs(float_vec[offset + j]));
        }
        const float d = amax / 127.0f;
        q_vec[i].d = GGML_FP32_TO_FP16(d);
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            q_vec[i].qs[j] = static_cast<int8_t>(roundf(float_vec[offset + j] * id));
        }
    }
}

void print_help(char* app_name) {
    std::cout << "Usage: " << app_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -n <size>         Vector size (must be multiple of " << QK8_0 << ", default: 2048)\n";
    std::cout << "  -r <runs>         Number of test runs for performance measurement (default: 1000)\n";
    std::cout << "  --test-all        Run all available tests\n";
    std::cout << "  --test-intrinsics Run the RVV intrinsics test\n";
    std::cout << "  --test-asm        Run all hand-tuned assembly tests\n";
    std::cout << "  --help            Show this help message\n";
    std::cout << "If no test flags are provided, only the scalar version runs.\n";
}

void parse_args(int argc, char** argv, TestConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n") {
            config.n = std::stoi(argv[++i]);
        } else if (arg == "-r") {
            config.runs = std::stoi(argv[++i]);
        } else if (arg == "--test-all") {
            config.tests_to_run["all"] = true;
        } else if (arg == "--test-intrinsics") {
            config.tests_to_run["intrinsics"] = true;
        } else if (arg == "--test-asm") {
            config.tests_to_run["asm"] = true;
        } else if (arg == "--help") {
            print_help(argv[0]);
            exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_help(argv[0]);
            exit(1);
        }
    }
}

int main(int argc, char** argv) {
    TestConfig config;
    parse_args(argc, argv, config);

    std::cout << "Vector Size N = " << config.n << ", Test Runs = " << config.runs << std::endl;
    std::cout << "Preparing test data..." << std::endl;

    std::vector<float> x_f32(config.n), y_f32(config.n);
    generate_random_floats(x_f32);
    generate_random_floats(y_f32);

    std::vector<block_q8_0> x_q8_0, y_q8_0;
    quantize_to_q8_0(x_f32, x_q8_0);
    quantize_to_q8_0(y_f32, y_q8_0);

    // --- Define all tests ---
    std::vector<TestEntry> all_tests;
    all_tests.push_back({"Scalar C++", ggml_vec_dot_q8_0_q8_0_scalar});
#if defined(__riscv_v) || defined(__riscv_xtheadvector)
    if (config.tests_to_run["all"] || config.tests_to_run["intrinsics"]) {
        all_tests.push_back({"RVV Intrinsics", ggml_vec_dot_q8_0_q8_0_rvv_intrinsics});
    }
#endif
#if defined(__RVV_ASM_XTHEAD)
    if (config.tests_to_run["all"] || config.tests_to_run["asm"]) {
        all_tests.push_back({"Asm Unroll=1 (Serial)", ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll1});
        all_tests.push_back({"Asm Unroll=2 (Serial)", ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll2});
        all_tests.push_back({"Asm Unroll=2 (Fused)", ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll2_fused});
        all_tests.push_back({"Asm Unroll=4 (Serial)", ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll4});
        all_tests.push_back({"Asm Unroll=4 (Pipelined)", ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll4_pipelined});
        all_tests.push_back({"Asm FP64 Accum", ggml_vec_dot_q8_0_q8_0_rvv_asm_fp64_accum});
        all_tests.push_back({"Asm Blocked Accum", ggml_vec_dot_q8_0_q8_0_rvv_asm_blocked_accum});
        all_tests.push_back({"Asm Unroll=4 Prefetch", ggml_vec_dot_q8_0_q8_0_rvv_asm_unroll4_prefetch});
        all_tests.push_back({"Asm auto generated", ggml_vec_dot_q8_0_q8_0_rvv_asm_auto_generated});
    }
#endif

    // --- Run Tests ---
    float baseline_result = 0.0f;
    std::cout << "\n--- Correctness & Performance Check ---\n";
    std::cout << std::left << std::setw(30) << "Kernel Name"
              << std::setw(15) << "Avg Time (ms)"
              << std::setw(18) << "Result"
              << "Diff from Scalar\n";
    std::cout << std::string(80, '-') << std::endl;

    for (size_t i = 0; i < all_tests.size(); ++i) {
        auto const& test = all_tests[i];
        float result = 0.0f;
        volatile float result_sink = 0.0f; // Prevent over-optimization

        auto start_time = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < config.runs; ++k) {
            test.func(config.n, &result, x_q8_0.data(), y_q8_0.data());
            result_sink += result;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_time = end_time - start_time;
        
        double avg_time = total_time.count() / config.runs;

        if (i == 0) { // First test is always the scalar baseline
            baseline_result = result;
        }
        
        std::cout << std::left << std::setw(30) << test.name
                  << std::fixed << std::setprecision(6) << std::setw(15) << avg_time
                  << std::setw(18) << result
                  << std::abs(result - baseline_result) << "\n";
    }

    return 0;
}