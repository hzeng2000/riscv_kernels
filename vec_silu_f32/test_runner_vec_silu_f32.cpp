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

#include "../common_defs.h"
#include "vec_silu_f32_kernels.h"

// --- Test Configuration ---
struct TestConfig {
    int n = 2048;
    int runs = 1000;
    std::map<std::string, bool> tests_to_run;
};

// --- Test Entry ---
struct TestEntry {
    std::string name;
    vec_silu_f32_t func;
};

// --- Helper Functions ---
void generate_random_floats(std::vector<float>& vec, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-10.0, 10.0); // Use a wider range for SiLU
    for (float& val : vec) {
        val = dis(gen);
    }
}

void print_help(char* app_name) {
    std::cout << "Usage: " << app_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -n <size>         Vector size (default: 2048)\n";
    std::cout << "  -r <runs>         Number of test runs (default: 1000)\n";
    std::cout << "  --test-all        Run all available tests\n";
    std::cout << "  --test-intrinsics Run the RVV intrinsics test\n";
    std::cout << "  --test-asm        Run all hand-tuned assembly tests\n";
    std::cout << "  --help            Show this help message\n";
}

void parse_args(int argc, char** argv, TestConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n") config.n = std::stoi(argv[++i]);
        else if (arg == "-r") config.runs = std::stoi(argv[++i]);
        else if (arg == "--test-all") config.tests_to_run["all"] = true;
        else if (arg == "--test-intrinsics") config.tests_to_run["intrinsics"] = true;
        else if (arg == "--test-asm") config.tests_to_run["asm"] = true;
        else if (arg == "--help") { print_help(argv[0]); exit(0); }
        else { std::cerr << "Unknown argument: " << arg << std::endl; print_help(argv[0]); exit(1); }
    }
}

double check_correctness(int n, const std::vector<float>& base, const std::vector<float>& test) {
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, (double)std::abs(base[i] - test[i]));
    }
    return max_diff;
}

int main(int argc, char** argv) {
    TestConfig config;
    parse_args(argc, argv, config);

    std::cout << "Vector Size N = " << config.n << ", Test Runs = " << config.runs << std::endl;
    std::cout << "Preparing test data for SiLU f32..." << std::endl;

    std::vector<float> x_f32(config.n);
    generate_random_floats(x_f32, 1337);

    // --- Define all tests ---
    std::vector<TestEntry> all_tests;
    all_tests.push_back({"Scalar C++", ggml_vec_silu_f32_scalar});
#if defined(__riscv_v) || defined(__riscv_xtheadvector)
    if (config.tests_to_run["all"] || config.tests_to_run["intrinsics"]) {
        all_tests.push_back({"Intrinsics (Fast)", ggml_vec_silu_f32_rvv_intrinsics_fast});
    }
#endif
#if defined(__RVV_ASM_XTHEAD)
    if (config.tests_to_run["all"] || config.tests_to_run["asm"]) {
        all_tests.push_back({"Asm U2 Serial (Fast)", ggml_vec_silu_f32_rvv_asm_unroll2_serial_fast});
        all_tests.push_back({"Asm U2 Pipe (Fast)", ggml_vec_silu_f32_rvv_asm_unroll2_pipelined_fast});
        all_tests.push_back({"Asm auto generated", ggml_vec_silu_f32_asm_auto_generated});
    }
#endif

    // --- Run Tests ---
    std::vector<float> base_y(config.n);
    ggml_vec_silu_f32_scalar(config.n, base_y.data(), x_f32.data());

    std::cout << "\n--- SiLU F32 Correctness & Performance Check ---\n";
    std::cout << std::left << std::setw(30) << "Kernel Name"
              << std::setw(18) << "Avg Time (ms)"
              << "Max Diff\n";
    std::cout << std::string(70, '-') << std::endl;

    for (const auto& test : all_tests) {
        std::vector<float> y(config.n, 0.0f);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < config.runs; ++k) {
            test.func(config.n, y.data(), x_f32.data());
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_time = end_time - start_time;
        
        double avg_time = total_time.count() / config.runs;
        double diff = check_correctness(config.n, base_y, y);
        
        std::cout << std::left << std::setw(30) << test.name
                  << std::fixed << std::setprecision(6) << std::setw(18) << avg_time
                  << diff << "\n";
    }

    return 0;
}
