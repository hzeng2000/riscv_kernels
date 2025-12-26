#!/bin/bash

# 模型列表（可直接从命令行或变量定义）
models=(
    "ollama-gemma3-4b-q8:latest"
    "ollama-gemma3-4b-q4:latest"
    "ollama-gemma3-270m-q8:latest"
    "ollama-gemma3-270m-q4:latest"
    "ollama-qwen3-1-7B-q8:latest"
    "ollama-qwen3-1-7B-q4:latest"
    "ollama-qwen3-0-6B-q8:latest"
    "ollama-qwen3-0-6B-q4:latest"
)

# 日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 获取当前时间戳（用于日志文件名）
timestamp=$(date +"%Y%m%d_%H%M")

# 遍历所有模型
for model in "${models[@]}"; do
    # 去掉 :latest 后缀（因为你的脚本可能只接受模型名部分）
    model_name="${model%:latest}"

    echo ">>> Processing model: $model_name"

    # 运行 pp512.sh 并记录日志
    pp_log="$LOG_DIR/pp512_${model_name}_$timestamp.log"
    echo "Running pp512.sh for $model_name..." | tee "$pp_log"
    bash pp512.sh "$model_name" >> "$pp_log" 2>&1
    echo ">>> Finished pp512 for $model_name (log: $pp_log)"

    # 运行 tg128.sh 并记录日志
    tg_log="$LOG_DIR/tg128_${model_name}_$timestamp.log"
    echo "Running tg128.sh for $model_name..." | tee "$tg_log"
    bash tg128.sh "$model_name" >> "$tg_log" 2>&1
    echo ">>> Finished tg128 for $model_name (log: $tg_log)"

    echo "--------------------------------------------------"
done

echo "✅ All models processed. Logs saved in $LOG_DIR/"