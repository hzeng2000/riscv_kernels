#!/bin/bash

# 检查是否提供了模型名称作为参数
if [ -z "$1" ]; then
  echo "错误: 请提供模型名称作为第一个参数。"
  echo "用法: $0 <model_name>"
  exit 1
fi

# --- 配置 ---
MODEL_NAME="$1" # 从命令行参数获取模型名称
PROMPT_FILE="prompt_512.txt"

# --- 准备 Prompt ---
PROMPT_TEXT=$(cat "$PROMPT_FILE" | tr -d '\n\r' | sed 's/"/\\"/g')

# --- 预热模型 ---
echo " warming up the model: $MODEL_NAME..."
curl -s http://localhost:11434/api/generate -d '{
  "model": "'"$MODEL_NAME"'",
  "prompt": "hello",
  "stream": false,
  "options": {
    "num_predict": 1
  }
}' > /dev/null

# --- 执行测试并获取结果 ---
echo "Running Prompt Processing (PP) test..."
OLLAMA_RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "'"$MODEL_NAME"'",
  "prompt": "'"$PROMPT_TEXT"'",
  "stream": false,
  "options": {
    "num_predict": 1
  }
}')

# --- 解析和计算 ---
PROMPT_EVAL_DURATION_NS=$(echo "$OLLAMA_RESPONSE" | jq '.prompt_eval_duration')
NUM_PROMPT_TOKENS=$(echo "$OLLAMA_RESPONSE" | jq '.prompt_eval_count')
PP_TIME_S=$(echo "scale=6; $PROMPT_EVAL_DURATION_NS / 1000000000" | bc)
PP_THROUGHPUT=$(echo "scale=4; $NUM_PROMPT_TOKENS / $PP_TIME_S" | bc)

# --- 打印结果 ---
echo "--------------------------------------------------"
echo "Prompt Processing (PP) Results (from Ollama API)"
echo "--------------------------------------------------"
echo "Model: $MODEL_NAME"
echo "Prompt Tokens (actual): $NUM_PROMPT_TOKENS"
echo "PP Time (prompt_eval_duration): ${PP_TIME_S} s"
echo "PP Throughput: ${PP_THROUGHPUT} tokens/s"
echo "--------------------------------------------------"