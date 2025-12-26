#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

MODEL_NAME="$1"
TOKENS_TO_GENERATE=128

echo "Warming up and measuring Prompt Processing (PP) time for model: $MODEL_NAME..."

# --- Step 1: 获取纯 Prompt Processing 时间 ---
# 使用 stream=true，只读取第一个 chunk（包含 prompt_eval_count 和 prompt_eval_duration）
FIRST_CHUNK=$(timeout 30 curl -s http://localhost:11434/api/generate \
  -d '{
    "model": "'"$MODEL_NAME"'",
    "prompt": "Hi",
    "stream": true,
    "options": {
      "num_predict": 0
    }
  }' | head -n1)

if [ -z "$FIRST_CHUNK" ]; then
    echo "ERROR: Failed to get response from Ollama API."
    exit 1
fi

# 提取 prompt_eval_duration（单位：纳秒 → 转为秒）
PP_DURATION_NS=$(echo "$FIRST_CHUNK" | jq -r '.prompt_eval_duration // empty')
if [ -z "$PP_DURATION_NS" ]; then
    echo "ERROR: prompt_eval_duration not found in response."
    echo "Response was: $FIRST_CHUNK"
    exit 1
fi

PP_TIME=$(echo "scale=6; $PP_DURATION_NS / 1000000000" | bc)
echo "Measured Prompt Processing (PP) time: ${PP_TIME} s"

# --- Step 2: 测量生成 128 tokens 的总时间（含 PP）---
echo "Running Token Generation (TG) test (generating $TOKENS_TO_GENERATE tokens)..."

TOTAL_TIME=$(curl -s -o /dev/null -w "%{time_total}" http://localhost:11434/api/generate -d '{
  "model": "'"$MODEL_NAME"'",
  "prompt": "Hi",
  "stream": false,
  "options": {
    "num_predict": '"$TOKENS_TO_GENERATE"'
  }
}')

# --- Step 3: 计算纯生成时间 ---
TG_TIME=$(echo "scale=6; $TOTAL_TIME - $PP_TIME" | bc)

# 安全检查：防止负值
if (( $(echo "$TG_TIME <= 0" | bc -l) )); then
    echo "WARNING: Computed TG time is non-positive ($TG_TIME s). Using total time as fallback."
    TG_TIME=$TOTAL_TIME
fi

# --- Step 4: 计算吞吐 ---
TG_THROUGHPUT=$(echo "scale=4; $TOKENS_TO_GENERATE / $TG_TIME" | bc)

# --- 输出结果 ---
echo "------------------------------------"
echo "Token Generation (TG) Results"
echo "------------------------------------"
echo "Model: $MODEL_NAME"
echo "Tokens Generated: $TOKENS_TO_GENERATE"
echo "Total Request Time: ${TOTAL_TIME} s"
echo "Prompt Processing (PP) Time: ${PP_TIME} s"
echo "Pure Generation Time (TG): ${TG_TIME} s"
echo "TG Throughput: ${TG_THROUGHPUT} tokens/s"
echo "------------------------------------"