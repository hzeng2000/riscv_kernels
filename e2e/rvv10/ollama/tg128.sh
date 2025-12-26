#!/bin/bash

MODEL_NAME="ollama-qwen3-0-6B"
TOKENS_TO_GENERATE=128

# 1. 首先测量一个极短prompt（例如"Hi"）的PP时间，作为基准扣除
# 假设"Hi"被tokenize为1个token
SHORT_PROMPT_TOKENS=1
SHORT_PROMPT_PP_TIME=$(curl -s -o /dev/null -w "%{time_starttransfer}" http://localhost:11434/api/generate -d '{
  "model": "'"$MODEL_NAME"'",
  "prompt": "Hi",
  "stream": false,
  "options": {
    "num_predict": 1
  }
}')
echo "Baseline PP time for a short prompt is: ${SHORT_PROMPT_PP_TIME} s"


echo "Running Token Generation (TG) test..."

# 2. 测量生成128个token的总时间
TOTAL_TIME=$(curl -s -o /dev/null -w "%{time_total}" http://localhost:11434/api/generate -d '{
  "model": "'"$MODEL_NAME"'",
  "prompt": "Hi",
  "stream": false,
  "options": {
    "num_predict": '"$TOKENS_TO_GENERATE"'
  }
}')

# 3. 计算纯生成时间
# Generation Time = Total Time - Prompt Processing Time
TG_TIME=$(echo "scale=6; $TOTAL_TIME - $SHORT_PROMPT_PP_TIME" | bc)

# 4. 计算生成吞吐量
TG_THROUGHPUT=$(echo "scale=4; $TOKENS_TO_GENERATE / $TG_TIME" | bc)

echo "------------------------------------"
echo "Token Generation (TG) Results"
echo "------------------------------------"
echo "Tokens Generated: $TOKENS_TO_GENERATE"
echo "Total Request Time: ${TOTAL_TIME} s"
echo "Pure Generation Time (Total - PP): ${TG_TIME} s"
echo "TG Throughput: ${TG_THROUGHPUT} tokens/s"
echo "------------------------------------"