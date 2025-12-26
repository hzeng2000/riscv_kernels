#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

MODEL_NAME="$1"
TOKENS_TO_GENERATE=128

RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "'"$MODEL_NAME"'",
  "prompt": "Hi",
  "stream": false,
  "options": {"num_predict": '"$TOKENS_TO_GENERATE"'}
}')

# 使用 jq 提取
PP_NS=$(echo "$RESPONSE" | jq -r '.prompt_eval_duration // empty')
TG_NS=$(echo "$RESPONSE" | jq -r '.eval_duration // empty')
PP_CNT=$(echo "$RESPONSE" | jq -r '.prompt_eval_count // empty')
TG_CNT=$(echo "$RESPONSE" | jq -r '.eval_count // empty')

if [ -z "$PP_NS" ] || [ -z "$TG_NS" ] || [ -z "$TG_CNT" ]; then
    echo "ERROR parsing response:"
    echo "$RESPONSE"
    exit 1
fi

PP_TIME=$(echo "scale=6; $PP_NS / 1e9" | bc)
TG_TIME=$(echo "scale=6; $TG_NS / 1e9" | bc)
THROUGHPUT=$(echo "scale=4; $TG_CNT / $TG_TIME" | bc)

echo "------------------------------------"
echo "Model: $MODEL_NAME"
echo "Prompt Tokens: $PP_CNT"
echo "Generated Tokens: $TG_CNT"
echo "PP Time: ${PP_TIME}s"
echo "TG Time: ${TG_TIME}s"
echo "TG Throughput: ${THROUGHPUT} tokens/s"
echo "------------------------------------"