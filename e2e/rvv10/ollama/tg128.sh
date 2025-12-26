#!/bin/bash

# 确保安装了 jq
if ! command -v jq &> /dev/null; then
    echo "jq could not be found, please install it."
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

MODEL_NAME="$1"
TOKENS_TO_GENERATE=128

echo "Running Token Generation (TG) test for model: $MODEL_NAME (target: ${TOKENS_TO_GENERATE} tokens)..."

# 调用 Ollama API（非流式）
RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "'"$MODEL_NAME"'",
  "prompt": "Hi",
  "stream": false,
  "options": {
    "num_predict": '"$TOKENS_TO_GENERATE"'
  }
}')

# 检查是否包含 done:true（确保请求完成）
if ! echo "$RESPONSE" | jq -e '.done' > /dev/null; then
    echo "ERROR: Incomplete response from Ollama."
    echo "Raw: $RESPONSE"
    exit 1
fi

# 使用 jq 提取关键字段
PP_NS=$(echo "$RESPONSE" | jq '.prompt_eval_duration')
TG_NS=$(echo "$RESPONSE" | jq '.eval_duration')
PP_CNT=$(echo "$RESPONSE" | jq '.prompt_eval_count')
TG_CNT=$(echo "$RESPONSE" | jq '.eval_count')

# 检查是否提取成功并转换为整数
if [[ "$PP_NS" =~ ^[0-9]+$ ]] && [[ "$TG_NS" =~ ^[0-9]+$ ]] && [[ "$PP_CNT" =~ ^[0-9]+$ ]] && [[ "$TG_CNT" =~ ^[0-9]+$ ]]; then
    # 转换为秒
    PP_TIME=$(echo "scale=6; $PP_NS / 1000000000" | bc)
    TG_TIME=$(echo "scale=6; $TG_NS / 1000000000" | bc)

    # 计算吞吐（使用实际生成的 token 数）
    if (( $(echo "$TG_TIME > 0" | bc -l) )); then
        TG_THROUGHPUT=$(echo "scale=4; $TG_CNT / $TG_TIME" | bc)
    else
        TG_THROUGHPUT="N/A"
    fi
else
    echo "ERROR: Failed to extract timing or token count from response. Ensure all values are integers."
    echo "Response: $RESPONSE"
    exit 1
fi

# 输出结果
echo "------------------------------------"
echo "Token Generation (TG) Results"
echo "------------------------------------"
echo "Model: $MODEL_NAME"
echo "Prompt Tokens: $PP_CNT"
echo "Generated Tokens (actual): $TG_CNT"
echo "Prompt Processing Time: ${PP_TIME} s"
echo "Token Generation Time: ${TG_TIME} s"
echo "TG Throughput: ${TG_THROUGHPUT} tokens/s"
echo "------------------------------------"