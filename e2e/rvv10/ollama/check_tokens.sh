#!/bin/bash

MODEL_NAME="ollama-qwen3-0-6B"
PROMPT_FILE="prompt_512.txt"
PROMPT_TEXT=$(cat "$PROMPT_FILE" | tr -d '\n\r' | sed 's/"/\\"/g')

# 实际的Token数量，需要你自己确认，或者通过Ollama的 /api/tokenize 接口获取
# 这里我们假设为512个
NUM_PROMPT_TOKENS=512

echo " warming up the model..."
# 预热模型，结果不计入
curl -s http://localhost:11434/api/generate -d '{
  "model": "'"$MODEL_NAME"'",
  "prompt": "hello",
  "stream": false,
  "options": {
    "num_predict": 1
  }
}' > /dev/null

echo "Running Prompt Processing (PP) test..."

# 使用 curl 的 time_starttransfer 来测量到第一个字节的时间
# 这就是我们的提示处理时间 (PP Time)
curl -s http://localhost:11434/api/generate -d '{
  "model": "'"$MODEL_NAME"'",
  "prompt": "'"$PROMPT_TEXT"'",
  "stream": false,
  "options": {
    "num_predict": 1
  }
}'

# curl -s http://localhost:11434/api/generate -d '{ "model": "ollama-qwen3-0-6B", "prompt": "hello","stream": false,  "options": {"num_predict": 1 } }'
