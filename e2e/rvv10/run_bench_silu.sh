#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ================= 配置区域 =================
# 根目录，根据你的描述是 /root/wmw
BASE_DIR="${SCRIPT_DIR}/out/phase_silu"

# 模型所在的根目录
MODEL_ROOT_DIR="/root/wmw/unsloth_riscv"

# 需要测试的变体目录
VARIANTS=("vw" "1" "2" "3" "4" "5" "6")

# 日志文件名称，包含时间戳
LOG_FILE="${BASE_DIR}/benchmark_silu_$(date +%Y%m%d_%H%M%S).log"

# ===========================================

echo "开始测试..." | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"

# 1. 遍历变体目录
for variant in "${VARIANTS[@]}"; do
    
    # 构建 binary 所在的完整路径
    # BIN_DIR="${BASE_DIR}/${variant}/VectorWeaver_rvv1/build-riscv/bin"
    BIN_DIR="${BASE_DIR}/${variant}"
    
    echo "正在处理变体目录: $variant" | tee -a "$LOG_FILE"
    
    # 检查该目录是否存在
    if [ ! -d "$BIN_DIR" ]; then
        echo "错误: 目录不存在 $BIN_DIR, 跳过..." | tee -a "$LOG_FILE"
        continue
    fi

    # 进入二进制文件目录 (通常 bench 工具需要在当前目录下运行以加载依赖)
    pushd "$BIN_DIR" > /dev/null
    
    # 检查 llama-bench 是否存在且可执行
    if [ ! -x "./llama-bench" ]; then
        echo "错误: 在 $BIN_DIR 下未找到可执行的 llama-bench" | tee -a "$LOG_FILE"
        popd > /dev/null
        continue
    fi
    # 2. 查找 unsloth_riscv 下所有的 .gguf 文件
    # 使用 find 命令递归查找
    find "$MODEL_ROOT_DIR" -type f -name "Qwen3*" | sort | while read model_path; do
        model_name=$(basename "$model_path")
        
        # 3. 线程数循环：从 64 开始，每次加 1，直到 64
        for t in {64..64..1}; do
            
            # 4. 每个配置重复执行两次 (Run 1, Run 2)
            for run_idx in 1 2; do
                echo "-------------------------------------------" | tee -a "$LOG_FILE"
                echo "执行时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
                echo "测试配置: [变体: $variant] [模型: $model_name] [线程: $t] [第 ${run_idx} 次运行]" | tee -a "$LOG_FILE"
                echo "命令: ./llama-bench -m $model_path -t $t" >> "$LOG_FILE"
                
                # === 执行测试 ===
                # 这里的 stderr (2>&1) 也重定向到日志中，以便捕获错误信息
                ./llama-bench -m "$model_path" -t "$t" >> "$LOG_FILE" 2>&1
                
                # 检查上一条命令是否执行成功
                if [ $? -eq 0 ]; then
                    echo "状态: 完成" | tee -a "$LOG_FILE"
                else
                    echo "状态: 失败 (请查看日志详情)" | tee -a "$LOG_FILE"
                fi
                
                # 可选：如果你希望两次运行之间稍微间隔一下，可以取消下面这行的注释
                # sleep 1 
            done
            
        done
    done
    
    # 返回之前的目录
    popd > /dev/null

done

echo "===========================================" | tee -a "$LOG_FILE"
echo "所有测试已结束。" | tee -a "$LOG_FILE"
