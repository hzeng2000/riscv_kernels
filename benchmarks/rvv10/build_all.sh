#!/bin/bash
# build_all.sh - 执行所有子目录中的build.sh脚本
#
# 用法: ./build_all.sh [kernel_name]
#   kernel_name: 可选，指定只编译某个算子

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

KERNEL_FILTER="${1:-all}"

echo "=============================================="
echo "  Building RVV 1.0 Kernels"
echo "=============================================="
echo ""

SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

# 遍历当前目录下的所有子目录
for dir in */; do
    # 移除末尾的斜杠
    dir="${dir%/}"
    
    # 如果指定了过滤器，只处理匹配的目录
    if [ "$KERNEL_FILTER" != "all" ] && [ "$KERNEL_FILTER" != "$dir" ]; then
        continue
    fi
    
    # 检查是否是目录且包含build.sh
    if [ -d "$dir" ] && [ -f "$dir/build.sh" ]; then
        echo "=== Building $dir ==="
        cd "$dir"
        
        # 执行build.sh
        if bash build.sh; then
            echo "    ✓ Success: $dir"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "    ✗ Failed: $dir"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        
        cd "$SCRIPT_DIR"
        echo ""
    elif [ -d "$dir" ]; then
        echo "=== Skipping $dir (no build.sh found) ==="
        SKIP_COUNT=$((SKIP_COUNT + 1))
        echo ""
    fi
done

echo "=============================================="
echo "  Build Complete!"
echo "=============================================="
echo "Success: $SUCCESS_COUNT | Failed: $FAIL_COUNT | Skipped: $SKIP_COUNT"
echo "=============================================="

