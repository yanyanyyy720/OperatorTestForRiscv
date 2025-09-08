#!/bin/bash

# 批量测试脚本
# 用法: ./run_tests.sh <base_dir> <mode> <test_mode> <max_sets> [start_set]
#   base_dir: 基础目录
#   mode: 算子库类型 (native 或 device)
#   test_mode: 测试模式 (normal 或 temperature)
#   max_sets: 最大集合数量
#   start_set: 起始集合索引 (可选，默认为0)

if [ $# -lt 4 ]; then
    echo "用法: $0 <base_dir> <mode> <test_mode> <max_sets> [start_set]"
    exit 1
fi

BASE_DIR=$1
MODE=$2
TEST_MODE=$3
MAX_SETS=$4
START_SET=${5:-0}

for ((i=$START_SET; i<$MAX_SETS; i++)); do
    OP_LIB_DIR="$BASE_DIR/set_${i}/$MODE"

    # 根据测试模式选择输出目录
    if [ "$TEST_MODE" == "temperature" ]; then
        OUTPUT_DIR="$BASE_DIR/set_${i}/temperature_output"
    else
        OUTPUT_DIR="$BASE_DIR/set_${i}/output"
    fi

    if [ ! -d "$OP_LIB_DIR" ]; then
        echo "跳过不存在的算子库: $OP_LIB_DIR"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    echo "===== 处理算子库集合 $i (模式: $TEST_MODE) ====="
    for cfg_file in "$OP_LIB_DIR"/*.cfg; do
        ./validator-riscv "$TEST_MODE" "$cfg_file" "$OUTPUT_DIR"
    done
done