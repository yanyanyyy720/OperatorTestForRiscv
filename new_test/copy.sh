#!/bin/bash
# TVM文件复制和重组脚本 - Shell版本

TARGET_DIR=$1
NEW_DIR=$2

if [ -z "$TARGET_DIR" ] || [ -z "$NEW_DIR" ]; then
    echo "用法: $0 <源目录> <目标目录>"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 源目录 $TARGET_DIR 不存在"
    exit 1
fi

echo "开始处理目录重组..."

# 查找所有run目录
for run_dir in "$TARGET_DIR"/run_*/; do
    if [ ! -d "$run_dir" ]; then
        continue
    fi
    
    # 提取run编号
    run_name=$(basename "$run_dir")
    run_i=${run_name#run_}
    
    echo "处理 $run_name..."
    
    # 处理rv和rvv
    for arch in rv rvv; do
        # 检查output目录是否存在
        output_arch_dir="$run_dir/output/$arch"
        if [ ! -d "$output_arch_dir" ]; then
            echo "  跳过 $arch - 目录不存在"
            continue
        fi
        
        # 处理每个参数规模
        for param_size in large medium small; do
            # 创建目标目录
            target_dir="$NEW_DIR/run_$run_i/$arch/$param_size"
            mkdir -p "$target_dir"
            
            # 复制cfg文件
            cfg_source="$output_arch_dir/$param_size/tvm_output/*.cfg"
            if ls $cfg_source 1> /dev/null 2>&1; then
                cp $cfg_source "$target_dir/" 2>/dev/null
                echo "  复制CFG文件到 $target_dir"
            fi
            
            # 复制json文件
            json_source="$run_dir/tvm/rv/$param_size/*.json"
            if ls $json_source 1> /dev/null 2>&1; then
                cp $json_source "$target_dir/" 2>/dev/null
                echo "  复制JSON文件到 $target_dir"
            fi
        done
    done
done

echo "完成！"
