#!/bin/bash

#riscv64-linux-gnu-g++ -std=c++17 -O2 -fPIC \
#-I/home/yab/tvm/device/include \
#-I/home/yab/tvm/device/3rdparty/dmlc-core/include \
#-I/home/yab/tvm/device/3rdparty/dlpack/include \
#-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
#-o validator \
#new_test.cc lib/libtvm_runtime_pack.o -static \
#-L/home/yab/tvm/device/build -ldl -pthread

#export LD_LIBRARY_PATH=./lib/lib:${LD_LIBRARY_PATH}
#export DYLD_LIBRARY_PATH=./lib/lib:${DYLD_LIBRARY_PATH}
# 验证100套算子的正确性
for i in {0..5}; do
    echo "验证第 ${i} 套算子..."
    set_dir="op_lib/set_${i}"

    # 检查set目录是否存在
    if [ ! -d "${set_dir}" ]; then
        echo "错误：${set_dir} 目录不存在"
        continue
    fi

    # 检查数据目录是否存在
    data_dir="${set_dir}"
    if [ ! -d "${data_dir}" ]; then
        echo "错误：${set_dir} 中没有找到参考数据"
        continue
    fi

    # 使用device算子库进行验证
    ./validator validate "${set_dir}" device

    # 记录完成状态
    echo "完成第 ${i} 套算子验证"
    echo "验证结果保存在: ${set_dir}/validation_results.csv"
    echo "------------------------------"
done

echo "所有算子验证完成！"
