#!/bin/bash

riscv64-linux-gnu-g++ -std=c++17 -O2 -fPIC \
-I/home/yab/tvm/device/include \
-I/home/yab/tvm/device/3rdparty/dmlc-core/include \
-I/home/yab/tvm/device/3rdparty/dlpack/include \
-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
-o validator \
new_test.cc lib/libtvm_runtime_pack.o -static \
-L/home/yab/tvm/device/build -ldl -pthread

export LD_LIBRARY_PATH=./lib/lib:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=./lib/lib:${DYLD_LIBRARY_PATH}
# 生成100套算子的参考数据
for i in {0..9}; do
    echo "生成第 ${i} 套算子参考数据..."
    set_dir="op_lib/set_${i}"

    # 确保数据目录存在
    mkdir -p "${set_dir}"

    # 使用native算子库生成参考数据
    ./validator generate "${set_dir}" device

    # 记录完成状态
    echo "完成第 ${i} 套算子参考数据生成"
    echo "------------------------------"
done

echo "所有算子参考数据生成完成！"