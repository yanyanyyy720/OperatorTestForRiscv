#!/bin/bash

g++ -std=c++17 -O2 -fPIC \
-I/home/yab/tvm/native/include \
-I/home/yab/tvm/native/3rdparty/dmlc-core/include \
-I/home/yab/tvm/native/3rdparty/dlpack/include \
-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
-o validator \
test_performerce.cc lib/libtvm_runtime_pack_native.o -static \
-L/home/yab/tvm/native/build -ldl -pthread

export LD_LIBRARY_PATH=/home/yab/tvm/native/build:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=/home/yab/tvm/native/build:${DYLD_LIBRARY_PATH}

./validator .