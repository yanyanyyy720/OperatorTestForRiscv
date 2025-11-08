# 目标平台设置 (riscv 或 native)
TARGET ?= riscv

# 编译器设置
ifeq ($(TARGET),riscv)
    CXX := riscv64-linux-gnu-g++
    TVM_ROOT := /home/yab/tvm/device
    RUNTIME_LIB := libtvm_runtime_pack.o
else
    CXX := g++
    TVM_ROOT := /home/yab/tvm/native
    RUNTIME_LIB := libtvm_runtime_pack_native.o
endif

# 公共编译选项
CXXFLAGS := -std=c++17 -O2 -fPIC -Wall -Wextra
INCLUDES := -I$(TVM_ROOT)/include \
            -I$(TVM_ROOT)/3rdparty/dmlc-core/include \
            -I$(TVM_ROOT)/3rdparty/dlpack/include \
            -I./include \
            -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>
LDFLAGS := -L$(TVM_ROOT)/build
LIBS := -ldl -pthread -static

# 目录结构
SRC_DIR := src
OBJ_DIR := obj/$(TARGET)
LIB_DIR := lib
EXEC := validator-$(TARGET)

# 源文件和目标文件
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

# 构建目标
all: $(EXEC)

$(EXEC): $(OBJS) | $(LIB_DIR)/$(RUNTIME_LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIB_DIR)/$(RUNTIME_LIB) $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 创建目录
$(OBJ_DIR):
	@mkdir -p $@

# 清理
clean:
	rm -rf obj validator-riscv validator-native

.PHONY: all clean