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
    LDFLAGS :=
    INCLUDES :=
endif

CXXFLAGS := -std=c++17 -O2 -fPIC -Wall -Wextra
INCLUDES += -I$(TVM_ROOT)/include \
            -I$(TVM_ROOT)/3rdparty/dmlc-core/include \
            -I$(TVM_ROOT)/3rdparty/dlpack/include \
            -I./include \
            -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>
LDFLAGS += -L$(TVM_ROOT)/build
LIBS := -ldl -pthread -static

# 源文件和目标文件
SRC_DIR := src
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,obj/$(TARGET)/%.o,$(SRCS))
EXEC := validator-$(TARGET)

# 编译规则
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ ./lib/$(RUNTIME_LIB) $(LIBS)

obj/$(TARGET)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf obj validator-riscv validator-native

.PHONY: all clean
