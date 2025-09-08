#include "operator_executor.h"
#include <tvm/runtime/registry.h>
#include <stdexcept>
#include <random>
#include <cstring>
#include <cmath>

using namespace tvm::runtime;

DLDevice kCPUDevice{DLDeviceType::kDLCPU, 0};

// 实现辅助函数
DLDataType OperatorExecutor::StrToDLDataType(const std::string& dtype_str) {
    if (dtype_str == "float32") {
        return DLDataType{kDLFloat, 32, 1};
    } else if (dtype_str == "float16") {
        return DLDataType{kDLFloat, 16, 1};
    } else if (dtype_str == "bfloat16") {
        return DLDataType{kDLBfloat, 16, 1};
    } else if (dtype_str == "int32") {
        return DLDataType{kDLInt, 32, 1};
    } else if (dtype_str == "int64") {
        return DLDataType{kDLInt, 64, 1};
    } else if (dtype_str == "int8") {
        return DLDataType{kDLInt, 8, 1};
    } else if (dtype_str == "uint8") {
        return DLDataType{kDLUInt, 8, 1};
    } else {
        throw std::runtime_error("Unsupported data type: " + dtype_str);
    }
}

tvm::runtime::NDArray OperatorExecutor::CreateRandomArray(
    const std::vector<int64_t>& shape,
    DLDataType dtype
) {
    auto arr = tvm::runtime::NDArray::Empty(shape, dtype, kCPUDevice);
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;

    std::random_device rd;
    std::mt19937 gen(rd());

    switch (dtype.code) {
        case kDLFloat:
            if (dtype.bits == 32) {
                float* data = static_cast<float*>(arr->data);
                std::uniform_real_distribution<float> dis(0.0f, 1.0f);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            } else if (dtype.bits == 16) {
                uint16_t* data = static_cast<uint16_t*>(arr->data);
                std::uniform_int_distribution<uint16_t> dis(0, std::numeric_limits<uint16_t>::max());
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            }
            break;

        case kDLInt:
            if (dtype.bits == 64) {
                int64_t* data = static_cast<int64_t*>(arr->data);
                std::uniform_int_distribution<int64_t> dis(0, 100);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            } else if (dtype.bits == 32) {
                int32_t* data = static_cast<int32_t*>(arr->data);
                std::uniform_int_distribution<int32_t> dis(0, 100);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            } else if (dtype.bits == 8) {
                int8_t* data = static_cast<int8_t*>(arr->data);
                std::uniform_int_distribution<int16_t> dis(-128, 127);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = static_cast<int8_t>(dis(gen));
                }
            }
            break;

        case kDLUInt:
            if (dtype.bits == 8) {
                uint8_t* data = static_cast<uint8_t*>(arr->data);
                std::uniform_int_distribution<uint16_t> dis(0, 255);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = static_cast<uint8_t>(dis(gen));
                }
            }
            break;

        default:
            throw std::runtime_error("Unsupported data type for random array");
    }
    return arr;
}

tvm::runtime::NDArray OperatorExecutor::CreateZeroArray(
    const std::vector<int64_t>& shape,
    DLDataType dtype
) {
    auto arr = tvm::runtime::NDArray::Empty(shape, dtype, kCPUDevice);
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;

    size_t type_size = (dtype.bits + 7) / 8;
    memset(arr->data, 0, num_elements * type_size);

    return arr;
}

double OperatorExecutor::calculate_array_size_mb(
    const std::vector<int64_t>& shape,
    DLDataType dtype
) {
    // 计算元素数量
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }

    // 计算每个元素的字节数
    size_t element_size = (dtype.bits + 7) / 8; // 向上取整

    // 计算总字节数并转换为 MB
    double total_bytes = static_cast<double>(num_elements) * element_size;
    return total_bytes / (1024.0 * 1024.0); // 转换为 MB
}

OperatorExecutor::OperatorExecutor(const std::string& lib_path) {
    Module mod_factory = Module::LoadFromFile(lib_path);
    gmod_ = mod_factory.GetFunction("default")(kCPUDevice);
}

void OperatorExecutor::set_inputs(const std::vector<std::tuple<std::string, std::vector<int64_t>, std::string>>& input_configs) {
    input_configs_ = input_configs; // 仅存储配置信息

    auto set_input = gmod_.GetFunction("set_input");

    for (const auto& [name, shape, dtype_str] : input_configs) {
        DLDataType dtype = StrToDLDataType(dtype_str);
        auto arr = CreateRandomArray(shape, dtype);
        set_input(name, arr);
        // 不存储数组，立即释放
    }
}

void OperatorExecutor::set_outputs(const std::vector<std::tuple<std::vector<int64_t>, std::string>>& output_configs) {
    output_configs_ = output_configs; // 仅存储配置信息
}

void OperatorExecutor::run() {
    auto run_func = gmod_.GetFunction("run");
    run_func();

    auto get_output = gmod_.GetFunction("get_output");

    // 仅在需要时创建输出数组
    for (size_t i = 0; i < output_configs_.size(); ++i) {
        const auto& [shape, dtype_str] = output_configs_[i];
        DLDataType dtype = StrToDLDataType(dtype_str);
        auto arr = CreateZeroArray(shape, dtype);
        get_output(i, arr);
        // 不存储数组，立即释放
    }
}

// 计算输入输出大小的方法
double OperatorExecutor::get_total_input_size_mb() const {
    double total_size = 0.0;
    for (const auto& [name, shape, dtype_str] : input_configs_) {
        DLDataType dtype = StrToDLDataType(dtype_str);
        total_size += calculate_array_size_mb(shape, dtype);
    }
    return total_size;
}

double OperatorExecutor::get_total_output_size_mb() const {
    double total_size = 0.0;
    for (const auto& [shape, dtype_str] : output_configs_) {
        DLDataType dtype = StrToDLDataType(dtype_str);
        total_size += calculate_array_size_mb(shape, dtype);
    }
    return total_size;
}

double OperatorExecutor::get_input_size_mb(const std::string& name) const {
    for (const auto& [input_name, shape, dtype_str] : input_configs_) {
        if (input_name == name) {
            DLDataType dtype = StrToDLDataType(dtype_str);
            return calculate_array_size_mb(shape, dtype);
        }
    }
    return 0.0;
}

double OperatorExecutor::get_output_size_mb(size_t index) const {
    if (index < output_configs_.size()) {
        const auto& [shape, dtype_str] = output_configs_[index];
        DLDataType dtype = StrToDLDataType(dtype_str);
        return calculate_array_size_mb(shape, dtype);
    }
    return 0.0;
}

double OperatorExecutor::get_temporary_memory_mb() const {
    return get_total_input_size_mb() + get_total_output_size_mb();
}