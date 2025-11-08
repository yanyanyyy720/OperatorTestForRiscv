#include "data_generator.h"
#include <limits>
#include <random>
#include <cstring>
#include <type_traits>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace tvm::runtime;

// 统一的创建数组接口
tvm::runtime::NDArray DataGenerator::CreateArray(
    const std::vector<int64_t>& shape,
    DLDataType dtype,
    DataGenMode mode,
    DLDevice device,
    double param1,
    double param2
) {
    switch (mode) {
        case DataGenMode::RANDOM:
            return CreateRandomArray(shape, dtype, device, param1, param2);
        case DataGenMode::ZERO:
            return CreateZeroArray(shape, dtype, device);
        case DataGenMode::CONSTANT:
            return CreateConstantArray(shape, dtype, param1, device);
        case DataGenMode::NORMAL:
            return CreateNormalArray(shape, dtype, param1, param2, device);
        case DataGenMode::SEQUENTIAL:
            return CreateSequentialArray(shape, dtype, param1, device);
        default:
            throw std::runtime_error("Unsupported data generation mode");
    }
}

// 内部实现：创建随机数组（带上下限）
tvm::runtime::NDArray DataGenerator::CreateRandomArray(
    const std::vector<int64_t>& shape,
    DLDataType dtype,
    DLDevice device,
    double min_val,
    double max_val
) {
    auto arr = tvm::runtime::NDArray::Empty(shape, dtype, device);
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;

    std::random_device rd;
    std::mt19937 gen(rd());

    switch (dtype.code) {
        case kDLFloat:
            if (dtype.bits == 32) {
                float* data = static_cast<float*>(arr->data);
                float min_f = static_cast<float>(min_val);
                float max_f = static_cast<float>(max_val);
                std::uniform_real_distribution<float> dis(min_f, max_f);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            } else if (dtype.bits == 16) {
                uint16_t* data = static_cast<uint16_t*>(arr->data);
                // 对于 float16，使用 float 生成然后转换
                std::vector<float> temp_data(num_elements);
                float min_f = static_cast<float>(min_val);
                float max_f = static_cast<float>(max_val);
                std::uniform_real_distribution<float> dis(min_f, max_f);
                for (size_t i = 0; i < num_elements; ++i) {
                    temp_data[i] = dis(gen);
                }

                // 转换 float 到 float16
                for (size_t i = 0; i < num_elements; ++i) {
                    // 简化转换，实际应用中应使用专门的 float16 转换函数
                    data[i] = static_cast<uint16_t>(temp_data[i] * 1000);
                }
            } else if (dtype.bits == 64) {
                double* data = static_cast<double*>(arr->data);
                std::uniform_real_distribution<double> dis(min_val, max_val);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            }
            break;

        case kDLInt:
            if (dtype.bits == 64) {
                int64_t* data = static_cast<int64_t*>(arr->data);
                int64_t min_i = static_cast<int64_t>(min_val);
                int64_t max_i = static_cast<int64_t>(max_val);
                std::uniform_int_distribution<int64_t> dis(min_i, max_i);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            } else if (dtype.bits == 32) {
                int32_t* data = static_cast<int32_t*>(arr->data);
                int32_t min_i = static_cast<int32_t>(min_val);
                int32_t max_i = static_cast<int32_t>(max_val);
                std::uniform_int_distribution<int32_t> dis(min_i, max_i);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            } else if (dtype.bits == 16) {
                int16_t* data = static_cast<int16_t*>(arr->data);
                int16_t min_i = static_cast<int16_t>(min_val);
                int16_t max_i = static_cast<int16_t>(max_val);
                std::uniform_int_distribution<int16_t> dis(min_i, max_i);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            } else if (dtype.bits == 8) {
                int8_t* data = static_cast<int8_t*>(arr->data);
                int8_t min_i = static_cast<int8_t>(min_val);
                int8_t max_i = static_cast<int8_t>(max_val);
                std::uniform_int_distribution<int16_t> dis(min_i, max_i);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = static_cast<int8_t>(dis(gen));
                }
            }
            break;

        case kDLUInt:
            if (dtype.bits == 8) {
                uint8_t* data = static_cast<uint8_t*>(arr->data);
                uint8_t min_u = static_cast<uint8_t>(min_val);
                uint8_t max_u = static_cast<uint8_t>(max_val);
                std::uniform_int_distribution<uint16_t> dis(min_u, max_u);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = static_cast<uint8_t>(dis(gen));
                }
            } else if (dtype.bits == 16) {
                uint16_t* data = static_cast<uint16_t*>(arr->data);
                uint16_t min_u = static_cast<uint16_t>(min_val);
                uint16_t max_u = static_cast<uint16_t>(max_val);
                std::uniform_int_distribution<uint16_t> dis(min_u, max_u);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            } else if (dtype.bits == 32) {
                uint32_t* data = static_cast<uint32_t*>(arr->data);
                uint32_t min_u = static_cast<uint32_t>(min_val);
                uint32_t max_u = static_cast<uint32_t>(max_val);
                std::uniform_int_distribution<uint32_t> dis(min_u, max_u);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            } else if (dtype.bits == 64) {
                uint64_t* data = static_cast<uint64_t*>(arr->data);
                uint64_t min_u = static_cast<uint64_t>(min_val);
                uint64_t max_u = static_cast<uint64_t>(max_val);
                std::uniform_int_distribution<uint64_t> dis(min_u, max_u);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            }
            break;

        default:
            throw std::runtime_error("Unsupported data type for random array");
    }
    return arr;
}

// 内部实现：创建零数组
tvm::runtime::NDArray DataGenerator::CreateZeroArray(
    const std::vector<int64_t>& shape,
    DLDataType dtype,
    DLDevice device
) {
    auto arr = tvm::runtime::NDArray::Empty(shape, dtype, device);
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;

    size_t type_size = (dtype.bits + 7) / 8;
    memset(arr->data, 0, num_elements * type_size);

    return arr;
}

// 内部实现：创建常量数组
tvm::runtime::NDArray DataGenerator::CreateConstantArray(
    const std::vector<int64_t>& shape,
    DLDataType dtype,
    double value,
    DLDevice device
) {
    auto arr = tvm::runtime::NDArray::Empty(shape, dtype, device);
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;

    switch (dtype.code) {
        case kDLFloat:
            if (dtype.bits == 32) {
                float* data = static_cast<float*>(arr->data);
                float fvalue = static_cast<float>(value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = fvalue;
                }
            } else if (dtype.bits == 16) {
                uint16_t* data = static_cast<uint16_t*>(arr->data);
                // 简化处理，实际应用中应使用专门的 float16 转换
                uint16_t uvalue = static_cast<uint16_t>(value * 1000);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = uvalue;
                }
            } else if (dtype.bits == 64) {
                double* data = static_cast<double*>(arr->data);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = value;
                }
            }
            break;

        case kDLInt:
            if (dtype.bits == 64) {
                int64_t* data = static_cast<int64_t*>(arr->data);
                int64_t ivalue = static_cast<int64_t>(value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = ivalue;
                }
            } else if (dtype.bits == 32) {
                int32_t* data = static_cast<int32_t*>(arr->data);
                int32_t ivalue = static_cast<int32_t>(value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = ivalue;
                }
            } else if (dtype.bits == 16) {
                int16_t* data = static_cast<int16_t*>(arr->data);
                int16_t ivalue = static_cast<int16_t>(value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = ivalue;
                }
            } else if (dtype.bits == 8) {
                int8_t* data = static_cast<int8_t*>(arr->data);
                int8_t ivalue = static_cast<int8_t>(value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = ivalue;
                }
            }
            break;

        case kDLUInt:
            if (dtype.bits == 8) {
                uint8_t* data = static_cast<uint8_t*>(arr->data);
                uint8_t uvalue = static_cast<uint8_t>(value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = uvalue;
                }
            } else if (dtype.bits == 16) {
                uint16_t* data = static_cast<uint16_t*>(arr->data);
                uint16_t uvalue = static_cast<uint16_t>(value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = uvalue;
                }
            } else if (dtype.bits == 32) {
                uint32_t* data = static_cast<uint32_t*>(arr->data);
                uint32_t uvalue = static_cast<uint32_t>(value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = uvalue;
                }
            } else if (dtype.bits == 64) {
                uint64_t* data = static_cast<uint64_t*>(arr->data);
                uint64_t uvalue = static_cast<uint64_t>(value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = uvalue;
                }
            }
            break;

        default:
            throw std::runtime_error("Unsupported data type for constant array");
    }
    return arr;
}

// 内部实现：创建正态分布数组
tvm::runtime::NDArray DataGenerator::CreateNormalArray(
    const std::vector<int64_t>& shape,
    DLDataType dtype,
    double mean,
    double stddev,
    DLDevice device
) {
    auto arr = tvm::runtime::NDArray::Empty(shape, dtype, device);
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;

    // 只支持浮点类型的正态分布
    if (dtype.code != kDLFloat) {
        throw std::runtime_error("Normal distribution only supported for floating point types");
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    if (dtype.bits == 32) {
        float* data = static_cast<float*>(arr->data);
        std::normal_distribution<float> dis(mean, stddev);
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = dis(gen);
        }
    } else if (dtype.bits == 64) {
        double* data = static_cast<double*>(arr->data);
        std::normal_distribution<double> dis(mean, stddev);
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = dis(gen);
        }
    } else if (dtype.bits == 16) {
        // 对于 float16，使用 float 生成然后转换
        std::vector<float> temp_data(num_elements);
        std::normal_distribution<float> dis(mean, stddev);
        for (size_t i = 0; i < num_elements; ++i) {
            temp_data[i] = dis(gen);
        }

        // 转换 float 到 float16
        uint16_t* data = static_cast<uint16_t*>(arr->data);
        for (size_t i = 0; i < num_elements; ++i) {
            // 简化转换，实际应用中应使用专门的 float16 转换函数
            data[i] = static_cast<uint16_t>(temp_data[i] * 1000);
        }
    } else {
        throw std::runtime_error("Unsupported floating point precision for normal distribution");
    }

    return arr;
}

// 内部实现：创建顺序数组（非随机输出）
tvm::runtime::NDArray DataGenerator::CreateSequentialArray(
    const std::vector<int64_t>& shape,
    DLDataType dtype,
    double start_value,
    DLDevice device
) {
    auto arr = tvm::runtime::NDArray::Empty(shape, dtype, device);
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;

    switch (dtype.code) {
        case kDLFloat:
            if (dtype.bits == 32) {
                float* data = static_cast<float*>(arr->data);
                float current = static_cast<float>(start_value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1.0f;
                }
            } else if (dtype.bits == 16) {
                uint16_t* data = static_cast<uint16_t*>(arr->data);
                uint16_t current = static_cast<uint16_t>(start_value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1;
                }
            } else if (dtype.bits == 64) {
                double* data = static_cast<double*>(arr->data);
                double current = start_value;
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1.0;
                }
            }
            break;

        case kDLInt:
            if (dtype.bits == 64) {
                int64_t* data = static_cast<int64_t*>(arr->data);
                int64_t current = static_cast<int64_t>(start_value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1;
                }
            } else if (dtype.bits == 32) {
                int32_t* data = static_cast<int32_t*>(arr->data);
                int32_t current = static_cast<int32_t>(start_value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1;
                }
            } else if (dtype.bits == 16) {
                int16_t* data = static_cast<int16_t*>(arr->data);
                int16_t current = static_cast<int16_t>(start_value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1;
                }
            } else if (dtype.bits == 8) {
                int8_t* data = static_cast<int8_t*>(arr->data);
                int8_t current = static_cast<int8_t>(start_value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1;
                }
            }
            break;

        case kDLUInt:
            if (dtype.bits == 8) {
                uint8_t* data = static_cast<uint8_t*>(arr->data);
                uint8_t current = static_cast<uint8_t>(start_value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1;
                }
            } else if (dtype.bits == 16) {
                uint16_t* data = static_cast<uint16_t*>(arr->data);
                uint16_t current = static_cast<uint16_t>(start_value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1;
                }
            } else if (dtype.bits == 32) {
                uint32_t* data = static_cast<uint32_t*>(arr->data);
                uint32_t current = static_cast<uint32_t>(start_value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1;
                }
            } else if (dtype.bits == 64) {
                uint64_t* data = static_cast<uint64_t*>(arr->data);
                uint64_t current = static_cast<uint64_t>(start_value);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = current;
                    current += 1;
                }
            }
            break;

        default:
            throw std::runtime_error("Unsupported data type for sequential array");
    }
    return arr;
}

// 从文件加载数组
tvm::runtime::NDArray DataGenerator::LoadFromFile(
    const std::string& filename,
    const std::vector<int64_t>& shape,
    DLDataType dtype,
    DLDevice device
) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    auto arr = tvm::runtime::NDArray::Empty(shape, dtype, device);
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;

    size_t element_size = (dtype.bits + 7) / 8;
    size_t total_size = num_elements * element_size;

    file.read(static_cast<char*>(arr->data), total_size);

    if (!file) {
        throw std::runtime_error("Error reading data from file: " + filename);
    }

    file.close();
    return arr;
}

// 保存数组到文件
void DataGenerator::SaveToFile(
    const tvm::runtime::NDArray& arr,
    const std::string& filename
) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }

    // 获取数组信息
    const DLTensor* tensor = arr.operator->();
    size_t num_elements = 1;
    for (int64_t i = 0; i < tensor->ndim; ++i) {
        num_elements *= tensor->shape[i];
    }

    size_t element_size = (tensor->dtype.bits + 7) / 8;
    size_t total_size = num_elements * element_size;

    file.write(static_cast<const char*>(tensor->data), total_size);

    if (!file) {
        throw std::runtime_error("Error writing data to file: " + filename);
    }

    file.close();
}

// 保存数组信息和数据到文本文件（便于查看）
void DataGenerator::SaveToTextFile(
    const tvm::runtime::NDArray& arr,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }

    const DLTensor* tensor = arr.operator->();

    // 写入数组信息
    file << "Shape: ";
    for (int64_t i = 0; i < tensor->ndim; ++i) {
        file << tensor->shape[i];
        if (i < tensor->ndim - 1) file << " x ";
    }
    file << "\n";

    file << "DataType: code=" << tensor->dtype.code << ", bits=" << tensor->dtype.bits
         << ", lanes=" << tensor->dtype.lanes << "\n";
    file << "Data:\n";

    size_t num_elements = 1;
    for (int64_t i = 0; i < tensor->ndim; ++i) {
        num_elements *= tensor->shape[i];
    }

    // 写入数据（简化版本，只支持基本类型）
    switch (tensor->dtype.code) {
        case kDLFloat:
            if (tensor->dtype.bits == 32) {
                float* data = static_cast<float*>(tensor->data);
                for (size_t i = 0; i < num_elements; ++i) {
                    file << data[i] << " ";
                    if ((i + 1) % tensor->shape[tensor->ndim - 1] == 0) file << "\n";
                }
            }
            break;
        case kDLInt:
            if (tensor->dtype.bits == 32) {
                int32_t* data = static_cast<int32_t*>(tensor->data);
                for (size_t i = 0; i < num_elements; ++i) {
                    file << data[i] << " ";
                    if ((i + 1) % tensor->shape[tensor->ndim - 1] == 0) file << "\n";
                }
            }
            break;
        default:
            file << "[Data type not supported for text output]";
            break;
    }

    file.close();
}

// 比较两个数组是否相等（用于验证）
bool DataGenerator::CompareArrays(
    const tvm::runtime::NDArray& arr1,
    const tvm::runtime::NDArray& arr2,
    double tolerance
) {
    const DLTensor* tensor1 = arr1.operator->();
    const DLTensor* tensor2 = arr2.operator->();

    // 检查形状是否相同
    if (tensor1->ndim != tensor2->ndim) {
        return false;
    }

    for (int64_t i = 0; i < tensor1->ndim; ++i) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            return false;
        }
    }

    // 检查数据类型是否相同
    if (tensor1->dtype.code != tensor2->dtype.code ||
        tensor1->dtype.bits != tensor2->dtype.bits) {
        return false;
    }

    size_t num_elements = 1;
    for (int64_t i = 0; i < tensor1->ndim; ++i) {
        num_elements *= tensor1->shape[i];
    }

    // 比较数据
    switch (tensor1->dtype.code) {
        case kDLFloat:
            if (tensor1->dtype.bits == 32) {
                float* data1 = static_cast<float*>(tensor1->data);
                float* data2 = static_cast<float*>(tensor2->data);
                for (size_t i = 0; i < num_elements; ++i) {
                    if (std::abs(data1[i] - data2[i]) > tolerance) {
                        return false;
                    }
                }
            }
            break;
        case kDLInt:
            if (tensor1->dtype.bits == 32) {
                int32_t* data1 = static_cast<int32_t*>(tensor1->data);
                int32_t* data2 = static_cast<int32_t*>(tensor2->data);
                for (size_t i = 0; i < num_elements; ++i) {
                    if (data1[i] != data2[i]) {
                        return false;
                    }
                }
            }
            break;
        default:
            // 对于不支持的类型，进行字节级比较
            return memcmp(tensor1->data, tensor2->data,
                         num_elements * ((tensor1->dtype.bits + 7) / 8)) == 0;
    }

    return true;
}

// 计算数组大小 (MB)
double DataGenerator::CalculateArraySizeMB(
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

// 打印数组信息
void DataGenerator::PrintArrayInfo(
    const tvm::runtime::NDArray& arr,
    const std::string& name
) {
    const DLTensor* tensor = arr.operator->();

    std::cout << "Array '" << name << "' Info:\n";
    std::cout << "  Shape: ";
    for (int64_t i = 0; i < tensor->ndim; ++i) {
        std::cout << tensor->shape[i];
        if (i < tensor->ndim - 1) std::cout << " x ";
    }
    std::cout << "\n";

    std::cout << "  DataType: code=" << tensor->dtype.code
              << ", bits=" << tensor->dtype.bits
              << ", lanes=" << tensor->dtype.lanes << "\n";

    std::cout << "  Size: " << CalculateArraySizeMB(arr) << " MB\n";
}

// 重载版本：从形状和数据类型计算大小
double DataGenerator::CalculateArraySizeMB(const tvm::runtime::NDArray& arr) {
    const DLTensor* tensor = arr.operator->();
    std::vector<int64_t> shape;
    for (int64_t i = 0; i < tensor->ndim; ++i) {
        shape.push_back(tensor->shape[i]);
    }
    return CalculateArraySizeMB(shape, tensor->dtype);
}