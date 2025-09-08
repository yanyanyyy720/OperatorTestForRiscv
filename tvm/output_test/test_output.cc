#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <random>
#include <cmath>
#include <algorithm>
#include <map>
#include <sstream>
#include <filesystem>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <cctype>
#include <chrono>
namespace fs = std::filesystem;

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

// ======================== 结果数据结构 ========================
struct ValidationResult {
    std::string op_name;
    double similarity; // 相似度指标 (0-1之间，越大越好)
    double max_abs_diff;

    // 新增字段：记录差异数据的位置和值
    int max_diff_group_idx = -1;      // 差异最大的测试组索引
    int max_diff_output_idx = -1;     // 差异最大的输出张量索引
    std::vector<int64_t> max_diff_position; // 差异数据在多维张量中的位置
    double max_diff_ref_value = 0.0;  // 参考输出值
    double max_diff_val_value = 0.0;  // 实际输出值
};

// 新增：比较结果结构体
struct ComparisonResult {
    double similarity;
    double max_abs_diff;
    size_t max_diff_index;   // 最大绝对误差对应的元素在一维数据中的索引
    double ref_value;        // 在最大绝对误差位置的参考值
    double val_value;        // 在最大绝对误差位置的实际值
};

struct OperatorConfig {
    std::string name;
    std::string target;
    std::vector<std::tuple<std::string, std::vector<int64_t>, std::string>> inputs;
    std::vector<std::tuple<std::vector<int64_t>, std::string>> outputs;
};

DLDevice kCPUDevice{DLDeviceType::kDLCPU, 0};

tvm::runtime::Module CreateExecutor(const std::string& lib_path) {
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(lib_path);
    return mod_factory.GetFunction("default")(kCPUDevice);
}

// 辅助函数：去除字符串首尾的空格
std::string trim(const std::string& str) {
    if (str.empty()) return str;
    auto start = str.begin();
    auto end = str.end() - 1;

    // 去除左侧空格
    while (start < end && std::isspace(static_cast<unsigned char>(*start))) {
        ++start;
    }

    // 去除右侧空格
    while (end >= start && std::isspace(static_cast<unsigned char>(*end))) {
        --end;
    }

    return std::string(start, end + 1);
}

OperatorConfig ParseConfig(const std::string& cfg_path) {
    OperatorConfig config;
    std::ifstream file(cfg_path);
    std::string line;
    std::string current_section;

    while (std::getline(file, line)) {
        // 移除行尾的换行符
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // 跳过空行
        if (line.empty()) continue;

        // 跳过注释行
        if (line.front() == '#') continue;

        // 处理section行
        if (line.front() == '[' && line.back() == ']') {
            current_section = line.substr(1, line.size() - 2);
            continue;
        }

        // 分割键值对
        auto delim_pos = line.find('=');
        if (delim_pos == std::string::npos) continue;

        std::string key = trim(line.substr(0, delim_pos));
        std::string value = trim(line.substr(delim_pos + 1));

        // 处理[operator]部分
        if (current_section == "operator") {
            if (key == "name") {
                config.name = value;
            } else if (key == "target") {
                config.target = value;
            }
        }
        // 处理[inputs]部分
        else if (current_section == "inputs") {
            // 解析键名中的索引和类型
            auto last_underscore = key.find_last_of('_');
            if (last_underscore == std::string::npos) continue;

            // 获取索引值
            int idx = -1;
            try {
                idx = std::stoi(key.substr(last_underscore - 1, 1));
            } catch (...) {
                continue;
            }

            // 确保有足够的输入位置
            if (idx < 0 || config.inputs.size() <= static_cast<size_t>(idx)) {
                config.inputs.resize(idx + 1);
            }

            auto& input_tuple = config.inputs[idx];
            std::string suffix = key.substr(last_underscore + 1);

            if (suffix == "name") {
                std::get<0>(input_tuple) = value;
            } else if (suffix == "shape") {
                // 解析形状字符串（逗号分隔的数字）
                std::vector<int64_t> shape;
                std::istringstream shape_stream(value);
                std::string dim_str;

                while (std::getline(shape_stream, dim_str, ',')) {
                    try {
                        shape.push_back(std::stol(trim(dim_str)));
                    } catch (...) {
                        // 忽略无效数值
                    }
                }

                std::get<1>(input_tuple) = shape;
            } else if (suffix == "dtype") {
                std::get<2>(input_tuple) = value;
            }
        }
        // 处理[outputs]部分
        else if (current_section == "outputs") {
            // 解析键名中的索引和类型
            auto last_underscore = key.find_last_of('_');
            if (last_underscore == std::string::npos) continue;

            // 获取索引值
            int idx = -1;
            try {
                idx = std::stoi(key.substr(last_underscore - 1, 1));
            } catch (...) {
                continue;
            }

            // 确保有足够的输出位置
            if (idx < 0 || config.outputs.size() <= static_cast<size_t>(idx)) {
                config.outputs.resize(idx + 1);
            }

            auto& output_tuple = config.outputs[idx];
            std::string suffix = key.substr(last_underscore + 1);

            if (suffix == "shape") {
                // 解析形状字符串（逗号分隔的数字）
                std::vector<int64_t> shape;
                std::istringstream shape_stream(value);
                std::string dim_str;

                while (std::getline(shape_stream, dim_str, ',')) {
                    try {
                        shape.push_back(std::stol(trim(dim_str)));
                    } catch (...) {
                        // 忽略无效数值
                    }
                }

                std::get<0>(output_tuple) = shape;
            } else if (suffix == "dtype") {
                std::get<1>(output_tuple) = value;
            }
        }
    }

    return config;
}

DLDataType StrToDLDataType(const std::string& dtype_str) {
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
    }  else {
        throw std::runtime_error("Unsupported data type: " + dtype_str);
    }
}

tvm::runtime::NDArray CreateRandomArray(
    const std::vector<int64_t>& shape,
    DLDataType dtype = {kDLFloat, 32, 1}
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
                std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = dis(gen);
                }
            } else if (dtype.bits == 16) {
                uint16_t* data = static_cast<uint16_t*>(arr->data);
                std::uniform_int_distribution<uint16_t> dis(-1.0, 1.0);
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
                std::uniform_int_distribution<int16_t> dis(-128, 127); // 16位以兼容
                for (size_t i = 0; i < num_elements; ++i) {
                    data[i] = static_cast<int8_t>(dis(gen));
                }
            }
            break;

        case kDLUInt:
            if (dtype.bits == 8) {
                uint8_t* data = static_cast<uint8_t*>(arr->data);
                std::uniform_int_distribution<uint16_t> dis(0, 255); // 16位以兼容
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

tvm::runtime::NDArray CreateZeroArray(
    const std::vector<int64_t>& shape,
    DLDataType dtype = {kDLFloat, 32, 1}
) {
    auto arr = tvm::runtime::NDArray::Empty(shape, dtype, kCPUDevice);
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;

    size_t type_size = (dtype.bits + 7) / 8; // 按字节计算
    memset(arr->data, 0, num_elements * type_size);

    return arr;
}

//// 新增：将一维索引转换为多维位置
//std::vector<int64_t> GetMultidimensionalPosition(
//    const std::vector<int64_t>& shape,
//    size_t index
//) {
//    std::vector<int64_t> position;
//    if (shape.empty()) return position;
//
//    position.reserve(shape.size());
//
//    // 反向计算每个维度上的位置
//    for (int dim = shape.size() - 1; dim >= 0; dim--) {
//        int64_t dim_size = shape[dim];
//        position.insert(position.begin(), index % dim_size);
//        index /= dim_size;
//    }
//
//    return position;
//}

// 比较两个NDArray的值并计算相似度
ComparisonResult CalculateSimilarity(
    const tvm::runtime::NDArray& ref,
    const tvm::runtime::NDArray& val
) {
    ComparisonResult result;
    result.similarity = 0.0;
    result.max_abs_diff = 0.0;
    result.max_diff_index = 0;
    result.ref_value = 0.0;
    result.val_value = 0.0;

    // 检查形状是否相同
    if (ref->ndim != val->ndim) {
        return result;
    }

    for (int i = 0; i < ref->ndim; ++i) {
        if (ref->shape[i] != val->shape[i]) {
            return result;
        }
    }

    size_t num_elements = 1;
    for (int i = 0; i < ref->ndim; ++i) {
        num_elements *= ref->shape[i];
    }

    if (ref->dtype.code == kDLFloat && ref->dtype.bits == 32) {
        float* ref_data = static_cast<float*>(ref->data);
        float* val_data = static_cast<float*>(val->data);

        for (size_t i = 0; i < num_elements; ++i) {
            float ref_val = ref_data[i];
            float val_val = val_data[i];
            float diff = std::abs(ref_val - val_val);

            // 更新最大差异信息
            if (diff > result.max_abs_diff) {
                result.max_abs_diff = diff;
                result.max_diff_index = i;
                result.ref_value = ref_val;
                result.val_value = val_val;
            }

            double exp_diff = std::exp(-diff);
            result.similarity += exp_diff / (1 + exp_diff);
        }

        result.similarity /= num_elements;
    }
    else if (ref->dtype.code == kDLInt) {
        if (ref->dtype.bits == 64) {
            int64_t* ref_data = static_cast<int64_t*>(ref->data);
            int64_t* val_data = static_cast<int64_t*>(val->data);

            for (size_t i = 0; i < num_elements; ++i) {
                int64_t ref_val = ref_data[i];
                int64_t val_val = val_data[i];
                int64_t diff = std::abs(ref_val - val_val);

                // 更新最大差异信息
                if (static_cast<double>(diff) > result.max_abs_diff) {
                    result.max_abs_diff = static_cast<double>(diff);
                    result.max_diff_index = i;
                    result.ref_value = static_cast<double>(ref_val);
                    result.val_value = static_cast<double>(val_val);
                }

                // 对于整数类型，使用相同的相似度计算方法
                double exp_diff = std::exp(-static_cast<double>(diff));
                result.similarity += exp_diff / (1 + exp_diff);
            }
            result.similarity /= num_elements;
        } else if (ref->dtype.bits == 32) {
            int32_t* ref_data = static_cast<int32_t*>(ref->data);
            int32_t* val_data = static_cast<int32_t*>(val->data);

            for (size_t i = 0; i < num_elements; ++i) {
                int32_t ref_val = ref_data[i];
                int32_t val_val = val_data[i];
                int32_t diff = std::abs(ref_val - val_val);

                // 更新最大差异信息
                if (static_cast<double>(diff) > result.max_abs_diff) {
                    result.max_abs_diff = static_cast<double>(diff);
                    result.max_diff_index = i;
                    result.ref_value = static_cast<double>(ref_val);
                    result.val_value = static_cast<double>(val_val);
                }

                double exp_diff = std::exp(-static_cast<double>(diff));
                result.similarity += exp_diff / (1 + exp_diff);
            }
            result.similarity /= num_elements;
        } else if (ref->dtype.bits == 8) {
            int8_t* ref_data = static_cast<int8_t*>(ref->data);
            int8_t* val_data = static_cast<int8_t*>(val->data);

            for (size_t i = 0; i < num_elements; ++i) {
                int8_t ref_val = ref_data[i];
                int8_t val_val = val_data[i];
                int8_t diff = std::abs(ref_val - val_val);

                // 更新最大差异信息
                if (static_cast<double>(diff) > result.max_abs_diff) {
                    result.max_abs_diff = static_cast<double>(diff);
                    result.max_diff_index = i;
                    result.ref_value = static_cast<double>(ref_val);
                    result.val_value = static_cast<double>(val_val);
                }

                double exp_diff = std::exp(-static_cast<double>(diff));
                result.similarity += exp_diff / (1 + exp_diff);
            }
            result.similarity /= num_elements;
        }
    }
    else if (ref->dtype.code == kDLUInt && ref->dtype.bits == 8) {
        uint8_t* ref_data = static_cast<uint8_t*>(ref->data);
        uint8_t* val_data = static_cast<uint8_t*>(val->data);

        for (size_t i = 0; i < num_elements; ++i) {
            uint8_t ref_val = ref_data[i];
            uint8_t val_val = val_data[i];
            uint8_t diff = (ref_val > val_val) ? (ref_val - val_val) : (val_val - ref_val);

            // 更新最大差异信息
            if (static_cast<double>(diff) > result.max_abs_diff) {
                result.max_abs_diff = static_cast<double>(diff);
                result.max_diff_index = i;
                result.ref_value = static_cast<double>(ref_val);
                result.val_value = static_cast<double>(val_val);
            }

            double exp_diff = std::exp(-static_cast<double>(diff));
            result.similarity += exp_diff / (1 + exp_diff);
        }
        result.similarity /= num_elements;
    }

    return result;
}

// 保存NDArray到文件
void SaveNDArray(const tvm::runtime::NDArray& arr, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);

    // 保存数据类型
    file.write(reinterpret_cast<const char*>(&arr->dtype), sizeof(DLDataType));

    // 保存维度数
    int32_t ndim = arr->ndim;
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

    // 保存形状
    file.write(reinterpret_cast<const char*>(arr->shape), ndim * sizeof(int64_t));

    // 保存数据
    size_t num_elements = 1;
    for (int i = 0; i < ndim; ++i) {
        num_elements *= arr->shape[i];
    }

    size_t elem_size = (arr->dtype.bits + 7) / 8;
    file.write(reinterpret_cast<const char*>(arr->data), num_elements * elem_size);
}

// 从文件加载NDArray
tvm::runtime::NDArray LoadNDArray(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);

    // 读取数据类型
    DLDataType dtype;
    file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

    // 读取维度数
    int32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    if (ndim < 0 || ndim > 16) {
        throw std::runtime_error("Invalid ndim in NDArray file");
    }

    // 读取形状
    std::vector<int64_t> shape(ndim);
    file.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int64_t));

    // 创建数组
    auto arr = tvm::runtime::NDArray::Empty(shape, dtype, kCPUDevice);

    // 读取数据
    size_t num_elements = 1;
    for (int i = 0; i < ndim; ++i) {
        num_elements *= shape[i];
    }

    size_t elem_size = (dtype.bits + 7) / 8;
    file.read(reinterpret_cast<char*>(arr->data), num_elements * elem_size);

    return arr;
}

// ======================== 生成参考输出模式 ========================
void GenerateReference(
    const OperatorConfig& config,
    const std::string& lib_dir,
    const std::string& data_dir
) {
    const std::string lib_path = lib_dir + "/" + config.name + ".so";
    auto gmod = CreateExecutor(lib_path);
    auto set_input = gmod.GetFunction("set_input");
    auto run = gmod.GetFunction("run");
    auto get_output = gmod.GetFunction("get_output");

    // 创建输出目录
    fs::create_directories(data_dir);

    // 生成10组随机参数
    const int num_groups = 1000;
    for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
        // 创建输入数组
        std::vector<tvm::runtime::NDArray> inputs;
        for (const auto& [name, shape, dtype_str] : config.inputs) {
            DLDataType dtype = StrToDLDataType(dtype_str);
            inputs.push_back(CreateRandomArray(shape, dtype));
        }

        // 创建输出数组
        std::vector<tvm::runtime::NDArray> outputs;
        for (const auto& [shape, dtype_str] : config.outputs) {
            DLDataType dtype = StrToDLDataType(dtype_str);
            outputs.push_back(CreateZeroArray(shape, dtype));
        }

        // 设置输入
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& [name, shape, dtype_str] = config.inputs[i];
            set_input(name, inputs[i]);
        }

        // 执行算子
        run();

        // 获取输出
        for (size_t i = 0; i < outputs.size(); ++i) {
            get_output(i, outputs[i]);
        }

        // 保存输入和输出
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& [name, shape, dtype_str] = config.inputs[i];
            std::string filename = data_dir + "/" + config.name + "_input" +
                                  std::to_string(i) + "_group" + std::to_string(group_idx) + ".bin";
            SaveNDArray(inputs[i], filename);
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            std::string filename = data_dir + "/" + config.name + "_output" +
                                  std::to_string(i) + "_group" + std::to_string(group_idx) + ".bin";
            SaveNDArray(outputs[i], filename);
        }
    }
}

// ======================== 验证模式 ========================
ValidationResult Validate(
    const OperatorConfig& config,
    const std::string& lib_dir,
    const std::string& data_dir
) {
    const std::string lib_path = lib_dir + "/" + config.name + ".so";
    auto gmod = CreateExecutor(lib_path);
    auto set_input = gmod.GetFunction("set_input");
    auto run = gmod.GetFunction("run");
    auto get_output = gmod.GetFunction("get_output");

    ValidationResult result;
    result.op_name = config.name;
    result.similarity = 0.0;
    result.max_abs_diff = 0.0;
    result.max_diff_group_idx = -1;
    result.max_diff_output_idx = -1;

    const int num_groups = 1000;
    int valid_groups = 0;

    for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
        try {
            // 加载输入数组
            std::vector<tvm::runtime::NDArray> inputs;
            for (size_t i = 0; i < config.inputs.size(); ++i) {
                const auto& [name, shape, dtype_str] = config.inputs[i];
                std::string filename = data_dir + "/" + config.name + "_input" +
                                      std::to_string(i) + "_group" + std::to_string(group_idx) + ".bin";
                inputs.push_back(LoadNDArray(filename));
            }

            // 加载参考输出数组
            std::vector<tvm::runtime::NDArray> ref_outputs;
            for (size_t i = 0; i < config.outputs.size(); ++i) {
                std::string filename = data_dir + "/" + config.name + "_output" +
                                      std::to_string(i) + "_group" + std::to_string(group_idx) + ".bin";
                ref_outputs.push_back(LoadNDArray(filename));
            }

            // 设置输入
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto& [name, shape, dtype_str] = config.inputs[i];
                set_input(name, inputs[i]);
            }

            // 创建本地输出数组
            std::vector<tvm::runtime::NDArray> local_outputs;
            for (const auto& [shape, dtype_str] : config.outputs) {
                DLDataType dtype = StrToDLDataType(dtype_str);
                local_outputs.push_back(CreateZeroArray(shape, dtype));
            }

            // 执行算子
            run();

            // 获取输出
            for (size_t i = 0; i < local_outputs.size(); ++i) {
                get_output(i, local_outputs[i]);
            }

            // 比较输出并计算相似度
            double group_similarity = 1.0;
            double group_max_abs_diff = 0.0;

            // 当前组的最大差异信息
            double cur_max_abs_diff = 0.0;
            int cur_max_output_idx = -1;
            std::vector<int64_t> cur_max_position;
            double cur_ref_value = 0.0;
            double cur_val_value = 0.0;

            for (size_t i = 0; i < local_outputs.size(); ++i) {
                auto cmp_result = CalculateSimilarity(ref_outputs[i], local_outputs[i]);

                // 取最小相似度作为该组的相似度
                group_similarity = std::min(group_similarity, cmp_result.similarity);

                // 更新组内最大差异信息
                if (cmp_result.max_abs_diff > cur_max_abs_diff) {
                    cur_max_abs_diff = cmp_result.max_abs_diff;
                    cur_max_output_idx = i;


                    cur_ref_value = cmp_result.ref_value;
                    cur_val_value = cmp_result.val_value;
                }
            }

            // 更新全局最大差异信息
            if (cur_max_abs_diff > result.max_abs_diff) {
                result.max_abs_diff = cur_max_abs_diff;
                result.max_diff_group_idx = group_idx;
                result.max_diff_output_idx = cur_max_output_idx;
                result.max_diff_position = cur_max_position;
                result.max_diff_ref_value = cur_ref_value;
                result.max_diff_val_value = cur_val_value;
            }

            // 累加相似度
            result.similarity += group_similarity;
            group_max_abs_diff = cur_max_abs_diff;  // 使用组内最大值
            valid_groups++;
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing group " << group_idx << " for op " << config.name
                      << ": " << e.what() << std::endl;
        }
    }

    // 计算平均相似度
    if (valid_groups > 0) {
        result.similarity /= valid_groups;
    } else {
        result.similarity = 0.0;
    }

    return result;
}

// ======================== 验证结果打印 ========================
void PrintValidationResults(const std::vector<ValidationResult>& results) {
    if (results.empty()) return;

    std::cout << "\n验证结果汇总:\n";
    std::cout << "=================================================================\n";
    std::cout << std::left
              << std::setw(30) << "算子名称"
              << std::setw(15) << "相似度"
              << std::setw(15) << "最大绝对误差" << "\n";
    std::cout << "-----------------------------------------------------------------\n";

    double total_similarity = 0.0;
    int count = 0;

    for (const auto& res : results) {
        std::cout << std::setw(30) << res.op_name
                  << std::setw(15) << std::fixed << std::setprecision(4) << res.similarity
                  << std::setw(15) << std::fixed << std::setprecision(6) << res.max_abs_diff << "\n";

        if (res.similarity > 0) {
            total_similarity += res.similarity;
            count++;
        }
    }

    std::cout << "-----------------------------------------------------------------\n";
    if (count > 0) {
        std::cout << std::setw(30) << "平均相似度"
                  << std::setw(15) << std::fixed << std::setprecision(4) << total_similarity / count
                  << "\n";
    }
    std::cout << "=================================================================\n";

    // 找出相似度最低的算子
    auto min_similarity = *std::min_element(results.begin(), results.end(),
        [](const ValidationResult& a, const ValidationResult& b) {
            return a.similarity < b.similarity;
        });

    std::cout << "\n性能分析:\n";
    std::cout << "  相似度最低的算子: " << min_similarity.op_name
              << " (" << min_similarity.similarity << ")\n";
    std::cout << "  平均相似度: " << (count > 0 ? total_similarity / count : 0.0) << "\n";

    // 新增：差异详情
    std::cout << "\n差异详情:\n";
    std::cout << "=================================================================\n";
    for (const auto& res : results) {
        if (res.max_diff_group_idx >= 0 && res.max_abs_diff > 1e-6) {
            std::cout << "算子: " << res.op_name << "\n";
            std::cout << "  最大绝对误差出现在测试组: " << res.max_diff_group_idx << "\n";
            std::cout << "  输出张量: " << res.max_diff_output_idx << "\n";

            std::cout << "  位置: [";
            for (size_t i = 0; i < res.max_diff_position.size(); ++i) {
                std::cout << res.max_diff_position[i];
                if (i < res.max_diff_position.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";

            std::cout << "  参考值: " << std::setprecision(16) << res.max_diff_ref_value << "\n";
            std::cout << "  实际值: " << std::setprecision(16) << res.max_diff_val_value << "\n";
            std::cout << "  绝对差值: " << std::abs(res.max_diff_ref_value - res.max_diff_val_value) << "\n";
            std::cout << "-----------------------------------------------------------------\n";
        }
    }
    std::cout << "=================================================================\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <mode> <set_dir> <device_type>\n"
                  << "Modes:\n"
                  << "  generate - Generate random parameters and reference outputs\n"
                  << "  validate - Validate operator execution against reference outputs\n"
                  << "Device types:\n"
                  << "  native - Use native operator library\n"
                  << "  device - Use device operator library\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string set_dir = argv[2];
    std::string device_type = argv[3];

    // 根据设备类型确定算子库目录
    std::string lib_dir;
    if (device_type == "native") {
        lib_dir = set_dir + "/native";
    } else if (device_type == "device") {
        lib_dir = set_dir + "/device";
    } else {
        std::cerr << "Invalid device type: " << device_type << ". Use 'native' or 'device'." << std::endl;
        return 1;
    }

    // 数据目录设置为set目录
    std::string data_dir = set_dir;

    // 获取所有算子配置文件名
    std::vector<std::string> config_files;
    for (const auto& entry : fs::directory_iterator(lib_dir)) {
        if (entry.path().extension() == ".cfg") {
            config_files.push_back(entry.path().string());
        }
    }

    int total = config_files.size();
    int completed = 0;

    if (total == 0) {
        std::cerr << "No operator config files found in " << lib_dir << std::endl;
        return 1;
    }

    std::cout << "Found " << total << " operators\n";
    std::cout << "Progress: [";
    std::cout.flush();

    std::vector<ValidationResult> validation_results;
    for (const auto& cfg_path : config_files) {
        try {
            // 解析配置文件
            auto config = ParseConfig(cfg_path);

            if (mode == "generate") {
                // 生成模式：生成随机参数和参考输出
                GenerateReference(config, lib_dir, data_dir);
            } else if (mode == "validate") {
                // 验证模式：执行算子并验证输出
                auto res = Validate(config, lib_dir, data_dir);
                validation_results.push_back(res);
            }

            completed++;

            // 打印进度
            float progress = static_cast<float>(completed) / total;
            int bars = static_cast<int>(progress * 50);
            std::cout << "\rProgress: [";
            for (int i = 0; i < 50; ++i) {
                std::cout << (i < bars ? '=' : ' ');
            }
            std::cout << "] " << std::setprecision(1) << std::fixed
                      << (progress * 100) << "% (" << completed << "/" << total << ")";
            std::cout.flush();
        }
        catch (const std::exception& e) {
            // 提取算子名
            fs::path p(cfg_path);
            std::string op_name = p.stem().string();

            std::cerr << "\nError processing op: " << op_name << " - " << e.what() << std::endl;

            // 为验证结果添加错误条目
            if (mode == "validate") {
                ValidationResult res;
                res.op_name = op_name;
                res.similarity = 0.0;
                res.max_abs_diff = 0.0;
                validation_results.push_back(res);
            }
        }
    }

    std::cout << "\nProcessing completed! " << completed << "/" << total << " operators processed successfully." << std::endl;

    // 打印验证结果（如果是在验证模式下运行）
    if (mode == "validate") {
        PrintValidationResults(validation_results);

        // 保存验证结果到文件
        std::ofstream outfile(data_dir + "/validation_results.csv");
        if (outfile.is_open()) {
            outfile << "Operator,Similarity,Max Abs Diff,Group Index,Output Index,Position,Ref Value,Val Value\n";
            for (const auto& res : validation_results) {
                outfile << res.op_name << ","
                        << res.similarity << ","
                        << res.max_abs_diff << ","
                        << res.max_diff_group_idx << ","
                        << res.max_diff_output_idx << ",\"";

                // 格式化位置
                if (!res.max_diff_position.empty()) {
                    for (size_t i = 0; i < res.max_diff_position.size(); ++i) {
                        outfile << res.max_diff_position[i];
                        if (i < res.max_diff_position.size() - 1) outfile << ",";
                    }
                }
                outfile << "\"," << std::setprecision(16) << res.max_diff_ref_value
                        << "," << std::setprecision(16) << res.max_diff_val_value << "\n";
            }
            std::cout << "Validation results saved to " << data_dir << "/validation_results.csv\n";
        }
    }

    return 0;
}