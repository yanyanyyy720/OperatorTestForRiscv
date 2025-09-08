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
    int max_diff_group_idx = -1;
    int max_diff_output_idx = -1;
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

    while (start < end && std::isspace(static_cast<unsigned char>(*start))) {
        ++start;
    }

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
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;
        if (line.front() == '#') continue;

        if (line.front() == '[' && line.back() == ']') {
            current_section = line.substr(1, line.size() - 2);
            continue;
        }

        auto delim_pos = line.find('=');
        if (delim_pos == std::string::npos) continue;

        std::string key = trim(line.substr(0, delim_pos));
        std::string value = trim(line.substr(delim_pos + 1));

        if (current_section == "operator") {
            if (key == "name") config.name = value;
            else if (key == "target") config.target = value;
        }
        else if (current_section == "inputs") {
            auto last_underscore = key.find_last_of('_');
            if (last_underscore == std::string::npos) continue;

            int idx = -1;
            try { idx = std::stoi(key.substr(last_underscore - 1, 1)); }
            catch (...) { continue; }

            if (idx < 0 || config.inputs.size() <= static_cast<size_t>(idx)) {
                config.inputs.resize(idx + 1);
            }

            auto& input_tuple = config.inputs[idx];
            std::string suffix = key.substr(last_underscore + 1);

            if (suffix == "name") std::get<0>(input_tuple) = value;
            else if (suffix == "shape") {
                std::vector<int64_t> shape;
                std::istringstream shape_stream(value);
                std::string dim_str;
                while (std::getline(shape_stream, dim_str, ',')) {
                    try { shape.push_back(std::stol(trim(dim_str))); }
                    catch (...) { }
                }
                std::get<1>(input_tuple) = shape;
            }
            else if (suffix == "dtype") std::get<2>(input_tuple) = value;
        }
        else if (current_section == "outputs") {
            auto last_underscore = key.find_last_of('_');
            if (last_underscore == std::string::npos) continue;

            int idx = -1;
            try { idx = std::stoi(key.substr(last_underscore - 1, 1)); }
            catch (...) { continue; }

            if (idx < 0 || config.outputs.size() <= static_cast<size_t>(idx)) {
                config.outputs.resize(idx + 1);
            }

            auto& output_tuple = config.outputs[idx];
            std::string suffix = key.substr(last_underscore + 1);

            if (suffix == "shape") {
                std::vector<int64_t> shape;
                std::istringstream shape_stream(value);
                std::string dim_str;
                while (std::getline(shape_stream, dim_str, ',')) {
                    try { shape.push_back(std::stol(trim(dim_str))); }
                    catch (...) { }
                }
                std::get<0>(output_tuple) = shape;
            }
            else if (suffix == "dtype") std::get<1>(output_tuple) = value;
        }
    }
    return config;
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
DLDataType StrToDLDataType(const std::string& dtype_str) {
    if (dtype_str == "float32") return DLDataType{kDLFloat, 32, 1};
    else if (dtype_str == "float16") return DLDataType{kDLFloat, 16, 1};
    else if (dtype_str == "bfloat16") return DLDataType{kDLBfloat, 16, 1};
    else if (dtype_str == "int32") return DLDataType{kDLInt, 32, 1};
    else if (dtype_str == "int64") return DLDataType{kDLInt, 64, 1};
    else if (dtype_str == "int8") return DLDataType{kDLInt, 8, 1};
    else if (dtype_str == "uint8") return DLDataType{kDLUInt, 8, 1};
    else throw std::runtime_error("Unsupported data type: " + dtype_str);
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
                for (size_t i = 0; i < num_elements; ++i) data[i] = dis(gen);
            } else if (dtype.bits == 16) {
                uint16_t* data = static_cast<uint16_t*>(arr->data);
                std::uniform_int_distribution<uint16_t> dis(0, 65535);
                for (size_t i = 0; i < num_elements; ++i) data[i] = dis(gen);
            }
            break;

        case kDLInt:
            if (dtype.bits == 64) {
                int64_t* data = static_cast<int64_t*>(arr->data);
                std::uniform_int_distribution<int64_t> dis(0, 100);
                for (size_t i = 0; i < num_elements; ++i) data[i] = dis(gen);
            } else if (dtype.bits == 32) {
                int32_t* data = static_cast<int32_t*>(arr->data);
                std::uniform_int_distribution<int32_t> dis(0, 100);
                for (size_t i = 0; i < num_elements; ++i) data[i] = dis(gen);
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

// ======================== 多组数据存储函数 ========================
void SaveMultiGroupData(
    const std::vector<std::vector<tvm::runtime::NDArray>>& groups,
    const std::string& filename
) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);

    // 写入组数
    int32_t num_groups = groups.size();
    file.write(reinterpret_cast<const char*>(&num_groups), sizeof(num_groups));

    for (const auto& group : groups) {
        // 写入当前组的张量数量
        int32_t num_arrays = group.size();
        file.write(reinterpret_cast<const char*>(&num_arrays), sizeof(num_arrays));

        for (const auto& arr : group) {
            // 写入数据类型和维度
            file.write(reinterpret_cast<const char*>(&arr->dtype), sizeof(DLDataType));
            int32_t ndim = arr->ndim;
            file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
            file.write(reinterpret_cast<const char*>(arr->shape), ndim * sizeof(int64_t));

            // 写入数据
            size_t num_elements = 1;
            for (int i = 0; i < ndim; ++i) num_elements *= arr->shape[i];
            size_t elem_size = (arr->dtype.bits + 7) / 8;
            file.write(reinterpret_cast<const char*>(arr->data), num_elements * elem_size);
        }
    }
}

// ======================== 多组数据加载函数 ========================
std::vector<std::vector<tvm::runtime::NDArray>> LoadMultiGroupData(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);

    // 读取组数
    int32_t num_groups;
    file.read(reinterpret_cast<char*>(&num_groups), sizeof(num_groups));

    std::vector<std::vector<tvm::runtime::NDArray>> groups;
    groups.resize(num_groups);

    for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
        // 读取当前组的张量数量
        int32_t num_arrays;
        file.read(reinterpret_cast<char*>(&num_arrays), sizeof(num_arrays));

        std::vector<tvm::runtime::NDArray> group;
        for (int arr_idx = 0; arr_idx < num_arrays; ++arr_idx) {
            // 读取数据类型
            DLDataType dtype;
            file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

            // 读取维度
            int32_t ndim;
            file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
            std::vector<int64_t> shape(ndim);
            file.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int64_t));

            // 创建数组
            auto arr = tvm::runtime::NDArray::Empty(shape, dtype, kCPUDevice);

            // 读取数据
            size_t num_elements = 1;
            for (auto dim : shape) num_elements *= dim;
            size_t elem_size = (dtype.bits + 7) / 8;
            file.read(reinterpret_cast<char*>(arr->data), num_elements * elem_size);

            group.push_back(arr);
        }
        groups[group_idx] = group;
    }
    return groups;
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

    fs::create_directories(data_dir);

    const int num_groups = 10;
    std::vector<std::vector<tvm::runtime::NDArray>> all_inputs;
    std::vector<std::vector<tvm::runtime::NDArray>> all_outputs;

    for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
        // 输入数据
        std::vector<tvm::runtime::NDArray> inputs;
        for (const auto& [name, shape, dtype_str] : config.inputs) {
            DLDataType dtype = StrToDLDataType(dtype_str);
            inputs.push_back(CreateRandomArray(shape, dtype));
        }

        // 输出数据
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

        all_inputs.push_back(inputs);
        all_outputs.push_back(outputs);
    }

    // 保存合并文件
    SaveMultiGroupData(all_inputs, data_dir + "/" + config.name + "_inputs.bin");
    SaveMultiGroupData(all_outputs, data_dir + "/" + config.name + "_ref_outputs.bin");
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

    // 加载合并的输入数据
    auto all_inputs = LoadMultiGroupData(data_dir + "/" + config.name + "_inputs.bin");
    const int num_groups = all_inputs.size();

    std::vector<std::vector<tvm::runtime::NDArray>> all_val_outputs;

    for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
        try {
            const auto& inputs = all_inputs[group_idx];

            // 设置输入
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto& [name, shape, dtype_str] = config.inputs[i];
                set_input(name, inputs[i]);
            }

            // 创建输出数组
            std::vector<tvm::runtime::NDArray> outputs;
            for (const auto& [shape, dtype_str] : config.outputs) {
                DLDataType dtype = StrToDLDataType(dtype_str);
                outputs.push_back(CreateZeroArray(shape, dtype));
            }

            // 执行算子
            run();

            // 获取输出
            for (size_t i = 0; i < outputs.size(); ++i) {
                get_output(i, outputs[i]);
            }

            all_val_outputs.push_back(outputs);
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing group " << group_idx << " for op " << config.name
                      << ": " << e.what() << std::endl;
        }
    }

    // 保存合并的实际输出
    SaveMultiGroupData(all_val_outputs, data_dir + "/" + config.name + "_val_outputs.bin");
    return result;
}

// ======================== 主函数（保持不变） ========================
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <mode> <set_dir> <device_type>\n"
                  << "Modes:\n"
                  << "  generate - Generate random parameters and reference outputs\n"
                  << "  validate - Validate operator execution and save outputs\n"
                  << "Device types:\n"
                  << "  native - Use native operator library\n"
                  << "  device - Use device operator library\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string set_dir = argv[2];
    std::string device_type = argv[3];

    std::string lib_dir;
    if (device_type == "native") lib_dir = set_dir + "/native";
    else if (device_type == "device") lib_dir = set_dir + "/device";
    else {
        std::cerr << "Invalid device type: " << device_type << ". Use 'native' or 'device'." << std::endl;
        return 1;
    }

    std::string data_dir = set_dir + "/data";
    std::vector<std::string> config_files;
    for (const auto& entry : fs::directory_iterator(lib_dir)) {
        if (entry.path().extension() == ".cfg") {
            config_files.push_back(entry.path().string());
        }
    }

    int total = config_files.size();
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
            auto config = ParseConfig(cfg_path);
            if (mode == "generate") GenerateReference(config, lib_dir, data_dir);
            else if (mode == "validate") validation_results.push_back(Validate(config, lib_dir, data_dir));

            // 进度显示逻辑（略）
        }
        catch (const std::exception& e) {
            fs::path p(cfg_path);
            std::cerr << "\nError processing op: " << p.stem().string() << " - " << e.what() << std::endl;
            if (mode == "validate") {
                ValidationResult res;
                res.op_name = p.stem().string();
                validation_results.push_back(res);
            }
        }
    }

    std::cout << "\nProcessing completed! " << validation_results.size() << "/" << total << " operators processed successfully." << std::endl;
    return 0;
}