#include "operator_executor.h"
#include "monitor.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <thread>
#include <atomic>
#include <mutex>
#include <cctype>
#include <stdexcept>
#include <algorithm>
#include <tuple>
#include <sstream>
#include <unistd.h>
#include <random>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

namespace fs = std::filesystem;

// 配置结构
struct OperatorConfig {
    std::string name;
    std::string target;
    std::vector<std::tuple<std::string, std::vector<int64_t>, std::string>> inputs;
    std::vector<std::tuple<std::vector<int64_t>, std::string>> outputs;
};

// 辅助函数
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
    if (!file.is_open()) {
        throw std::runtime_error("无法打开配置文件: " + cfg_path);
    }

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
            if (key == "name") {
                config.name = value;
            } else if (key == "target") {
                config.target = value;
            }
        }
        else if (current_section == "inputs") {
            auto last_underscore = key.find_last_of('_');
            if (last_underscore == std::string::npos) continue;

            int idx = -1;
            try {
                idx = std::stoi(key.substr(last_underscore - 1, 1));
            } catch (...) {
                continue;
            }

            if (idx < 0 || config.inputs.size() <= static_cast<size_t>(idx)) {
                config.inputs.resize(idx + 1);
            }

            auto& input_tuple = config.inputs[idx];
            std::string suffix = key.substr(last_underscore + 1);

            if (suffix == "name") {
                std::get<0>(input_tuple) = value;
            } else if (suffix == "shape") {
                std::vector<int64_t> shape;
                std::istringstream shape_stream(value);
                std::string dim_str;

                while (std::getline(shape_stream, dim_str, ',')) {
                    try {
                        shape.push_back(std::stol(trim(dim_str)));
                    } catch (...) {
                    }
                }

                std::get<1>(input_tuple) = shape;
            } else if (suffix == "dtype") {
                std::get<2>(input_tuple) = value;
            }
        }
        else if (current_section == "outputs") {
            auto last_underscore = key.find_last_of('_');
            if (last_underscore == std::string::npos) continue;

            int idx = -1;
            try {
                idx = std::stoi(key.substr(last_underscore - 1, 1));
            } catch (...) {
                continue;
            }

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
                    try {
                        shape.push_back(std::stol(trim(dim_str)));
                    } catch (...) {
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

// 修改为单个算子测试函数
void run_operator_test(const std::string& cfg_path, const std::string& output_dir) {
    try {
        auto config = ParseConfig(cfg_path);
        std::cout << "测试算子: " << config.name << std::endl;

        // 获取算子库目录（配置文件所在目录）
        fs::path cfg_dir = fs::path(cfg_path).parent_path();
        std::string so_path = (cfg_dir / (config.name + ".so")).string();

        PerformanceRecorder recorder(getpid());
        recorder.metric("开始之前");

        OperatorExecutor executor(so_path);
        executor.set_inputs(config.inputs);
        executor.set_outputs(config.outputs);

        recorder.metric("执行开始");
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 3; i++) {
            executor.set_inputs(config.inputs);
            //run的时候会自动读取输出
            executor.run();
            recorder.metric(std::to_string(i));
        }
        auto end = std::chrono::high_resolution_clock::now();

        recorder.metric("执行完成");

        // 确保输出目录存在
        fs::create_directories(output_dir);
        recorder.export_metrics(output_dir + "/" + config.name + "_metric.cfg");

        double total_time = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_time = total_time / 10.0;

        std::cout << "  总时间: " << total_time << " ms" << std::endl;
        std::cout << "  平均时间: " << avg_time << " ms" << std::endl;
    }
    catch (const std::exception& e) {
        fs::path p(cfg_path);
        std::string op_name = p.stem().string();
        std::cerr << "\n处理算子时出错: " << op_name << " - " << e.what() << std::endl;
        throw; // 重新抛出异常以便在shell脚本中处理
    }
}

// 修改后的main函数
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <cfg_path> <output_dir>\n"
                  << "  cfg_path: 算子配置文件路径\n"
                  << "  output_dir: 输出目录" << std::endl;
        return 1;
    }

    const std::string cfg_path = argv[1];
    const std::string output_dir = argv[2];

    run_operator_test(cfg_path, output_dir);

    return 0;
}
