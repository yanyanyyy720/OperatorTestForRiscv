#include "operator_executor.h"
#include "data_generator.h"
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
#include <numeric>

namespace fs = std::filesystem;
using namespace tvm::runtime;

// 辅助函数：去除字符串两端的空白字符
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

// 将DLDataType转换为字符串的辅助函数
std::string DLDataTypeToString(DLDataType dtype) {
    std::string result;

    switch (dtype.code) {
        case kDLInt: result = "int"; break;
        case kDLUInt: result = "uint"; break;
        case kDLFloat: result = "float"; break;
        case kDLBfloat: result = "bfloat"; break;
        default: result = "unknown"; break;
    }

    result += std::to_string(dtype.bits);
    return result;
}

// 命令行参数结构体
struct CommandLineArgs {
    std::string so_path;
    std::string output_dir;
    std::string input_path;  // 输入文件或文件夹路径
    int rounds = 1;          // 运行轮数
    bool save_outputs = true; // 是否保存输出
    bool save_random_inputs = false; // 是否保存随机生成的输入
    bool has_input = false;
    bool has_output = false;
};

// 解析命令行参数
CommandLineArgs parse_command_line(int argc, char** argv) {
    CommandLineArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--input" && i + 1 < argc) {
            args.input_path = argv[++i];
            args.has_input = true;
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_dir = argv[++i];
            args.has_output = true;
        } else if (arg == "--rounds" && i + 1 < argc) {
            args.rounds = std::stoi(argv[++i]);
        } else if (arg == "--no-save-outputs") {
            args.save_outputs = false;
        } else if (arg == "--save-random-inputs") {
            args.save_random_inputs = true;
        } else if (arg.find("--") != 0) {
            // 位置参数：第一个是so_path，第二个是output_dir（如果没有--output）
            if (args.so_path.empty()) {
                args.so_path = arg;
            } else if (args.output_dir.empty() && !args.has_output) {
                args.output_dir = arg;
                args.has_output = true;
            }
        }
    }

    return args;
}

// 显示用法信息
void print_usage(const std::string& program_name) {
    std::cout << "用法: " << program_name << " [选项] <so_path> [output_dir]\n"
              << "选项:\n"
              << "  --input <path>          输入数据路径（文件或文件夹）\n"
              << "  --output <path>         输出目录路径\n"
              << "  --rounds <n>            运行轮数（默认: 1）\n"
              << "  --no-save-outputs       不保存输出数据（默认保存）\n"
              << "  --save-random-inputs    保存随机生成的输入数据\n"
              << "\n示例:\n"
              << "  " << program_name << " --input data/input_dir --output results/ operator.so\n"
              << "  " << program_name << " --input data/input_dir/round_0 --output results/ --rounds 5 operator.so\n"
              << "  " << program_name << " --output results/ --rounds 10 --save-random-inputs operator.so\n"
              << "  " << program_name << " --output results/ --no-save-outputs operator.so\n"
              << "  " << program_name << " operator.so results/" << std::endl;
}

// 从单个文件加载单个输入张量
NDArray load_single_input_from_file(const std::string& file_path,
                                   const std::vector<int64_t>& shape,
                                   DLDataType dtype) {
    std::cout << "从文件加载输入: " << file_path << std::endl;
    std::cout << "  形状: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "], 数据类型: " << DLDataTypeToString(dtype) << std::endl;

    try {
        auto input_array = DataGenerator::LoadFromFile(file_path, shape, dtype, DLDevice{kDLCPU, 0});
        std::cout << "  加载成功" << std::endl;
        return input_array;
    } catch (const std::exception& e) {
        throw std::runtime_error("加载文件失败: " + std::string(e.what()));
    }
}

// 从指定轮次目录加载输入数据
std::vector<NDArray> load_inputs_from_round_dir(const std::string& round_dir,
                                               const std::vector<OperatorExecutor::TensorInfo>& input_info) {
    std::vector<NDArray> inputs;

    if (!fs::exists(round_dir) || !fs::is_directory(round_dir)) {
        throw std::runtime_error("轮次目录不存在: " + round_dir);
    }

    // 查找输入文件
    std::vector<std::string> input_files;
    for (const auto& entry : fs::directory_iterator(round_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find("input_") == 0 &&
                (entry.path().extension() == ".bin" || entry.path().extension() == ".dat")) {
                input_files.push_back(entry.path().string());
            }
        }
    }

    if (input_files.empty()) {
        throw std::runtime_error("在轮次目录中未找到输入文件: " + round_dir);
    }

    // 按文件名排序
    std::sort(input_files.begin(), input_files.end());

    if (input_files.size() != input_info.size()) {
        std::cout << "警告: 找到 " << input_files.size() << " 个输入文件，但需要 "
                  << input_info.size() << " 个输入" << std::endl;
    }

    // 加载文件，尽量匹配输入信息
    for (size_t i = 0; i < std::min(input_files.size(), input_info.size()); ++i) {
        const auto& info = input_info[i];
        DLDataType dtype = OperatorExecutor::StrToDLDataType(info.dtype);
        auto input_array = load_single_input_from_file(input_files[i], info.shape, dtype);
        inputs.push_back(input_array);
    }

    return inputs;
}

// 从输入目录加载所有轮次的输入数据
std::vector<std::vector<NDArray>> load_all_rounds_from_directory(const std::string& input_dir,
                                                                const std::vector<OperatorExecutor::TensorInfo>& input_info) {
    std::vector<std::vector<NDArray>> all_rounds_inputs;

    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        throw std::runtime_error("输入路径不是有效的目录: " + input_dir);
    }

    // 查找所有轮次目录
    std::vector<std::string> round_dirs;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_directory()) {
            std::string dir_name = entry.path().filename().string();
            if (dir_name.find("round_") == 0) {
                round_dirs.push_back(entry.path().string());
            }
        }
    }

    if (round_dirs.empty()) {
        // 如果没有轮次目录，尝试直接加载输入文件
        std::cout << "未找到轮次目录，尝试直接加载输入文件" << std::endl;
        try {
            auto inputs = load_inputs_from_round_dir(input_dir, input_info);
            all_rounds_inputs.push_back(inputs);
            return all_rounds_inputs;
        } catch (const std::exception& e) {
            throw std::runtime_error("无法加载输入数据: " + std::string(e.what()));
        }
    }

    // 按轮次序号排序
    std::sort(round_dirs.begin(), round_dirs.end());

    std::cout << "找到 " << round_dirs.size() << " 个轮次目录" << std::endl;

    // 加载每个轮次的输入
    for (size_t i = 0; i < round_dirs.size(); ++i) {
        std::string round_name = fs::path(round_dirs[i]).filename().string();
        std::cout << "[" << (i + 1) << "/" << round_dirs.size() << "] 加载轮次: " << round_name << std::endl;

        try {
            auto inputs = load_inputs_from_round_dir(round_dirs[i], input_info);
            all_rounds_inputs.push_back(inputs);
        } catch (const std::exception& e) {
            std::cerr << "警告: 跳过轮次 " << round_name << " - " << e.what() << std::endl;
        }
    }

    if (all_rounds_inputs.empty()) {
        throw std::runtime_error("未能成功加载任何输入数据");
    }

    return all_rounds_inputs;
}

// 从单个文件加载输入数据
std::vector<NDArray> load_inputs_from_file(const std::string& file_path,
                                          const std::vector<OperatorExecutor::TensorInfo>& input_info) {
    std::vector<NDArray> inputs;

    // 尝试从单个文件加载第一个输入
    if (input_info.empty()) {
        throw std::runtime_error("没有输入信息");
    }

    const auto& info = input_info[0];
    DLDataType dtype = OperatorExecutor::StrToDLDataType(info.dtype);
    auto input_array = load_single_input_from_file(file_path, info.shape, dtype);
    inputs.push_back(input_array);

    return inputs;
}

// 从文件或文件夹加载输入数据
std::vector<std::vector<NDArray>> load_input_from_path(const std::string& input_path,
                                                      const std::vector<OperatorExecutor::TensorInfo>& input_info) {
    if (fs::is_regular_file(input_path)) {
        // 单个文件
        auto inputs = load_inputs_from_file(input_path, input_info);
        return {inputs};
    } else if (fs::is_directory(input_path)) {
        // 文件夹：可能包含多个轮次的输入
        return load_all_rounds_from_directory(input_path, input_info);
    } else {
        throw std::runtime_error("输入路径既不是文件也不是目录: " + input_path);
    }
}

// 生成随机输入数据
std::vector<NDArray> generate_random_inputs(const std::vector<OperatorExecutor::TensorInfo>& input_info) {
    std::vector<NDArray> inputs;

    for (const auto& info : input_info) {
        DLDataType dtype = OperatorExecutor::StrToDLDataType(info.dtype);

        // 根据数据类型设置合适的随机数范围
        double min_val = 0.0;
        double max_val = 1.0;

        if (dtype.code == kDLInt || dtype.code == kDLUInt) {
            min_val = 0.0;
            max_val = 100.0;
        }

        auto arr = DataGenerator::CreateArray(
            info.shape,
            dtype,
            DataGenMode::RANDOM,
            DLDevice{kDLCPU, 0},
            min_val,
            max_val
        );

        inputs.push_back(arr);
    }

    return inputs;
}

// 保存输入数据
void save_inputs(const std::vector<NDArray>& inputs,
                 const std::string& base_output_dir,
                 int round_index) {

    // 创建输入保存目录
    std::string input_dir = base_output_dir + "/input_data/round_" + std::to_string(round_index);
    fs::create_directories(input_dir);

    // 保存每个输入张量
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::string filename = input_dir + "/input_" + std::to_string(i) + ".bin";

        try {
            DataGenerator::SaveToFile(inputs[i], filename);
            std::cout << "  输入 " << i << " 保存到: " << filename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "警告: 保存输入 " << i << " 失败: " << e.what() << std::endl;
        }
    }
}

// 保存输出数据
void save_outputs(const std::vector<NDArray>& outputs,
                  const std::string& base_output_dir,
                  int round_index) {

    // 创建输出保存目录
    std::string output_dir = base_output_dir + "/output_data/round_" + std::to_string(round_index);
    fs::create_directories(output_dir);

    // 保存每个输出张量
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::string filename = output_dir + "/output_" + std::to_string(i) + ".bin";

        try {
            DataGenerator::SaveToFile(outputs[i], filename);
            std::cout << "  输出 " << i << " 保存到: " << filename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "警告: 保存输出 " << i << " 失败: " << e.what() << std::endl;
        }
    }
}

// 单个算子测试函数
void run_operator_test(const CommandLineArgs& args) {
    // 从.so文件路径中提取算子名称
    fs::path so_file_path(args.so_path);
    std::string op_name = so_file_path.stem().string();
    std::cout << "测试算子: " << op_name << std::endl;

    if (args.has_input) {
        std::cout << "输入路径: " << args.input_path << std::endl;
    } else {
        std::cout << "使用随机输入数据" << std::endl;
        if (args.save_random_inputs) {
            std::cout << "将保存随机生成的输入数据" << std::endl;
        }
    }

    std::cout << "输出目录: " << args.output_dir << std::endl;
    std::cout << "运行轮数: " << args.rounds << std::endl;
    std::cout << "保存输出: " << (args.save_outputs ? "是" : "否") << std::endl;

    // 创建算子执行器
    OperatorExecutor executor(args.so_path);

    // 获取输入信息
    const auto& input_info = executor.get_input_info();
    std::cout << "输入信息 (" << input_info.size() << " 个输入):" << std::endl;
    for (size_t i = 0; i < input_info.size(); ++i) {
        const auto& info = input_info[i];
        std::cout << "  [" << i << "] " << info.name
                  << ": shape=[";
        for (size_t j = 0; j < info.shape.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << info.shape[j];
        }
        std::cout << "], dtype=" << info.dtype << std::endl;
    }

    // 获取输出信息
    const auto& output_info = executor.get_output_info();
    std::cout << "输出信息 (" << output_info.size() << " 个输出):" << std::endl;
    for (size_t i = 0; i < output_info.size(); ++i) {
        const auto& info = output_info[i];
        std::cout << "  [" << i << "] " << info.name
                  << ": shape=[";
        for (size_t j = 0; j < info.shape.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << info.shape[j];
        }
        std::cout << "], dtype=" << info.dtype << std::endl;
    }

    // 准备输入数据
    std::vector<std::vector<NDArray>> all_rounds_inputs;

    if (args.has_input) {
        // 从指定路径加载输入
        all_rounds_inputs = load_input_from_path(args.input_path, input_info);
        std::cout << "成功加载 " << all_rounds_inputs.size() << " 个轮次的输入数据" << std::endl;
    } else {
        // 生成随机输入
        for(int count = 0;count < args.rounds;count++){
            auto inputs = generate_random_inputs(input_info);
            all_rounds_inputs.push_back(inputs);
        }
        std::cout << "生成 " << all_rounds_inputs.size() << " 轮随机输入" << std::endl;
    }

    // 确保输出目录存在
    fs::create_directories(args.output_dir);

    // 性能统计
    std::vector<double> execution_times;

    std::cout << "\n开始执行测试..." << std::endl;

    // 运行指定轮数
    for (int round = 0; round < args.rounds; ++round) {
        std::cout << "轮次 " << (round + 1) << "/" << args.rounds << "...";
        std::cout.flush();

        // 选择当前轮次的输入数据
        int input_round_index = round % all_rounds_inputs.size();
        const auto& inputs = all_rounds_inputs[input_round_index];

        if (round >= all_rounds_inputs.size()) {
            std::cout << " (重复使用输入轮次 " << input_round_index << ")";
        }

        // 设置输入
        for (size_t i = 0; i < inputs.size() && i < input_info.size(); ++i) {
            executor.set_input(input_info[i].name, inputs[i]);
        }

        auto start = std::chrono::high_resolution_clock::now();

        // 执行算子
        executor.run();

        auto end = std::chrono::high_resolution_clock::now();

        double round_time = std::chrono::duration<double, std::milli>(end - start).count();
        execution_times.push_back(round_time);

        // 保存输入数据（如果需要）
        if ((!args.has_input && args.save_random_inputs) || (args.has_input && args.save_random_inputs)) {
            save_inputs(inputs, args.output_dir, round);
        }

        // 获取并保存输出（如果启用保存）
        if (args.save_outputs) {
            auto outputs = executor.get_outputs();
            if (!outputs.empty()) {
                save_outputs(outputs, args.output_dir, round);
            }
        }

        std::cout << " 完成 (" << round_time << " ms)" << std::endl;
    }

    // 性能统计摘要
    std::cout << "\n=== 性能统计 ===" << std::endl;
    if (!execution_times.empty()) {
        double total_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0);
        double avg_time = total_time / execution_times.size();
        double min_time = *std::min_element(execution_times.begin(), execution_times.end());
        double max_time = *std::max_element(execution_times.begin(), execution_times.end());

        std::cout << "总执行次数: " << execution_times.size() << std::endl;
        std::cout << "总时间: " << total_time << " ms" << std::endl;
        std::cout << "平均时间: " << avg_time << " ms" << std::endl;
        std::cout << "最短时间: " << min_time << " ms" << std::endl;
        std::cout << "最长时间: " << max_time << " ms" << std::endl;

        // 计算内存使用量
        double total_input_size = 0.0;
        for (const auto& info : input_info) {
            DLDataType dtype = OperatorExecutor::StrToDLDataType(info.dtype);
            total_input_size += DataGenerator::CalculateArraySizeMB(info.shape, dtype);
        }

        double total_output_size = 0.0;
        for (const auto& info : output_info) {
            DLDataType dtype = OperatorExecutor::StrToDLDataType(info.dtype);
            total_output_size += DataGenerator::CalculateArraySizeMB(info.shape, dtype);
        }

        std::cout << "输入总大小: " << total_input_size << " MB" << std::endl;
        std::cout << "输出总大小: " << total_output_size << " MB" << std::endl;

        // 保存总体统计信息
        std::ofstream stats_file(args.output_dir + "/" + op_name + "_summary.txt");
        if (stats_file.is_open()) {
            stats_file << "算子: " << op_name << "\n"
                      << "输入路径: " << (args.has_input ? args.input_path : "随机生成") << "\n"
                      << "保存随机输入: " << (args.save_random_inputs ? "是" : "否") << "\n"
                      << "保存输出: " << (args.save_outputs ? "是" : "否") << "\n"
                      << "输入轮次数: " << all_rounds_inputs.size() << "\n"
                      << "运行轮数: " << args.rounds << "\n"
                      << "总执行次数: " << execution_times.size() << "\n"
                      << "总时间: " << total_time << " ms\n"
                      << "平均时间: " << avg_time << " ms\n"
                      << "最短时间: " << min_time << " ms\n"
                      << "最长时间: " << max_time << " ms\n"
                      << "输入大小: " << total_input_size << " MB\n"
                      << "输出大小: " << total_output_size << " MB\n";
            stats_file.close();
        }
    }

    std::cout << "\n测试完成！结果保存在: " << args.output_dir << std::endl;
}

// 主函数
int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // 解析命令行参数
    CommandLineArgs args = parse_command_line(argc, argv);

    // 验证必需参数
    if (args.so_path.empty()) {
        std::cerr << "错误: 必须指定算子库文件路径 (.so)" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (args.output_dir.empty()) {
        std::cerr << "错误: 必须指定输出目录" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (args.rounds < 1) {
        std::cerr << "错误: 运行轮数必须大于0" << std::endl;
        return 1;
    }

    // 显示配置信息
    std::cout << "=== 测试配置 ===" << std::endl;
    std::cout << "算子库: " << args.so_path << std::endl;
    std::cout << "输出目录: " << args.output_dir << std::endl;
    std::cout << "输入数据: " << (args.has_input ? args.input_path : "随机生成") << std::endl;
    std::cout << "运行轮数: " << args.rounds << std::endl;
    std::cout << "保存输出: " << (args.save_outputs ? "是" : "否") << std::endl;
    std::cout << "保存随机输入: " << (args.save_random_inputs ? "是" : "否") << std::endl;
    std::cout << "================\n" << std::endl;

    try {
        run_operator_test(args);
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        return 2;
    }

    return 0;
}