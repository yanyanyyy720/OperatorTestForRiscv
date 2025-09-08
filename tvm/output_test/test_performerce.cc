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
#include <thread>
#include <atomic>
#include <mutex>
namespace fs = std::filesystem;

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<Clock>;

// ======================== 内存资源监控类 ========================
class ResourceMonitor {
private:
    pid_t pid_;

public:
    ResourceMonitor(pid_t pid) : pid_(pid) {}

    double get_current_memory_mb() {
        std::string status_path = "/proc/" + std::to_string(pid_) + "/status";
        std::ifstream status(status_path);
        if (!status.is_open()) return 0.0;

        std::string line;
        for(int i = 0;i < 22;i++){
            std::getline(status, line);
        }
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line);
                std::string key, value, unit;
                iss >> key >> value >> unit;
                return std::stod(value)/1024;
            }
        }
        return -1.0;
    }
};

// ======================== 内存曲线数据结构 ========================
struct MemorySample {
    double time_offset_ms;
    double memory_mb;
};

// ======================== 性能数据结构 ========================
struct HardwareMetrics {
    double time_ms = 0.0;        // 执行时间（毫秒）
    double peak_memory_mb = 0.0; // 峰值内存（MB）
    std::vector<MemorySample> memory_curve; // 内存曲线
};

// ======================== 修改后的配置读取结构 ========================
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

// ======================== 内存监控器类 ========================
class MemoryMonitor {
private:
    std::atomic<bool> running_{false};
    std::thread monitor_thread_;
    ResourceMonitor resource_monitor_;
    TimePoint start_time_;
    std::vector<MemorySample> samples_;
    std::mutex samples_mutex_;
    double peak_memory_mb_ = 0.0;

public:
    MemoryMonitor(pid_t pid) : resource_monitor_(pid) {}

    void start() {
        running_ = true;
        start_time_ = Clock::now();
        peak_memory_mb_ = -10;
        samples_.clear();

        monitor_thread_ = std::thread([this]() {
            while (running_) {
                auto now = Clock::now();
                Duration elapsed = now - start_time_;
                double mem_usage = resource_monitor_.get_current_memory_mb();

                {
                    std::lock_guard<std::mutex> lock(samples_mutex_);
                    samples_.push_back({elapsed.count(), mem_usage});

                    if (mem_usage > peak_memory_mb_) {
                        peak_memory_mb_ = mem_usage;
                    }
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    HardwareMetrics stop() {
        running_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }

        HardwareMetrics metrics;
        metrics.peak_memory_mb = peak_memory_mb_;

        // 获取内存曲线样本
        {
            std::lock_guard<std::mutex> lock(samples_mutex_);
            metrics.memory_curve = std::move(samples_);
        }

        return metrics;
    }
};

// ======================== 性能结果结构 ========================
struct PerformanceResult {
    std::string op_name;
    HardwareMetrics hw_metrics;
};

// ======================== 修改后的性能测试模式 ========================
PerformanceResult RunPerformanceTest(
    const OperatorConfig& config,
    const std::string& output_dir
) {
    PerformanceResult result;
    result.op_name = config.name;

    pid_t pid = getpid();
    MemoryMonitor memory_monitor(pid);

    const std::string lib_path = "op_lib/set_0/native/" + config.name + ".so";
    auto gmod = CreateExecutor(lib_path);
    auto set_input = gmod.GetFunction("set_input");
    auto run = gmod.GetFunction("run");
    auto get_output = gmod.GetFunction("get_output");

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

    // 预热运行（不记录时间）
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& [name, shape, dtype_str] = config.inputs[i];
        set_input(name, inputs[i]);
    }
    run();
    for (size_t i = 0; i < outputs.size(); ++i) {
        get_output(i, outputs[i]);
    }

    // 动态确定迭代次数
    int min_iterations = 10;
    int max_iterations = 100000;
    int iterations = min_iterations;

    // 测量单次执行时间
    auto start_time = Clock::now();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& [name, shape, dtype_str] = config.inputs[i];
        set_input(name, inputs[i]);
    }
    run();
    for (size_t i = 0; i < outputs.size(); ++i) {
        get_output(i, outputs[i]);
    }
    auto end_time = Clock::now();
    Duration single_exec_time = end_time - start_time;

    // 如果单次执行时间小于1ms，增加迭代次数
    if (single_exec_time.count() < 10.0) {
        iterations = std::min(max_iterations,
                             static_cast<int>(100.0 / single_exec_time.count()));
        std::cout << "算子执行时间过短(" << single_exec_time.count() << "ms), "
                  << "增加迭代次数至: " << iterations << "\n";
    }

    // 启动内存监控（在正式执行前启动）
    memory_monitor.start();

    // 开始计时
    start_time = Clock::now();
    for(int k = 0; k < iterations; k++) {
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
    }

    // 停止计时
    end_time = Clock::now();
    Duration elapsed = end_time - start_time;

    // 停止内存监控并获取数据
    auto hw_metrics = memory_monitor.stop();
    hw_metrics.time_ms = elapsed.count();  // 计算单次执行时间
    // 设置硬件指标
    result.hw_metrics = hw_metrics;

    // 打印硬件指标
    std::cout << "\n[性能测试 - " << config.name << "]"
              << "  执行时间: " << std::fixed << std::setprecision(4) << result.hw_metrics.time_ms << " ms"
              << "  峰值内存: " << std::fixed << std::setprecision(1) << result.hw_metrics.peak_memory_mb << " MB"
              << "  迭代次数: " << iterations << "\n";

    // 保存内存曲线
    fs::create_directories(output_dir);
    std::string curve_file = output_dir + "/" + config.name + "_memory_curve.csv";
    std::ofstream curve_stream(curve_file);
    curve_stream << "Time(ms),Memory(MB)\n";
    for (const auto& sample : result.hw_metrics.memory_curve) {
        curve_stream << std::fixed << std::setprecision(3)
                     << sample.time_offset_ms << ","
                     << std::setprecision(1) << sample.memory_mb << "\n";
    }
    curve_stream.close();
    std::cout << "  内存曲线已保存至: " << curve_file << "\n";

    return result;
}

// ======================== 性能结果打印 ========================
void PrintPerformanceResults(const std::vector<PerformanceResult>& results, const std::string& output_dir) {
    if (results.empty()) return;

    // 创建综合报告文件
    std::string summary_file = output_dir + "/performance_summary.csv";
    std::ofstream summary_stream(summary_file);
    summary_stream << "Operator,Time(ms),Peak Memory(MB)\n";

    // 打印表头
    std::cout << "\n性能测试报告:\n";
    std::cout << "=============================================================\n";
    std::cout << std::left
              << std::setw(30) << "算子名称"
              << std::setw(15) << "时间(ms)"
              << std::setw(15) << "峰值内存(MB)" << "\n";
    std::cout << "-------------------------------------------------------------\n";

    double total_time = 0.0;
    double max_peak_memory = 0.0;

    for (const auto& res : results) {
        // 打印结果
        std::cout << std::setw(30) << res.op_name
                  << std::setw(15) << std::fixed << std::setprecision(2) << res.hw_metrics.time_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << res.hw_metrics.peak_memory_mb << "\n";

        // 写入CSV
        summary_stream << res.op_name << ","
                       << res.hw_metrics.time_ms << ","
                       << res.hw_metrics.peak_memory_mb << "\n";

        // 更新总计
        total_time += res.hw_metrics.time_ms;
        if (res.hw_metrics.peak_memory_mb > max_peak_memory) {
            max_peak_memory = res.hw_metrics.peak_memory_mb;
        }
    }

    summary_stream.close();
    std::cout << "性能摘要已保存至: " << summary_file << "\n";

    std::cout << "-------------------------------------------------------------\n";
    std::cout << std::setw(30) << "累计"
              << std::setw(15) << std::fixed << std::setprecision(2) << total_time
              << std::setw(15) << std::fixed << std::setprecision(1) << max_peak_memory << "\n";
    std::cout << "=============================================================\n";

    // 找出性能最差的算子
    auto longest_time = *std::max_element(results.begin(), results.end(),
        [](const PerformanceResult& a, const PerformanceResult& b) {
            return a.hw_metrics.time_ms < b.hw_metrics.time_ms;
        });

    auto highest_memory = *std::max_element(results.begin(), results.end(),
        [](const PerformanceResult& a, const PerformanceResult& b) {
            return a.hw_metrics.peak_memory_mb < b.hw_metrics.peak_memory_mb;
        });

    std::cout << "\n性能分析:\n";
    std::cout << "  最耗时的算子: " << longest_time.op_name
              << " (" << longest_time.hw_metrics.time_ms << " ms)\n";
    std::cout << "  内存占用最高的算子: " << highest_memory.op_name
              << " (" << highest_memory.hw_metrics.peak_memory_mb << " MB)\n";
    std::cout << "  平均执行时间: " << total_time / results.size() << " ms\n";
    std::cout << "  平均内存峰值: " << (max_peak_memory / results.size()) << " MB\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <output_directory>\n";
        return 1;
    }

    std::string output_dir = argv[1];
    std::cout << "Running in PERFORMANCE TEST mode, saving results to: " << output_dir << std::endl;

    // 获取所有算子配置文件名
    std::vector<std::string> config_files;
    for (const auto& entry : fs::directory_iterator("op_lib/set_0/native")) {
        if (entry.path().extension() == ".cfg") {
            config_files.push_back(entry.path().string());
        }
    }

    int total = config_files.size();
    int completed = 0;

    if (total == 0) {
        std::cerr << "No operator config files found in op_lib directory" << std::endl;
        return 1;
    }

    std::cout << "Found " << total << " operators\n";
    std::cout << "Progress: [";
    std::cout.flush();

    std::vector<PerformanceResult> results;
    for (const auto& cfg_path : config_files) {
        try {
            // 解析配置文件
            auto config = ParseConfig(cfg_path);

            // 运行性能测试
            auto res = RunPerformanceTest(config, output_dir);
            results.push_back(res);

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
        }
    }

    std::cout << "\nProcessing completed! " << completed << "/" << total << " operators processed successfully." << std::endl;

    // 打印性能结果
    PrintPerformanceResults(results, output_dir);

    return 0;
}