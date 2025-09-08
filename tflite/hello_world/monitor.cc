#include "monitor.h"
#include <cctype>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <mutex>
#include <thread>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <limits>

// 自定义 starts_with 函数（C++17 及以下）
bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

PerformanceRecorder::PerformanceRecorder(pid_t pid) : pid_(pid) {
    cache_line_offsets();
}

PerformanceRecorder::~PerformanceRecorder() {
    if (running_) {
        stop();
    }
}

void PerformanceRecorder::cache_line_offsets() {
    std::string status_path = "/proc/" + std::to_string(pid_) + "/status";
    std::ifstream status(status_path);
    if (!status.is_open()) return;

    std::string line;
    int line_num = 0;
    while (std::getline(status, line)) {
        line_num++;
        if (starts_with(line, "VmRSS:")) {
            vmrss_line_ = line_num;
        } else if (starts_with(line, "VmData:")) {
            vmdata_line_ = line_num;
        } else if (starts_with(line, "VmLib:")) {
            vmlib_line_ = line_num;
        } else if (starts_with(line, "Threads:")) {
            threads_line_ = line_num;
        }
    }
}

double PerformanceRecorder::read_memory_usage() {
    if (vmrss_line_ <= 0) cache_line_offsets();

    std::string status_path = "/proc/" + std::to_string(pid_) + "/status";
    std::ifstream status(status_path);
    if (!status.is_open()) return -1.0;

    for (int i = 1; i < vmrss_line_; i++) {
        status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::string line;
    if (std::getline(status, line)) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string value_str = line.substr(pos + 1);
            value_str.erase(value_str.begin(),
                std::find_if(value_str.begin(), value_str.end(),
                [](int ch) { return !std::isspace(ch); }));

            std::istringstream iss(value_str);
            long value;
            std::string unit;
            iss >> value >> unit;

            if (unit == "kB") {
                return value / 1024.0;
            }
        }
    }
    return -1.0;
}

PerformanceRecorder::ProcessMetrics PerformanceRecorder::get_current_metrics() {
    ProcessMetrics metrics;

    // 读取状态文件
    std::string status_path = "/proc/" + std::to_string(pid_) + "/status";
    std::ifstream status(status_path);
    if (status.is_open()) {
        std::string line;
        int line_num = 0;
        while (std::getline(status, line)) {
            line_num++;
            if (line_num == vmrss_line_) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    std::string value_str = line.substr(pos + 1);
                    value_str.erase(value_str.begin(),
                        std::find_if(value_str.begin(), value_str.end(),
                        [](int ch) { return !std::isspace(ch); }));

                    std::istringstream iss(value_str);
                    long value;
                    std::string unit;
                    iss >> value >> unit;

                    if (unit == "kB") {
                        metrics.memory_mb = value / 1024.0;
                    }
                }
            } else if (line_num == vmdata_line_) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    std::string value_str = line.substr(pos + 1);
                    value_str.erase(value_str.begin(),
                        std::find_if(value_str.begin(), value_str.end(),
                        [](int ch) { return !std::isspace(ch); }));

                    std::istringstream iss(value_str);
                    long value;
                    std::string unit;
                    iss >> value >> unit;

                    if (unit == "kB") {
                        metrics.data_segment_mb = value / 1024.0;
                    }
                }
            } else if (line_num == vmlib_line_) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    std::string value_str = line.substr(pos + 1);
                    value_str.erase(value_str.begin(),
                        std::find_if(value_str.begin(), value_str.end(),
                        [](int ch) { return !std::isspace(ch); }));

                    std::istringstream iss(value_str);
                    long value;
                    std::string unit;
                    iss >> value >> unit;

                    if (unit == "kB") {
                        metrics.shared_lib_mb = value / 1024.0;
                    }
                }
            } else if (line_num == threads_line_) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    std::string value_str = line.substr(pos + 1);
                    value_str.erase(value_str.begin(),
                        std::find_if(value_str.begin(), value_str.end(),
                        [](int ch) { return !std::isspace(ch); }));
                    metrics.thread_count = std::stoi(value_str);
                }
            }
        }
    }

    // 读取stat文件 - 改进版本
    std::string stat_path = "/proc/" + std::to_string(pid_) + "/stat";
    std::ifstream stat(stat_path);
    if (stat.is_open()) {
        std::string line;
        if (std::getline(stat, line)) {
            // 查找右括号位置，解决进程名包含空格的问题
            size_t rparen_pos = line.rfind(')');
            if (rparen_pos != std::string::npos) {
                // 创建字符串流，从右括号后开始解析
                std::istringstream iss(line.substr(rparen_pos + 1));
                std::string token;

                // 跳过状态字段（第3个字段）
                if (!(iss >> token)) return metrics;

                // 跳过接下来的10个字段（第4到第13字段）
                for (int i = 0; i < 10; i++) {
                    if (!(iss >> token)) return metrics;
                }

                // 读取utime（第14字段）和stime（第15字段）
                long utime = 0, stime = 0;
                if (iss >> utime >> stime) {
                    long ticks_per_sec = sysconf(_SC_CLK_TCK);
                    if (ticks_per_sec <= 0) {
                        ticks_per_sec = 100; // 默认值
                    }

                    // 分别计算用户时间和内核时间（毫秒）
                    metrics.user_cpu_time_ms = (utime * 1000) / ticks_per_sec;
                    metrics.system_cpu_time_ms = (stime * 1000) / ticks_per_sec;
                    metrics.total_cpu_time_ms = metrics.user_cpu_time_ms + metrics.system_cpu_time_ms;
                }

                // 读取缺页信息（第12和13字段）
                // 需要回退到第12字段
                iss.clear();
                iss.seekg(0, std::ios::beg);
                // 跳过前11个字段（状态+10个字段）
                for (int i = 0; i < 11; i++) {
                    if (!(iss >> token)) return metrics;
                }
                // 第12字段：次缺页（minor faults）
                if (iss >> token) {
                    metrics.minor_faults = std::stol(token);
                }
                // 第13字段：主缺页（major faults）
                if (iss >> token) {
                    metrics.major_faults = std::stol(token);
                }
            }
        }
    }

    return metrics;
}

void PerformanceRecorder::start() {
    if (running_) return;

    running_ = true;
    start_time_ = std::chrono::high_resolution_clock::now();

    // 清空之前的数据
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        memory_samples_.clear();
        metric_samples_.clear();
        peak_memory_mb_ = 0.0;
    }

    // 启动内存监控线程
    memory_thread_ = std::thread(&PerformanceRecorder::memory_monitor_thread_func, this);
}

void PerformanceRecorder::stop() {
    if (!running_) return;

    running_ = false;
    if (memory_thread_.joinable()) {
        memory_thread_.join();
    }
}

void PerformanceRecorder::memory_monitor_thread_func() {
    while (running_) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(now - start_time_);

        double memory_mb = read_memory_usage();

        if (memory_mb >= 0) {
            std::lock_guard<std::mutex> lock(data_mutex_);
            memory_samples_.push_back({elapsed.count(), memory_mb});

            if (memory_mb > peak_memory_mb_) {
                peak_memory_mb_ = memory_mb;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void PerformanceRecorder::metric(const std::string& name) {
    auto now = std::chrono::high_resolution_clock::now();
    if(metric_samples_.size() == 0){
        start_time_ = now;
    }
    auto elapsed = std::chrono::duration<double, std::milli>(now - start_time_);

    ProcessMetrics metrics = get_current_metrics();

    std::lock_guard<std::mutex> lock(data_mutex_);
    metric_samples_.push_back({
        elapsed.count(),
        metrics.memory_mb,
        metrics.total_cpu_time_ms,    // 总CPU时间
        metrics.user_cpu_time_ms,     // 用户CPU时间
        metrics.system_cpu_time_ms,   // 系统CPU时间
        metrics.thread_count,
        metrics.minor_faults,         // 次缺页
        metrics.major_faults,         // 主缺页
        metrics.data_segment_mb,      // 数据段大小
        metrics.shared_lib_mb,         // 共享库大小
        name
    });
}

void PerformanceRecorder::export_memory_curve(const std::string& file_path) const {
    std::ofstream file(file_path);
    if (!file) {
        return;
        //throw std::runtime_error("Cannot open file: " + file_path);
    }
    file << "Time(ms),Memory(MB)\n";

    std::lock_guard<std::mutex> lock(data_mutex_);
    for (const auto& sample : memory_samples_) {
        file << std::fixed << std::setprecision(3) << sample.time_offset_ms
             << "," << std::setprecision(1) << sample.memory_mb
             << "\n";
    }
}

void PerformanceRecorder::export_metrics(const std::string& file_path) const {
    std::ofstream file(file_path);
    if (!file) {
        return;
        //throw std::runtime_error("Cannot open file: " + file_path);
    }

    // 定义列宽常量
    constexpr int TIME_WIDTH = 10;
    constexpr int MEMORY_WIDTH = 10;
    constexpr int TOTAL_CPU_WIDTH = 12;
    constexpr int USER_CPU_WIDTH = 12;
    constexpr int SYSTEM_CPU_WIDTH = 12;
    constexpr int THREAD_COUNT_WIDTH = 8;
    constexpr int MINOR_FAULTS_WIDTH = 12;
    constexpr int MAJOR_FAULTS_WIDTH = 12;
    constexpr int DATA_SEG_WIDTH = 12;
    constexpr int SHARED_LIB_WIDTH = 12;
    constexpr int NAME_WIDTH = 20;

    // 输出标题行（右对齐）
    file << std::right
         << std::setw(TIME_WIDTH) << "Time(ms)" << ","
         << std::setw(MEMORY_WIDTH) << "RSS(MB)" << ","
         << std::setw(TOTAL_CPU_WIDTH) << "Total CPU(ms)" << ","
         << std::setw(USER_CPU_WIDTH) << "User CPU(ms)" << ","
         << std::setw(SYSTEM_CPU_WIDTH) << "System CPU(ms)" << ","
         << std::setw(THREAD_COUNT_WIDTH) << "Threads" << ","
         << std::setw(MINOR_FAULTS_WIDTH) << "Minor Faults" << ","
         << std::setw(MAJOR_FAULTS_WIDTH) << "Major Faults" << ","
         << std::setw(DATA_SEG_WIDTH) << "Data Seg(MB)" << ","
         << std::setw(SHARED_LIB_WIDTH) << "Shared Lib(MB)" << ","
         << std::setw(NAME_WIDTH) << "Name"
         << "\n";

    //std::lock_guard<std::mutex> lock(data_mutex_);
    for (const auto& sample : metric_samples_) {
        file << std::fixed
             << std::setw(TIME_WIDTH) << std::setprecision(3) << sample.time_offset_ms << ","
             << std::setw(MEMORY_WIDTH) << std::setprecision(1) << sample.memory_mb << ","
             << std::setw(TOTAL_CPU_WIDTH) << sample.total_cpu_time_ms << ","
             << std::setw(USER_CPU_WIDTH) << sample.user_cpu_time_ms << ","
             << std::setw(SYSTEM_CPU_WIDTH) << sample.system_cpu_time_ms << ","
             << std::setw(THREAD_COUNT_WIDTH) << sample.thread_count << ","
             << std::setw(MINOR_FAULTS_WIDTH) << sample.minor_faults << ","
             << std::setw(MAJOR_FAULTS_WIDTH) << sample.major_faults << ","
             << std::setw(DATA_SEG_WIDTH) << std::setprecision(1) << sample.data_segment_mb << ","
             << std::setw(SHARED_LIB_WIDTH) << std::setprecision(1) << sample.shared_lib_mb << ","
             << std::setw(NAME_WIDTH) << sample.name
             << "\n";
    }
}

double PerformanceRecorder::peak_memory() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return peak_memory_mb_;
}
