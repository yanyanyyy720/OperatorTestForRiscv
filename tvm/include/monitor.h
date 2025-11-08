#ifndef PERFORMANCE_RECORDER_H
#define PERFORMANCE_RECORDER_H

#include <vector>
#include <string>
#include <mutex>
#include <chrono>
#include <thread>

class PerformanceRecorder {
public:
    struct MemorySample {
        double time_offset_ms;
        double memory_mb;
    };

    struct MetricSample {
        double time_offset_ms;
        double memory_mb;
        long total_cpu_time_ms;
        long user_cpu_time_ms;
        long system_cpu_time_ms;
        int thread_count;
        long minor_faults;
        long major_faults;
        double data_segment_mb;
        double shared_lib_mb;
        std::string name;
    };

    struct ProcessMetrics {
        double memory_mb = 0.0;          // 物理内存（VmRSS）
        double data_segment_mb = 0.0;     // 数据段大小（VmData）
        double shared_lib_mb = 0.0;       // 共享库大小（VmLib）
        int thread_count = 0;             // 线程数
        long user_cpu_time_ms = 0;        // 用户态CPU时间
        long system_cpu_time_ms = 0;      // 内核态CPU时间
        long total_cpu_time_ms = 0;       // 总CPU时间
        long minor_faults = 0;            // 次缺页
        long major_faults = 0;            // 主缺页
    };

    explicit PerformanceRecorder(pid_t pid);
    ~PerformanceRecorder();

    void start();
    void stop();
    void metric(const std::string& name);

    void export_memory_curve(const std::string& file_path) const;
    void export_metrics(const std::string& file_path) const;
    double peak_memory() const;

private:
    void cache_line_offsets();
    double read_memory_usage();
    ProcessMetrics get_current_metrics();
    void memory_monitor_thread_func();

    pid_t pid_;
    bool running_ = false;
    std::thread memory_thread_;
    std::chrono::high_resolution_clock::time_point start_time_;

    mutable std::mutex data_mutex_;
    std::vector<MemorySample> memory_samples_;
    std::vector<MetricSample> metric_samples_;
    double peak_memory_mb_ = 0.0;

    // 用于缓存/proc/pid/status中关键信息的行号
    int vmrss_line_ = -1;
    int vmdata_line_ = -1;
    int vmlib_line_ = -1;
    int threads_line_ = -1;
};

#endif // PERFORMANCE_RECORDER_H