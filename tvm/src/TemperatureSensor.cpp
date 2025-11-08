//#include "TemperatureSensor.h"
//#include <iostream>
//#include <fstream>
//#include <cmath>
//#include <iomanip>
//#include <ctime>
//#include <chrono>
//
//// ==================== RunningStats 成员函数实现 ====================
//
//TemperatureSensor::RunningStats::RunningStats()
//    : n(0), mean(0.0), S(0.0) {}
//
//void TemperatureSensor::RunningStats::update(double x) {
//    n += 1;
//    double delta = x - mean;
//    mean += delta / n;
//    double delta2 = x - mean;
//    S += delta * delta2;
//}
//
//double TemperatureSensor::RunningStats::current_mean() const {
//    return mean;
//}
//
//double TemperatureSensor::RunningStats::variance() const {
//    if (n < 2) return 0.0;
//    return S / (n - 1);
//}
//
//double TemperatureSensor::RunningStats::std_dev() const {
//    return std::sqrt(variance());
//}
//
//void TemperatureSensor::RunningStats::reset() {
//    n = 0; mean = 0.0; S = 0.0;
//}
//
//int TemperatureSensor::RunningStats::count() const {
//    return n;
//}
//
//// ==================== TemperatureSensor 成员函数实现 ====================
//
//TemperatureSensor::TemperatureSensor()
//    : ctx(nullptr), dev_handle(nullptr),
//      is_measuring(false) {}
//
//TemperatureSensor::~TemperatureSensor() {
//    stopMeasurement();
//    cleanup();
//}
//
//bool TemperatureSensor::initialize() {
////    int ret = libusb_init(&ctx);
////    if (ret < 0) {
////        std::cerr << "初始化 libusb 失败: " << libusb_error_name(ret) << std::endl;
////        return false;
////    }
////
////    libusb_set_option(ctx, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
////
////    dev_handle = libusb_open_device_with_vid_pid(ctx, VENDOR_ID, PRODUCT_ID);
////    if (!dev_handle) {
////        std::cerr << "找不到设备 " << VENDOR_ID << ":" << PRODUCT_ID << std::endl;
////        libusb_exit(ctx);
////        return false;
////    }
////
////    std::cout << "找到温度设备!" << std::endl;
////
////    if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
////        ret = libusb_detach_kernel_driver(dev_handle, 0);
////        if (ret < 0) {
////            std::cerr << "无法分离内核驱动: " << libusb_error_name(ret) << std::endl;
////            libusb_close(dev_handle);
////            libusb_exit(ctx);
////            dev_handle = nullptr;
////            return false;
////        }
////    }
////
////    ret = libusb_claim_interface(dev_handle, 0);
////    if (ret < 0) {
////        std::cerr << "无法声明接口: " << libusb_error_name(ret) << std::endl;
////        libusb_close(dev_handle);
////        libusb_exit(ctx);
////        dev_handle = nullptr;
////        return false;
////    }
////
////    ret = libusb_clear_halt(dev_handle, 0x81);
////    if (ret < 0) {
////        std::cerr << "清除 halt 状态失败: " << libusb_error_name(ret) << std::endl;
////    }
//
//    return true;
//}
//
//void TemperatureSensor::startMeasurement() {
//    if (is_measuring) return;
//
//    is_measuring = true;
//    measurement_thread = std::thread(&TemperatureSensor::measurementLoop, this);
//}
//
//void TemperatureSensor::stopMeasurement() {
//    is_measuring = false;
//    if (measurement_thread.joinable()) {
//        measurement_thread.join();
//    }
//}
//
//std::vector<TemperatureSensor::TemperatureData> TemperatureSensor::getTemperatureData() {
//    std::lock_guard<std::mutex> lock(data_mutex);
//    return temperature_data;
//}
//
//TemperatureSensor::RunningStats TemperatureSensor::getStatistics() {
//    std::lock_guard<std::mutex> lock(data_mutex);
//    return stats;
//}
//
//bool TemperatureSensor::saveToFile(const std::string& filename, bool includeStats) {
//    std::lock_guard<std::mutex> lock(data_mutex);
//
//    std::ofstream file(filename);
//    if (!file.is_open()) {
//        std::cerr << "无法打开文件: " << filename << std::endl;
//        return false;
//    }
//
//    file << "Timestamp,Temperature(°C)" << std::endl;
//
//    for (const auto& data : temperature_data) {
//        std::time_t t = std::chrono::system_clock::to_time_t(data.timestamp);
//        char buffer[20];
//        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
//        file << buffer << "," << std::fixed << std::setprecision(2) << data.temperature << std::endl;
//    }
//
//    if (includeStats) {
//        file << "\nStatistical Summary:\n";
//        file << "Total Samples," << stats.count() << "\n";
//        file << "Mean Temperature," << std::fixed << std::setprecision(2) << stats.current_mean() << "\n";
//        file << "Variance," << std::fixed << std::setprecision(4) << stats.variance() << "\n";
//        file << "Standard Deviation," << std::fixed << std::setprecision(2) << stats.std_dev() << "\n";
//    }
//
//    file.close();
//    std::cout << "数据已保存到: " << filename << std::endl;
//    return true;
//}
//
//void TemperatureSensor::cleanup() {
//    if (dev_handle) {
//        libusb_release_interface(dev_handle, 0);
//        libusb_close(dev_handle);
//        dev_handle = nullptr;
//    }
//    if (ctx) {
//        libusb_exit(ctx);
//        ctx = nullptr;
//    }
//}
//
//void TemperatureSensor::measurementLoop() {
//    int skip_count = 0;
//    unsigned char buf[40];
//
//    while (is_measuring) {
//        float temperature = readTemperature(buf, skip_count);
//        if (temperature != -1000.0f) {
//            auto now = std::chrono::system_clock::now();
//            {
//                std::lock_guard<std::mutex> lock(data_mutex);
//                temperature_data.emplace_back(now, temperature);
//                stats.update(temperature);
//            }
//            std::cout << "温度: " << temperature << "°C" << std::endl;
//        }
//
//        std::this_thread::sleep_for(std::chrono::seconds(1));
//    }
//}
//
//float TemperatureSensor::readTemperature(unsigned char* buf, int& skip_count) {
////    int transferred = 0;
////    int ret = libusb_bulk_transfer(dev_handle, 0x81, buf, 40, &transferred, TIMEOUT_MS);
////
////    if (ret < 0) {
////        std::cerr << "读取错误: " << libusb_error_name(ret) << std::endl;
////        return -1000.0f;
////    }
////
////    if (transferred < 40) {
////        std::cerr << "接收数据不足: " << transferred << "/40" << std::endl;
////        return -1000.0f;
////    }
////
////    uint16_t raw_value = (buf[2] << 8) | buf[3];
////
////    if (raw_value == 0x7FFF) {
////        return -1000.0f;
////    }
////
////    if (raw_value == 850) {
////        if (skip_count < 3) {
////            skip_count++;
////            return -1000.0f;
////        }
////    }
////
////    return static_cast<float>(raw_value) / 10.0f;
//}