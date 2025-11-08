//#ifndef TEMPERATURE_SENSOR_H
//#define TEMPERATURE_SENSOR_H
//
////#include <libusb-1.0/libusb.h>
//#include <chrono>
//#include <vector>
//#include <atomic>
//#include <mutex>
//#include <thread>
//
//#define VENDOR_ID  0x13a5
//#define PRODUCT_ID 0x4321
//#define TIMEOUT_MS 5000
//
//class TemperatureSensor {
//public:
//    struct TemperatureData {
//        std::chrono::system_clock::time_point timestamp;
//        float temperature;
//        TemperatureData(std::chrono::system_clock::time_point tp, float temp)
//    : timestamp(tp), temperature(temp) {}
//    };
//
//    class RunningStats {
//    public:
//        RunningStats();
//        void update(double x);
//        double current_mean() const;
//        double variance() const;
//        double std_dev() const;
//        void reset();
//        int count() const;
//
//    private:
//        int n;
//        double mean;
//        double S;
//    };
//
//    TemperatureSensor();
//    ~TemperatureSensor();
//
//    bool initialize();
//    void startMeasurement();
//    void stopMeasurement();
//    std::vector<TemperatureData> getTemperatureData();
//    RunningStats getStatistics();
//    bool saveToFile(const std::string& filename, bool includeStats = true);
//    void cleanup();
//
//    // 删除复制构造函数和赋值运算符
//    TemperatureSensor(const TemperatureSensor&) = delete;
//    TemperatureSensor& operator=(const TemperatureSensor&) = delete;
//
//private:
//    void measurementLoop();
//    float readTemperature(unsigned char* buf, int& skip_count);
//
//    libusb_context* ctx;
//    libusb_device_handle* dev_handle;
//    std::atomic<bool> is_measuring;
//    std::thread measurement_thread;
//    std::mutex data_mutex;
//    std::vector<TemperatureData> temperature_data;
//    RunningStats stats;
//};
//
//#endif // TEMPERATURE_SENSOR_H