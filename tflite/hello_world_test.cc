#include "tflite_runner.h"
#include "monitor.h"
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <unistd.h>
namespace fs = std::filesystem;

// 运行单个模型
bool runModel(const std::string& model_path, const std::string& output_dir) {
    std::cout << "\n========================================\n";
    std::cout << "Running model: " << model_path << "\n";
    std::cout << "========================================\n";
    PerformanceRecorder recorder(getpid());
    recorder.metric("开始之前");
    // 创建输出目录（如果不存在）
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }
    // 1. 创建 TFLiteRunner 实例并加载模型
    TFLiteRunner runner(model_path);
    if (!runner.isInitialized()) {
        std::cerr << "❌ Failed to initialize TFLiteRunner for model: " << model_path << std::endl;
        return false;
    }
    
    // 2. 获取模型输入输出信息
    int input_count = runner.getInputCount();
    int output_count = runner.getOutputCount();
    
    std::cout << "Model has " << input_count << " input(s) and " 
              << output_count << " output(s)" << std::endl;
    // 3. 为每个输入准备数据（使用随机数）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for(int k = 0;k < 3;k++){
      for (int i = 0; i < input_count; ++i) {
          size_t input_size = runner.getInputSize(i);
          
          std::cout << "Input " << i << " size: " << input_size << " bytes" << std::endl;
          
          // 分配内存并填充随机数据
          std::vector<float> input_data(input_size / sizeof(float));
          
          // 用随机数填充输入
          for (size_t j = 0; j < input_data.size(); ++j) {
              input_data[j] = static_cast<float>(dis(gen));
          }
          
          // 设置输入
          if (!runner.setInput(i, input_data.data())) {
              std::cerr << "❌ Failed to set input " << i << std::endl;
              return false;
          }
      }
      
      // 4. 运行推理并计时
      auto start = std::chrono::high_resolution_clock::now();
      if (!runner.run()) {
          std::cerr << "❌ Failed to run inference" << std::endl;
          return false;
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      
      std::cout << "✅ Inference completed successfully in " 
                << duration.count() << " microseconds" << std::endl;
      
      // 5. 获取并显示输出
      for (int i = 0; i < output_count; ++i) {
          size_t output_size = runner.getOutputSize(i);
          
          std::cout << "Output " << i << " size: " << output_size << " bytes" << std::endl;
          
          // 分配内存并获取输出
          std::vector<float> output_data(output_size / sizeof(float));
          if (!runner.getOutput(i, output_data.data())) {
              std::cerr << "❌ Failed to get output " << i << std::endl;
              return false;
          }
      }
      recorder.metric("running");
    }
    // 生成输出文件名（保留原文件名，添加.cfg后缀）
    fs::path model_path_obj(model_path);
    std::string output_filename = model_path_obj.filename().string() + ".cfg";
    fs::path output_path = fs::path(output_dir) / output_filename;
    recorder.export_metrics(output_path.string());
    std::cout << "running here 1"<<std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [output_dir]" << std::endl;
        std::cerr << "  model_path: Path to single .tflite model file" << std::endl;
        std::cerr << "  output_dir: Output directory for results (default: 'outputs')" << std::endl;
        return -1;
    }
    
    std::string model_path = argv[1];
    std::string output_dir = (argc >= 3) ? argv[2] : "outputs";
    
    // 检查模型文件是否存在
    if (!fs::exists(model_path) || !fs::is_regular_file(model_path)) {
        std::cerr << "❌ Error: Model file '" << model_path << "' does not exist or is not a file" << std::endl;
        return -1;
    }
    
    // 检查文件扩展名
    if (fs::path(model_path).extension() != ".tflite") {
        std::cerr << "❌ Error: File '" << model_path << "' is not a .tflite model" << std::endl;
        return -1;
    }
    
    std::cout << "========================================\n";
    std::cout << "Running single model: " << model_path << "\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "========================================\n";
    
    bool success = runModel(model_path, output_dir);
    
    std::cout << "running here 2"<<std::endl;
    std::cout << "\n========================================\n";
    std::cout << "Summary:\n";
    std::cout << "  Model: " << model_path << "\n";
    std::cout << "  Status: " << (success ? "✅ SUCCESS" : "❌ FAILED") << "\n";
    std::cout << "  Output saved to: " << output_dir << "\n";
    std::cout << "========================================\n";
    
    return success ? 0 : 1;
}
