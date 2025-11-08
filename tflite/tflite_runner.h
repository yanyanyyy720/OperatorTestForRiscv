#ifndef TFLITE_RUNNER_H
#define TFLITE_RUNNER_H

#include <string>
#include <cstddef>
#include <cstdint>  // 添加cstdint以支持uint8_t

// 前向声明TfLiteTensor
struct TfLiteTensor;

class TFLiteRunner {
public:
    /**
     * @brief 构造函数，加载并初始化TFLite模型
     * @param model_path 模型文件路径
     */
    explicit TFLiteRunner(const std::string& model_path);
    
    /**
     * @brief 析构函数，释放所有资源
     */
    ~TFLiteRunner();
    
    // 禁用拷贝构造函数和赋值操作符
    TFLiteRunner(const TFLiteRunner&) = delete;
    TFLiteRunner& operator=(const TFLiteRunner&) = delete;
    
    /**
     * @brief 设置模型输入数据
     * @param index 输入张量索引
     * @param data 输入数据指针
     * @return 成功返回true，失败返回false
     */
    bool setInput(int index, const void* data);
    
    /**
     * @brief 执行模型推理
     * @return 成功返回true，失败返回false
     */
    bool run();
    
    /**
     * @brief 获取模型输出数据
     * @param index 输出张量索引
     * @param data 输出数据缓冲区指针
     * @return 成功返回true，失败返回false
     */
    bool getOutput(int index, void* data);
    
    /**
     * @brief 获取模型输入张量数量
     * @return 输入张量数量，如果未初始化返回0
     */
    int getInputCount() const;
    
    /**
     * @brief 获取模型输出张量数量
     * @return 输出张量数量，如果未初始化返回0
     */
    int getOutputCount() const;
    
    /**
     * @brief 获取指定输入张量的大小（字节数）
     * @param index 输入张量索引
     * @return 张量大小，如果索引无效或未初始化返回0
     */
    std::size_t getInputSize(int index) const;
    
    /**
     * @brief 获取指定输出张量的大小（字节数）
     * @param index 输出张量索引
     * @return 张量大小，如果索引无效或未初始化返回0
     */
    std::size_t getOutputSize(int index) const;
    
    /**
     * @brief 检查模型是否已成功初始化
     * @return 已初始化返回true，否则返回false
     */
    bool isInitialized() const { return initialized_; }

private:
    // 模型数据
    uint8_t* model_data_;
    
    // Tensor Arena内存
    uint8_t* tensor_arena_;
    
    // TFLite解释器（使用void*避免暴露TFLite内部类型）
    void* interpreter_;
    
    // 初始化状态
    bool initialized_;
};

#endif // TFLITE_RUNNER_H
