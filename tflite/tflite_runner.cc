#include "tflite_runner.h"
#include <fstream>
#include <cstring>
#include <iostream>
#include <cstdint>
#include <memory>

// TFLite 头文件
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// 创建并配置一个完整的算子解析器
tflite::MicroMutableOpResolver<120> CreateCompleteOpResolver() {
    tflite::MicroMutableOpResolver<120> resolver;
    
    // 数学运算
    resolver.AddAbs();
    resolver.AddAdd();
    
    resolver.AddAddN();
    resolver.AddCeil();
    resolver.AddDiv();
    resolver.AddExp();
    resolver.AddFloor();
    resolver.AddFloorDiv();
    resolver.AddFloorMod();
    resolver.AddLog();
    resolver.AddMaximum();
    resolver.AddMinimum();
    resolver.AddMul();
    resolver.AddNeg();
    resolver.AddRound();
    resolver.AddRsqrt();
    resolver.AddSqrt();
    resolver.AddSquare();
    resolver.AddSquaredDifference();
    resolver.AddSub();
    
    // 神经网络激活函数
    resolver.AddElu();
    resolver.AddHardSwish();
    resolver.AddLeakyRelu();
    resolver.AddLogistic();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddTanh();
    
    // 神经网络层
    resolver.AddAveragePool2D();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddL2Normalization();
    resolver.AddL2Pool2D();
    resolver.AddMaxPool2D();
    resolver.AddSvdf();
    
    // 张量操作
    resolver.AddBatchToSpaceNd();
    resolver.AddBroadcastArgs();
    resolver.AddBroadcastTo();
    resolver.AddDepthToSpace();
    resolver.AddExpandDims();
    resolver.AddFill();
    resolver.AddGather();
    resolver.AddGatherNd();
    resolver.AddMirrorPad();
    resolver.AddPack();
    resolver.AddPad();
    resolver.AddPadV2();
    resolver.AddReduceMax();
    resolver.AddReduceMin();
    resolver.AddReshape();
    resolver.AddReverseV2();
    resolver.AddShape();
    resolver.AddSlice();
    resolver.AddSpaceToBatchNd();
    resolver.AddSpaceToDepth();
    resolver.AddSplit();
    resolver.AddSplitV();
    resolver.AddSqueeze();
    resolver.AddStridedSlice();
    resolver.AddTranspose();
    resolver.AddUnpack();
    resolver.AddZerosLike();
    
    // 逻辑运算
    resolver.AddEqual();
    resolver.AddGreater();
    resolver.AddGreaterEqual();
    resolver.AddLess();
    resolver.AddLessEqual();
    resolver.AddNotEqual();
    resolver.AddLogicalAnd();
    resolver.AddLogicalNot();
    resolver.AddLogicalOr();
    resolver.AddSelectV2();
    
    // 比较和统计运算
    resolver.AddArgMax();
    resolver.AddArgMin();
    resolver.AddMean();
    resolver.AddSum();
    
    // 转换运算
    resolver.AddCast();
    resolver.AddDequantize();
    resolver.AddQuantize();
    
    // 控制流
    resolver.AddCallOnce();
    resolver.AddIf();
    resolver.AddWhile();
    
    // 变量操作
    resolver.AddAssignVariable();
    resolver.AddReadVariable();
    resolver.AddVarHandle();
    
    // 特殊运算
    resolver.AddCumSum();
    resolver.AddSoftmax();
    resolver.AddLogSoftmax();
    
    // 图像处理
    resolver.AddResizeBilinear();
    resolver.AddResizeNearestNeighbor();
    
    // 三角函数
    resolver.AddCos();
    resolver.AddSin();
    
    // 自定义操作（如果需要）
    // resolver.AddCircularBuffer();
    // resolver.AddEthosU();
    // resolver.AddDetectionPostprocess();
    
    
    return resolver;
}

// TFLiteRunner 类实现
TFLiteRunner::TFLiteRunner(const std::string& model_path) : 
    model_data_(nullptr), 
    tensor_arena_(nullptr), 
    interpreter_(nullptr), 
    initialized_(false) {
    
    // 使用ifstream替代C风格文件操作
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return;
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    model_data_ = new uint8_t[size];
    if (!file.read(reinterpret_cast<char*>(model_data_), size)) {
        std::cerr << "Failed to read model file" << std::endl;
        delete[] model_data_;
        model_data_ = nullptr;
        return;
    }
    
    // 验证模型
    const tflite::Model* model = ::tflite::GetModel(model_data_);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::cerr << "Model schema version mismatch" << std::endl;
        delete[] model_data_;
        model_data_ = nullptr;
        return;
    }
    std::cout << "version" <<TFLITE_SCHEMA_VERSION << std :: endl;
    // 注册常用算子
    tflite::MicroMutableOpResolver<120> resolver = CreateCompleteOpResolver();
    
    // 分配内存并创建解释器
    constexpr long kTensorArenaSize = 1024 * 1024 * 1024; // 200MB (修正了原代码中的错误，应该是200KB而不是200MB)
    tensor_arena_ = new uint8_t[2 * kTensorArenaSize];
    
    // 使用智能指针管理解释器，避免内存泄漏
    auto interpreter = std::make_unique<tflite::MicroInterpreter>(
        model, resolver, tensor_arena_, kTensorArenaSize);
    
    // 分配张量
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Tensor allocation failed" << std::endl;
        delete[] model_data_;
        model_data_ = nullptr;
        delete[] tensor_arena_;
        tensor_arena_ = nullptr;
        return;
    }
    // 转移所有权到成员变量
    interpreter_ = interpreter.release();
    initialized_ = true;
    std::cout << "Model loaded successfully: " << model_path << std::endl;
    
}

TFLiteRunner::~TFLiteRunner() {
    if (interpreter_) {
        //delete static_cast<tflite::MicroInterpreter*>(interpreter_);
        interpreter_ = nullptr;
    }
    if (tensor_arena_) {
        delete[] tensor_arena_;
        tensor_arena_ = nullptr;
    }
    if (model_data_) {
        delete[] model_data_;
        model_data_ = nullptr;
    }
}

bool TFLiteRunner::setInput(int index, const void* data) {
    if (!initialized_ || !interpreter_) {
        std::cerr << "Error: Interpreter not initialized" << std::endl;
        return false;
    }
    
    auto* interpreter = static_cast<tflite::MicroInterpreter*>(interpreter_);
    int input_count = interpreter->inputs_size();
    
    // 检查索引是否有效
    if (index < 0 || index >= input_count) {
        std::cerr << "Error: Input index out of range: " << index 
                  << " (valid range: 0 to " << input_count - 1 << ")" << std::endl;
        return false;
    }
    
    TfLiteTensor* input = interpreter->input(index);
    
    if (!input || !input->data.data) {
        std::cerr << "Error: Input tensor not found or data pointer is null for index: " << index << std::endl;
        return false;
    }
    
    // 复制数据
    memcpy(input->data.data, data, input->bytes);
    return true;
}

bool TFLiteRunner::run() {
    if (!initialized_ || !interpreter_) {
        std::cerr << "Error: Interpreter not initialized" << std::endl;
        return false;
    }
    
    auto* interpreter = static_cast<tflite::MicroInterpreter*>(interpreter_);
    TfLiteStatus status = interpreter->Invoke();
    
    if (status != kTfLiteOk) {
        std::cerr << "Error: Inference failed with status: " << status << std::endl;
        return false;
    }
    
    return true;
}

bool TFLiteRunner::getOutput(int index, void* data) {
    if (!initialized_ || !interpreter_) {
        std::cerr << "Error: Interpreter not initialized" << std::endl;
        return false;
    }
    
    auto* interpreter = static_cast<tflite::MicroInterpreter*>(interpreter_);
    int output_count = interpreter->outputs_size();
    
    // 检查索引是否有效
    if (index < 0 || index >= output_count) {
        std::cerr << "Error: Output index out of range: " << index 
                  << " (valid range: 0 to " << output_count - 1 << ")" << std::endl;
        return false;
    }
    
    TfLiteTensor* output = interpreter->output(index);
    if (!output || !output->data.data) {
        std::cerr << "Error: Output tensor not found or data pointer is null for index: " << index << std::endl;
        return false;
    }
    
    // 复制数据
    memcpy(data, output->data.data, output->bytes);
    return true;
}

int TFLiteRunner::getInputCount() const {
    if (!interpreter_ || !initialized_) return 0;
    auto* interpreter = static_cast<tflite::MicroInterpreter*>(interpreter_);
    return interpreter->inputs_size();
}

int TFLiteRunner::getOutputCount() const {
    if (!interpreter_ || !initialized_) return 0;
    auto* interpreter = static_cast<tflite::MicroInterpreter*>(interpreter_);
    return interpreter->outputs_size();
}

size_t TFLiteRunner::getInputSize(int index) const {
    if (!interpreter_ || !initialized_) return 0;
    auto* interpreter = static_cast<tflite::MicroInterpreter*>(interpreter_);
    if (index < 0 || index >= (int)interpreter->inputs_size()) return 0;
    
    TfLiteTensor* input = interpreter->input(index);
    return input ? input->bytes : 0;
}

size_t TFLiteRunner::getOutputSize(int index) const {
    if (!interpreter_ || !initialized_) return 0;
    auto* interpreter = static_cast<tflite::MicroInterpreter*>(interpreter_);
    if (index < 0 || index >= (int)interpreter->outputs_size()) return 0;
    
    TfLiteTensor* output = interpreter->output(index);
    return output ? output->bytes : 0;
}
