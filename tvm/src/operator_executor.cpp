#include "operator_executor.h"
#include <tvm/runtime/registry.h>
#include <stdexcept>
#include <cstring>
#include <cmath>

using namespace tvm::runtime;

DLDevice kCPUDevice{DLDeviceType::kDLCPU, 0};

// 实现辅助函数
DLDataType OperatorExecutor::StrToDLDataType(const std::string& dtype_str) {
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
    } else {
        throw std::runtime_error("Unsupported data type: " + dtype_str);
    }
}

std::string OperatorExecutor::DLDataTypeToStr(DLDataType dtype) {
    if (dtype.code == kDLFloat && dtype.bits == 32) {
        return "float32";
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
        return "float16";
    } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
        return "bfloat16";
    } else if (dtype.code == kDLInt && dtype.bits == 32) {
        return "int32";
    } else if (dtype.code == kDLInt && dtype.bits == 64) {
        return "int64";
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
        return "int8";
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
        return "uint8";
    } else {
        return "unknown";
    }
}

OperatorExecutor::OperatorExecutor(const std::string& lib_path) {
    Module mod_factory = Module::LoadFromFile(lib_path);
    gmod_ = mod_factory.GetFunction("default")(kCPUDevice);

    // 提取输入信息
    extract_input_info();

    // 提取输出信息
    extract_output_info();
}

void OperatorExecutor::extract_input_info() {
    // 1. 获取模块中的 "get_input_info" 函数
    PackedFunc get_input_info = gmod_.GetFunction("get_input_info");

    // 2. 调用函数获取输入信息
    Map<String, ObjectRef> input_info = get_input_info();

    // 3. 提取形状信息
    Map<String, ShapeTuple> shapeInfoMap = Downcast<Map<String, ShapeTuple>>(input_info["shape"]);

    // 4. 提取数据类型信息
    Map<String, String> dtypeInfoMap = Downcast<Map<String, String>>(input_info["dtype"]);

    // 5. 存储输入信息
    for (auto map_node : shapeInfoMap) {
        std::string name = map_node.first;
        ShapeTuple shape_tuple = map_node.second;
        std::vector<int64_t> shape(shape_tuple.begin(), shape_tuple.end());
        std::string dtype_str = dtypeInfoMap[name];

        input_info_.push_back({name, shape, dtype_str});
    }
}

void OperatorExecutor::extract_output_info() {
    // 1. 获取模块中的 "get_output_info" 函数
    PackedFunc get_output_info = gmod_.GetFunction("get_output_info");

    // 2. 调用函数获取输出信息
    Map<String, ObjectRef> output_info = get_output_info();

    // 3. 提取形状信息
    Map<String, ShapeTuple> shapeInfoMap = Downcast<Map<String, ShapeTuple>>(output_info["shape"]);

    // 4. 提取数据类型信息
    Map<String, String> dtypeInfoMap = Downcast<Map<String, String>>(output_info["dtype"]);

    // 5. 存储输出信息
    for (auto map_node : shapeInfoMap) {
        std::string name = map_node.first;
        ShapeTuple shape_tuple = map_node.second;
        std::vector<int64_t> shape(shape_tuple.begin(), shape_tuple.end());
        std::string dtype_str = dtypeInfoMap[name];

        output_info_.push_back({name, shape, dtype_str});
    }
}

const std::vector<OperatorExecutor::TensorInfo>& OperatorExecutor::get_input_info() const {
    return input_info_;
}

const std::vector<OperatorExecutor::TensorInfo>& OperatorExecutor::get_output_info() const {
    return output_info_;
}

void OperatorExecutor::set_input(const std::string& name, const tvm::runtime::NDArray& data) {
    auto set_input = gmod_.GetFunction("set_input");

    // 验证输入名称是否存在
    bool found = false;
    for (const auto& info : input_info_) {
        if (info.name == name) {
            found = true;
            break;
        }
    }

    if (!found) {
        throw std::runtime_error("Unknown input name: " + name);
    }

    set_input(name, data);
}

void OperatorExecutor::run() {
    auto run_func = gmod_.GetFunction("run");
    run_func();
}

std::vector<tvm::runtime::NDArray> OperatorExecutor::get_outputs() {
    auto get_output = gmod_.GetFunction("get_output");
    std::vector<tvm::runtime::NDArray> outputs;

    for (const auto& info : output_info_) {
        outputs.push_back(get_output(info.name));
    }

    return outputs;
}