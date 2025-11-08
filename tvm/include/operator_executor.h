#ifndef OPERATOR_EXECUTOR_H
#define OPERATOR_EXECUTOR_H

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <vector>
#include <string>

class OperatorExecutor {
public:
    // 统一的张量信息结构
    struct TensorInfo {
        std::string name;
        std::vector<int64_t> shape;
        std::string dtype;
    };

    explicit OperatorExecutor(const std::string& lib_path);

    // 获取输入/输出信息
    const std::vector<TensorInfo>& get_input_info() const;
    const std::vector<TensorInfo>& get_output_info() const;

    // 设置输入数据
    void set_input(const std::string& name, const tvm::runtime::NDArray& data);

    // 执行模型
    void run();

    // 获取输出张量
    std::vector<tvm::runtime::NDArray> get_outputs();
    static DLDataType StrToDLDataType(const std::string& dtype_str);
private:
    tvm::runtime::Module gmod_;
    std::vector<TensorInfo> input_info_;
    std::vector<TensorInfo> output_info_;

    void extract_input_info();
    void extract_output_info();

    std::string DLDataTypeToStr(DLDataType dtype);
};

#endif // OPERATOR_EXECUTOR_H