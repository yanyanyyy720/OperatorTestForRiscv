#ifndef OPERATOR_EXECUTOR_H
#define OPERATOR_EXECUTOR_H

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <vector>
#include <string>
#include <tuple>

class OperatorExecutor {
public:
    OperatorExecutor(const std::string& lib_path);

    void set_inputs(const std::vector<std::tuple<std::string, std::vector<int64_t>, std::string>>& input_configs);
    void set_outputs(const std::vector<std::tuple<std::vector<int64_t>, std::string>>& output_configs);
    void run();

    // 获取输入输出大小信息
    double get_total_input_size_mb() const;
    double get_total_output_size_mb() const;
    double get_input_size_mb(const std::string& name) const;
    double get_output_size_mb(size_t index) const;
    double get_temporary_memory_mb() const;

private:
    tvm::runtime::Module gmod_;

    // 仅存储配置信息，不存储实际数组
    std::vector<std::tuple<std::string, std::vector<int64_t>, std::string>> input_configs_;
    std::vector<std::tuple<std::vector<int64_t>, std::string>> output_configs_;

    // 辅助函数
    static DLDataType StrToDLDataType(const std::string& dtype_str);
    static tvm::runtime::NDArray CreateRandomArray(const std::vector<int64_t>& shape, DLDataType dtype);
    static tvm::runtime::NDArray CreateZeroArray(const std::vector<int64_t>& shape, DLDataType dtype);
    static double calculate_array_size_mb(const std::vector<int64_t>& shape, DLDataType dtype);
};

#endif // OPERATOR_EXECUTOR_H