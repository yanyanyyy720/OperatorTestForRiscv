#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <tvm/runtime/ndarray.h>
#include <vector>
#include <random>
#include <cstring>

#include <string>
#include <vector>

// 在 DataGenMode 枚举中添加 SEQUENTIAL
enum class DataGenMode {
    RANDOM,
    ZERO,
    CONSTANT,
    NORMAL,
    SEQUENTIAL  // 新增：顺序数组模式
};
class DataGenerator {
public:
    /**
     * 创建数组的统一接口
     *
     * @param shape 数组形状
     * @param dtype 数据类型
     * @param mode 数据生成模式
     * @param device 设备类型 (默认为CPU)
     * @param param1 模式参数1 (常量值、正态分布的均值或随机数的下限)
     * @param param2 模式参数2 (正态分布的标准差或随机数的上限)
     * @return 创建的NDArray
     */
    static tvm::runtime::NDArray CreateArray(
        const std::vector<int64_t>& shape,
        DLDataType dtype,
        DataGenMode mode,
        DLDevice device = DLDevice{kDLCPU, 0},
        double param1 = 0.0,
        double param2 = 1.0
    );

    /**
     * 计算数组大小 (MB)
     *
     * @param shape 数组形状
     * @param dtype 数据类型
     * @return 数组大小 (MB)
     */
    static double CalculateArraySizeMB(
        const std::vector<int64_t>& shape,
        DLDataType dtype
    );
    // 新增功能：文件操作
    static tvm::runtime::NDArray LoadFromFile(
        const std::string& filename,
        const std::vector<int64_t>& shape,
        DLDataType dtype,
        DLDevice device = {kDLCPU, 0}
    );

    static void SaveToFile(
        const tvm::runtime::NDArray& arr,
        const std::string& filename
    );

    static void SaveToTextFile(
        const tvm::runtime::NDArray& arr,
        const std::string& filename
    );

    // 新增功能：数组比较
    static bool CompareArrays(
        const tvm::runtime::NDArray& arr1,
        const tvm::runtime::NDArray& arr2,
        double tolerance = 1e-6
    );

    // 新增功能：数组信息打印
    static void PrintArrayInfo(
        const tvm::runtime::NDArray& arr,
        const std::string& name = ""
    );

    // 重载版本：直接从NDArray计算大小
    static double CalculateArraySizeMB(const tvm::runtime::NDArray& arr);


private:
    // 内部实现函数
    static tvm::runtime::NDArray CreateRandomArray(
        const std::vector<int64_t>& shape,
        DLDataType dtype,
        DLDevice device,
        double min,
        double max
    );

    static tvm::runtime::NDArray CreateZeroArray(
        const std::vector<int64_t>& shape,
        DLDataType dtype,
        DLDevice device
    );

    static tvm::runtime::NDArray CreateConstantArray(
        const std::vector<int64_t>& shape,
        DLDataType dtype,
        double value,
        DLDevice device
    );

    static tvm::runtime::NDArray CreateNormalArray(
        const std::vector<int64_t>& shape,
        DLDataType dtype,
        double mean,
        double stddev,
        DLDevice device
    );
    // 新增内部实现：创建顺序数组
    static tvm::runtime::NDArray CreateSequentialArray(
        const std::vector<int64_t>& shape,
        DLDataType dtype,
        double start_value,
        DLDevice device
    );
};

#endif // DATA_GENERATOR_H