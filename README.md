# RVdoo

RVdoo 是一个用于测试深度学习算子在 RISC-V 架构上性能表现的工具。目前支持 TVM 和 TFLite-Micro 作为后端，可分别进行标量（RV）和向量（RVV）指令集的测试。

# 1. 环境配置

请按照以下步骤配置运行环境：

1. 参考 TVM 和 TFLite-Micro 的官方文档，分别完成两者的编译
2. 注意需要同时生成本地编译与跨平台编译的结果
3. 请使用旧版本 TVM（新版本已不再支持 Relay）
4. TVM 编译完成后，请安装对应的 Python 支持包
5. 确保所有 Python 依赖已正确安装，之后即可开始测试
6. 如果需要自动化测试，需要配置跟开发板的连接
7. 配置完毕后，按照makefile的要求，修改makefile中所需的lib地址

# 2. 使用python脚本

1. 测试rv和rvv对比：python new_test/test.py
2. 测试tflite：python new_test/tflite_test.py
# 3.使用cpp直接测试
## 1. 测试tvm

### 使用方法
./program_name [选项] <so_path> [output_dir]

### 参数说明
必需参数
<so_path>- 算子共享库文件路径

可选参数

[output_dir]- 输出目录路径（位置参数）

--input <path>- 输入数据路径（文件或文件夹）

--output <path>- 输出目录路径（选项形式）

--rounds <n>- 运行轮数（默认: 1）

--no-save-outputs- 不保存输出数据（默认保存）

--save-random-inputs- 保存随机生成的输入数据
### 使用示例
1. 基本用法：指定输入数据和输出目录 ./program_name --input data/input_dir --output results/ operator.so
2. 多轮测试：./program_name --input data/input_dir/round_0 --output results/ --rounds 5 operator.so
3. 生成随机输入并保存：./program_name --output results/ --rounds 10 --save-random-inputs operator.so
4. 不保存输出结果：./program_name --output results/ --no-save-outputs operator.so
5.使用位置参数指定输出目录：./program_name operator.so results/
注意事项
当同时使用 --output选项和位置参数指定输出目录时，选项形式优先级更高
默认情况下会保存输出数据，使用 --no-save-outputs可禁用此功能
使用 --save-random-inputs可保存工具自动生成的随机输入数据
