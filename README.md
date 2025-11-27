# RVdoo

RVdoo 是一个用于测试深度学习算子在 RISC-V 架构上性能表现的工具。目前支持 TVM 和 TFLite-Micro 作为后端，可分别进行标量（RV）和向量（RVV）指令集的测试。

## 使用流程

### 1. 环境配置

请按照以下步骤配置运行环境：

1. 参考 TVM 和 TFLite-Micro 的官方文档，分别完成两者的编译
2. 注意需要同时生成本地编译与跨平台编译的结果
3. 请使用旧版本 TVM（新版本已不再支持 Relay）
4. TVM 编译完成后，请安装对应的 Python 支持包
5. 确保所有 Python 依赖已正确安装，之后即可开始测试
6. 如果需要自动化测试，需要配置跟开发板的连接
7. 配置完毕后，按照makefile的要求，修改makefile中所需的lib地址

### 2. 使用方法

1. 测试rv和rvv对比：python new_test/test.py
2. 测试tflite：python new_test/tflite_test.py
输出的地址可以自定义
