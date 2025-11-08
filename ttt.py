import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os


class SoftmaxLayer(layers.Layer):
    """Softmax层"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.softmax(inputs)


class SigmoidLayer(layers.Layer):
    """Sigmoid层"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.sigmoid(inputs)


def test_layers_simple():
    """简化测试：只保存32位输入输出，每个算子100个样本"""

    # 设置随机种子
    tf.random.set_seed(42)
    np.random.seed(42)

    # 创建测试目录
    test_dir = "simple_layer_tests"
    os.makedirs(test_dir, exist_ok=True)

    print("开始简化测试Sigmoid和Softmax层...")
    print("=" * 50)

    # 测试配置
    test_cases = [
        {"name": "Sigmoid", "layer": SigmoidLayer(), "input_shape": (4, 1)},
        {"name": "Softmax", "layer": SoftmaxLayer(), "input_shape": (4, 10)}
    ]

    # 每个算子保存100个输入输出
    num_samples = 100

    for test_case in test_cases:
        print(f"\n测试算子: {test_case['name']}")
        print(f"输入形状: {test_case['input_shape']}")
        print(f"样本数量: {num_samples}")

        # 准备存储所有输入输出
        all_inputs = np.zeros((num_samples,) + test_case['input_shape'], dtype=np.float32)
        all_outputs = np.zeros((num_samples,) + test_case['input_shape'], dtype=np.float32)

        # 生成100个样本
        for i in range(num_samples):
            # 生成随机输入（-1到1之间的均匀分布）
            input_data = tf.random.uniform(
                shape=test_case['input_shape'],
                minval=-1.0,
                maxval=1.0,
                dtype=tf.float32
            )

            # 通过层进行前向传播
            output_data = test_case['layer'](input_data)

            # 保存到数组
            all_inputs[i] = input_data.numpy()
            all_outputs[i] = output_data.numpy()

        # 保存为32位numpy数组
        input_file = os.path.join(test_dir, f"{test_case['name'].lower()}_inputs_32bit.npy")
        output_file = os.path.join(test_dir, f"{test_case['name'].lower()}_outputs_32bit.npy")

        np.save(input_file, all_inputs)
        np.save(output_file, all_outputs)

        print(f"输入已保存: {input_file}")
        print(f"输出已保存: {output_file}")
        print(f"输入数据类型: {all_inputs.dtype}, 形状: {all_inputs.shape}")
        print(f"输出数据类型: {all_outputs.dtype}, 形状: {all_outputs.shape}")

        # 验证前几个样本
        print("样本0输入前3值:", all_inputs[0].flatten()[:3])
        print("样本0输出前3值:", all_outputs[0].flatten()[:3])

        # 验证Softmax输出和为1
        if test_case['name'] == "Softmax":
            sample_sums = np.sum(all_outputs, axis=2)  # 对每行的10个元素求和
            print(f"Softmax验证 - 样本0行和: {sample_sums[0]}")
            print(f"Softmax验证 - 所有行和接近1: {np.allclose(sample_sums, 1.0, atol=1e-6)}")

        # 验证Sigmoid输出范围
        if test_case['name'] == "Sigmoid":
            in_range = np.all(all_outputs >= 0) and np.all(all_outputs <= 1)
            print(f"Sigmoid验证 - 输出在0-1范围内: {in_range}")

    print(f"\n测试完成！数据保存在: {test_dir}")


def load_and_verify_data(test_dir="simple_layer_tests"):
    """加载并验证保存的数据"""
    if not os.path.exists(test_dir):
        print(f"测试目录不存在: {test_dir}")
        return

    files = os.listdir(test_dir)
    print(f"\n加载验证数据:")
    print("=" * 50)

    for file in sorted(files):
        if file.endswith(".npy"):
            filepath = os.path.join(test_dir, file)
            data = np.load(filepath)
            print(f"{file}:")
            print(f"  形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            print(f"  数值范围: [{np.min(data):.4f}, {np.max(data):.4f}]")

            # 验证是32位浮点数
            if data.dtype == np.float32:
                print("  ✓ 32位验证通过")
            else:
                print("  ✗ 不是32位数据")


if __name__ == "__main__":
    # 运行简化测试
    test_layers_simple()

    # 加载验证数据
    load_and_verify_data()