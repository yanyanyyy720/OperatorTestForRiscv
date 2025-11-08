import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# TVM相关导入
try:
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor
    import tflite

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    print("警告: TVM未安装，TVM导出功能将不可用")

# 平台定义
TARGET_PLATFORMS = {
    "rv": {
        "target": tvm.target.Target(
            "llvm -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c"
        ),
        "cc": "/usr/bin/riscv64-linux-gnu-g++",
        "dir": "rv"
    },
    "rvv": {
        "target": tvm.target.Target(
            "llvm -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c,+v"
        ),
        "cc": "/usr/bin/riscv64-linux-gnu-g++",
        "dir": "rvv"
    }
}


# 定义自定义Layer来包装TF函数
class AvgPoolNCHW(layers.Layer):
    """NCHW格式的平均池化"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.avg_pool(inputs, 2, 2, 'VALID', data_format='NCHW')


class MaxPoolNCHW(layers.Layer):
    """NCHW格式的最大池化"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.max_pool(inputs, 2, 2, 'VALID', data_format='NCHW')


class SoftmaxLayer(layers.Layer):
    """Softmax层"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.softmax(inputs)


class ReluLayer(layers.Layer):
    """ReLU层"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.relu(inputs)


class ReduceMeanLayer(layers.Layer):
    """沿axis=1的reduce_mean"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, 1, keepdims=False)


class ReduceMaxLayer(layers.Layer):
    """沿axis=1的reduce_max"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_max(inputs, 1, keepdims=False)


class SigmoidLayer(layers.Layer):
    """Sigmoid层"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.sigmoid(inputs)


class TanhLayer(layers.Layer):
    """Tanh层"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.tanh(inputs)


class SimpleOperatorExporter:
    """导出不需要参数的简单算子"""

    def __init__(self):
        # 只包含不需要参数的算子
        self.simple_ops = [
            'avg_pool',
            'max_pool',
            'softmax',
            'relu',
            'reduce_mean',
            'reduce_max',
            'sigmoid',
            'tanh'
        ]

    def build_model(self, op_name, input_shape=(1, 4, 32, 32)):
        """
        构建TF模型
        Args:
            op_name: 算子名称
            input_shape: 输入形状，默认为NCHW格式
        """
        if op_name == 'avg_pool':
            # 输入: NCHW格式
            inputs = layers.Input(shape=input_shape[1:], batch_size=input_shape[0])
            outputs = AvgPoolNCHW()(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        elif op_name == 'max_pool':
            inputs = layers.Input(shape=input_shape[1:], batch_size=input_shape[0])
            outputs = MaxPoolNCHW()(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        elif op_name == 'softmax':
            # Softmax通常用于1D或2D数据
            inputs = layers.Input(shape=(4,10))
            outputs = SoftmaxLayer()(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        elif op_name == 'relu':
            inputs = layers.Input(shape=input_shape[1:], batch_size=input_shape[0])
            outputs = ReluLayer()(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        elif op_name == 'reduce_mean':
            # reduce_mean在axis=1上操作
            inputs = layers.Input(shape=input_shape[1:], batch_size=input_shape[0])
            outputs = ReduceMeanLayer()(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        elif op_name == 'reduce_max':
            inputs = layers.Input(shape=input_shape[1:], batch_size=input_shape[0])
            outputs = ReduceMaxLayer()(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        elif op_name == 'sigmoid':
            inputs = layers.Input(shape=(4,1))
            outputs = SigmoidLayer()(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        elif op_name == 'tanh':
            inputs = layers.Input(shape=input_shape[1:], batch_size=input_shape[0])
            outputs = TanhLayer()(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        else:
            raise ValueError(f"不支持的算子: {op_name}")

    def convert_to_tflite(self, model):
        """将Keras模型转换为TFLite"""
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = []  # 不使用优化
            tflite_model = converter.convert()
            return tflite_model
        except Exception as e:
            print(f"TFLite转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_tflite_to_tvm(self, tflite_model, op_name, platform_config, output_dir):
        """将TFLite模型转换为TVM格式"""
        if not TVM_AVAILABLE:
            print("TVM不可用")
            return {}

        try:
            # 将TFLite模型转换为字节数组
            tflite_model_buf = bytearray(tflite_model)

            # 使用TFLite前端导入模型
            tflite_model_obj = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
            mod, params = relay.frontend.from_tflite(tflite_model_obj)

            # 编译模型
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=platform_config["target"], params=params)

            # 保存文件
            base_name = op_name
            base_path = os.path.join(output_dir, base_name)

            # 保存.so文件
            so_path = f"{base_path}.so"
            lib.export_library(so_path, cc=platform_config["cc"])

            # 保存图JSON
            graph_json_path = f"{base_path}.json"
            with open(graph_json_path, 'w') as f:
                f.write(lib.get_graph_json())

            # 保存参数
            params_path = f"{base_path}.params"
            with open(params_path, 'wb') as f:
                f.write(tvm.runtime.save_param_dict(lib.get_params()))

            return {
                'so_path': so_path,
                'graph_json_path': graph_json_path,
                'params_path': params_path
            }

        except Exception as e:
            print(f"TVM转换失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def export_operator(self, op_name, output_dir="./exported_models"):
        """
        导出单个算子到TFLite和TVM
        """
        print(f"\n{'=' * 60}")
        print(f"开始导出算子: {op_name}")
        print(f"{'=' * 60}")

        # 创建输出目录
        tflite_dir = os.path.join(output_dir, "tflite")
        os.makedirs(tflite_dir, exist_ok=True)

        result = {
            'op_name': op_name,
            'tflite_path': None,
            'tvm_paths': {},
            'status': 'failed'
        }

        try:
            # 1. 构建模型
            print(f"📦 步骤1: 构建TensorFlow模型...")
            if op_name in ['softmax']:
                input_shape = (1, 10)
            else:
                input_shape = (1, 4, 32, 32)

            model = self.build_model(op_name, input_shape)
            print(f"   ✓ 模型构建成功")
            print(f"   输入形状: {model.input_shape}")
            print(f"   输出形状: {model.output_shape}")

            # 2. 转换为TFLite
            print(f"\n📱 步骤2: 转换为TFLite格式...")
            tflite_model = self.convert_to_tflite(model)

            if tflite_model is not None:
                tflite_path = os.path.join(tflite_dir, f"{op_name}.tflite")
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                result['tflite_path'] = tflite_path
                print(f"   ✓ TFLite模型已保存: {tflite_path}")
                print(f"   文件大小: {len(tflite_model)} bytes")
            else:
                print(f"   ✗ TFLite转换失败")
                return result

            # 3. 转换为TVM（两个平台）
            if TVM_AVAILABLE:
                print(f"\n🚀 步骤3: 转换为TVM格式...")

                for platform_name, platform_config in TARGET_PLATFORMS.items():
                    print(f"\n   平台: {platform_name}")
                    tvm_dir = os.path.join(output_dir, "tvm", platform_name)
                    os.makedirs(tvm_dir, exist_ok=True)

                    tvm_paths = self.convert_tflite_to_tvm(
                        tflite_model, op_name, platform_config, tvm_dir
                    )

                    if tvm_paths:
                        result['tvm_paths'][platform_name] = tvm_paths
                        print(f"   ✓ TVM {platform_name} 模型导出成功")
                        print(f"      - SO文件: {tvm_paths['so_path']}")
                        print(f"      - JSON文件: {tvm_paths['graph_json_path']}")
                        print(f"      - 参数文件: {tvm_paths['params_path']}")
                    else:
                        print(f"   ✗ TVM {platform_name} 转换失败")
            else:
                print(f"\n⚠️  TVM不可用，跳过TVM导出")

            result['status'] = 'success'
            print(f"\n✅ 算子 {op_name} 导出完成!")

        except Exception as e:
            print(f"\n❌ 导出失败: {e}")
            result['error'] = str(e)
            import traceback
            traceback.print_exc()

        return result

    def export_all_operators(self, output_dir="./exported_models"):
        """导出所有简单算子"""
        print(f"\n🎯 开始导出所有简单算子")
        print(f"输出目录: {output_dir}")
        print(f"算子列表: {', '.join(self.simple_ops)}\n")

        all_results = []

        for op_name in self.simple_ops:
            result = self.export_operator(op_name, output_dir)
            all_results.append(result)

        # 生成摘要
        success_count = sum(1 for r in all_results if r['status'] == 'success')
        failed_count = len(all_results) - success_count

        summary = {
            'total_ops': len(self.simple_ops),
            'success_count': success_count,
            'failed_count': failed_count,
            'results': all_results
        }

        # 保存摘要
        summary_path = os.path.join(output_dir, "export_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # 打印总结
        print(f"\n{'=' * 60}")
        print(f"📊 导出总结")
        print(f"{'=' * 60}")
        print(f"总算子数: {len(self.simple_ops)}")
        print(f"成功: {success_count}")
        print(f"失败: {failed_count}")
        print(f"成功率: {success_count / len(self.simple_ops) * 100:.1f}%")
        print(f"\n摘要已保存到: {summary_path}")

        return summary


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="导出简单TF算子到TFLite和TVM")
    parser.add_argument("--op", type=str, help="导出指定算子")
    parser.add_argument("--output-dir", type=str, default="./exported_models",
                        help="输出目录")
    parser.add_argument("--list", action="store_true", help="列出所有支持的算子")

    args = parser.parse_args()

    exporter = SimpleOperatorExporter()

    if args.list:
        print("📋 支持的简单算子:")
        for i, op in enumerate(exporter.simple_ops, 1):
            print(f"  {i}. {op}")
        return

    if args.op:
        if args.op in exporter.simple_ops:
            exporter.export_operator(args.op, args.output_dir)
        else:
            print(f"❌ 不支持的算子: {args.op}")
            print(f"支持的算子: {', '.join(exporter.simple_ops)}")
    else:
        # 导出所有算子
        exporter.export_all_operators(args.output_dir)


if __name__ == "__main__":
    main()
