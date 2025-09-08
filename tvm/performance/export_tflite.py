import os
import tvm
from tvm import relay
import numpy as np
import random
import yaml
import math
import tensorflow as tf
import tflite
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# 当前脚本所在目录
cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/op_lib"


class ParameterProvider:
    """参数提供器，基于配置文件生成参数"""

    def __init__(self, config, seed=None):
        self.config = config
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.generated_params = {}

    def get(self, name, group_name, operator_name, min_val=None, max_val=None):
        if name in self.generated_params:
            return self.generated_params[name]

        try:
            param_config = self.config[operator_name][group_name][name]
            config_min, config_max = param_config

            if min_val is not None:
                config_min = max(config_min, min_val)
            if max_val is not None:
                config_max = min(config_max, max_val)

            min_val, max_val = config_min, config_max
        except KeyError:
            if min_val is None or max_val is None:
                if "channel" in name:
                    min_val, max_val = 1, 16
                elif "dim" in name or "size" in name or "feature" in name:
                    min_val, max_val = 1, 128
                elif "kernel" in name:
                    min_val, max_val = 1, 7
                elif "scale" in name:
                    min_val, max_val = 1, 4
                elif "rate" in name:
                    min_val, max_val = 0.1, 0.5
                elif "alpha" in name:
                    min_val, max_val = 0.01, 0.3
                elif "stride" in name:
                    min_val, max_val = 1, 3
                else:
                    min_val, max_val = 1, 64

        if min_val > max_val:
            min_val, max_val = max_val, min_val

        if isinstance(min_val, float) or isinstance(max_val, float):
            value = self.rng.uniform(min_val, max_val)
        else:
            value = self.rng.randint(min_val, max_val)

        if "kernel" in name and value % 2 == 0:
            value += 1

        self.generated_params[name] = value
        return value


def export_tvm_module(op_name, mod, params):
    """导出TVM模块"""
    # 执行类型推断
    mod = relay.transform.InferType()(mod)
    entry_func = mod["main"]

    # 构建配置内容
    config_lines = []
    config_lines.append(f"[operator]")
    config_lines.append(f"name = {op_name}")

    config_lines.append("\n[inputs]")
    for i, var in enumerate(entry_func.params):
        shape_str = ",".join(str(dim) for dim in var.type_annotation.shape)
        config_lines.append(f"input_{i}_name = {var.name_hint}")
        config_lines.append(f"input_{i}_shape = {shape_str}")
        config_lines.append(f"input_{i}_dtype = {var.type_annotation.dtype}")

    config_lines.append("\n[outputs]")
    if isinstance(entry_func.ret_type, relay.TupleType):
        for j, field in enumerate(entry_func.ret_type.fields):
            shape_str = ",".join(str(dim) for dim in field.shape)
            config_lines.append(f"output_{j}_shape = {shape_str}")
            config_lines.append(f"output_{j}_dtype = {field.dtype}")
    else:
        shape_str = ",".join(str(dim) for dim in entry_func.ret_type.shape)
        config_lines.append(f"output_0_shape = {shape_str}")
        config_lines.append(f"output_0_dtype = {entry_func.ret_type.dtype}")

    # 保存配置信息
    config_path_native = os.path.join(native_dir, f"{op_name}.cfg")
    config_path_device = os.path.join(device_dir, f"{op_name}.cfg")

    with open(config_path_native, "w") as f:
        f.write("\n".join(config_lines))
    with open(config_path_device, "w") as f:
        f.write("\n".join(config_lines))

    # 编译并导出模型
    # 平台1: 本地目标
    target_native = tvm.target.Target("llvm")
    cc_native = "/usr/bin/g++"

    with tvm.transform.PassContext(opt_level=3):
        lib_native = relay.build(mod, target=target_native, params=params)

    so_path_native = os.path.join(native_dir, f"{op_name}.so")
    lib_native.export_library(so_path_native, cc=cc_native)

    # 平台2: 设备目标
    target_device = tvm.target.Target(
        "llvm -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c"
    )
    cc_device = "/usr/bin/riscv64-linux-gnu-g++"

    with tvm.transform.PassContext(opt_level=3):
        lib_device = relay.build(mod, target=target_device, params=params)

    so_path_device = os.path.join(device_dir, f"{op_name}.so")
    lib_device.export_library(so_path_device, cc=cc_device)

    print(f"成功导出: {op_name} [native: {so_path_native}, device: {so_path_device}]")


def convert_tflite_to_tvm(tflite_model, op_name):
    """将TFLite模型转换为TVM格式"""
    try:
        # 解析 TFLite 模型
        model = tflite.Model.GetRootAsModel(bytearray(tflite_model), 0)

        # 转换为 Relay 模块
        mod, params = relay.frontend.from_tflite(
            model,
            shape_dict=None,
            dtype_dict=None
        )

        # 导出TVM模块
        export_tvm_module(op_name, mod, params)
        return True
    except Exception as e:
        print(f"❌ 转换为TVM失败: {str(e)}")
        return False


def build_tf_model(params, operator_name):
    """使用原生TensorFlow构建模型"""
    op_name = f"{operator_name}_{params['group_name']}"

    try:
        # 根据算子类型定义构建函数
        if operator_name == "conv2d":
            # 定义输入规格
            input_specs = [{
                'shape': (params["batch"], params["input_height"], params["input_width"], params["in_channels"]),
                'dtype': 'float32',
                'name': 'input'
            }]

            # 创建卷积核变量
            kernel_shape = (params["kernel_size"], params["kernel_size"], params["in_channels"], params["out_channels"])
            kernel = tf.Variable(tf.random.normal(kernel_shape), name="kernel")

            # 构建计算图函数
            def build_graph(inputs):
                return tf.nn.conv2d(
                    inputs[0],
                    kernel,
                    strides=[1, params["stride"], params["stride"], 1],
                    padding="SAME",
                    name="conv_output"
                )

        elif operator_name == "relu":
            # 定义输入规格
            input_specs = [{
                'shape': (params["batch"], params["input_height"], params["input_width"], params["channels"]),
                'dtype': 'float32',
                'name': 'input'
            }]

            # 构建计算图函数
            def build_graph(inputs):
                return tf.nn.relu(inputs[0], name="relu_output")

        elif operator_name == "batch_norm":
            # 定义输入规格
            input_specs = [{
                'shape': (params["batch"], params["input_height"], params["input_width"], params["channels"]),
                'dtype': 'float32',
                'name': 'input'
            }]

            # 创建BatchNorm参数
            scale = tf.Variable(tf.ones([params["channels"]]), name="scale")
            offset = tf.Variable(tf.zeros([params["channels"]]), name="offset")
            mean = tf.Variable(tf.zeros([params["channels"]]), name="mean")
            variance = tf.Variable(tf.ones([params["channels"]]), name="variance")

            # 构建计算图函数
            def build_graph(inputs):
                return tf.nn.batch_normalization(
                    inputs[0],
                    mean=mean,
                    variance=variance,
                    offset=offset,
                    scale=scale,
                    variance_epsilon=1e-6,
                    name="bn_output"
                )

        elif operator_name == "dense":
            # 定义输入规格
            input_specs = [{
                'shape': (params["batch"], params["in_features"]),
                'dtype': 'float32',
                'name': 'input'
            }]

            # 创建权重变量
            weight_shape = (params["in_features"], params["out_features"])
            weight = tf.Variable(tf.random.normal(weight_shape), name="weight")

            # 构建计算图函数
            def build_graph(inputs):
                return tf.matmul(inputs[0], weight, name="dense_output")

        elif operator_name == "softmax":
            # 定义输入规格
            input_specs = [{
                'shape': (params["batch"], params["features"]),
                'dtype': 'float32',
                'name': 'input'
            }]

            # 构建计算图函数
            def build_graph(inputs):
                return tf.nn.softmax(inputs[0], name="softmax_output")

        elif operator_name == "avg_pool2d":
            # 定义输入规格
            input_specs = [{
                'shape': (params["batch"], params["input_height"], params["input_width"], params["pool_channels"]),
                'dtype': 'float32',
                'name': 'input'
            }]

            # 构建计算图函数
            def build_graph(inputs):
                return tf.nn.avg_pool2d(
                    inputs[0],
                    ksize=[1, params["pool_size"], params["pool_size"], 1],
                    strides=[1, params["stride"], params["stride"], 1],
                    padding="SAME",
                    name="avg_pool_output"
                )

        elif operator_name == "bias_add":
            # 定义输入规格
            input_specs = [{
                'shape': (params["batch"], params["input_height"], params["input_width"], params["channels"]),
                'dtype': 'float32',
                'name': 'input'
            }]

            # 创建偏置变量
            bias = tf.Variable(tf.zeros([params["channels"]]), name="bias")

            # 构建计算图函数
            def build_graph(inputs):
                return tf.nn.bias_add(inputs[0], bias, name="bias_add_output")

        elif operator_name == "matmul":
            # 定义输入规格
            input_specs = [{
                'shape': (params["batch"], params["M"], params["K"]),
                'dtype': 'float32',
                'name': 'input'
            }]

            # 创建权重变量
            weight_shape = (params["K"], params["N"])
            weight = tf.Variable(tf.random.normal(weight_shape), name="weight")

            # 构建计算图函数
            def build_graph(inputs):
                return tf.matmul(inputs[0], weight, name="matmul_output")

        else:
            raise ValueError(f"未知的算子类型: {operator_name}")

        # 动态创建模型类
        model_class = type(
            "DynamicModel",
            (tf.Module,),
            {
                "__call__": tf.function(
                    lambda self, *inputs: build_graph(inputs),
                    input_signature=[
                        tf.TensorSpec(
                            shape=spec['shape'],
                            dtype=getattr(tf, spec['dtype']) if isinstance(spec['dtype'], str) else spec['dtype'],
                            name=spec.get('name', None)
                        ) for spec in input_specs
                    ]
                )
            }
        )

        return model_class(), op_name
    except Exception as e:
        print(f"❌ 构建TensorFlow模型失败: {str(e)}")
        return None, None


def convert_to_tflite(model, op_name):
    """将TensorFlow模型转换为TFLite格式"""
    try:
        # 转换为具体函数
        concrete_func = model.__call__.get_concrete_function()

        # 转换为TFLite模型
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()

        # 保存TFLite模型
        tflite_path = os.path.join(native_dir, f"{op_name}.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"✅ 生成: {op_name}.tflite")

        return tflite_model
    except Exception as e:
        print(f"❌ 转换为TFLite失败: {str(e)}")
        return None


def calculate_min_input_size(kernel_size, stride, padding):
    min_size = kernel_size - 2 * padding
    return max(1, min_size)


def generate_params_from_config(provider, operator_name, group_name):
    """根据配置生成参数"""
    params = {"group_name": group_name}
    params["dtype"] = "float32"
    params["batch"] = provider.get("batch", group_name, operator_name)

    if operator_name == "conv2d":
        params["in_channels"] = provider.get("in_channels", group_name, operator_name)
        params["out_channels"] = provider.get("out_channels", group_name, operator_name)
        params["kernel_size"] = provider.get("kernel_size", group_name, operator_name)
        params["stride"] = provider.get("stride", group_name, operator_name)
        params["padding"] = params["kernel_size"] // 2

        min_input_size = calculate_min_input_size(
            params["kernel_size"], params["stride"], params["padding"]
        )
        params["input_height"] = provider.get(
            "input_height", group_name, operator_name, min_val=min_input_size
        )
        params["input_width"] = provider.get(
            "input_width", group_name, operator_name, min_val=min_input_size
        )

    elif operator_name == "relu":
        params["channels"] = provider.get("channels", group_name, operator_name)
        params["input_height"] = provider.get("input_height", group_name, operator_name)
        params["input_width"] = provider.get("input_width", group_name, operator_name)

    elif operator_name == "batch_norm":
        params["channels"] = provider.get("channels", group_name, operator_name)
        params["input_height"] = provider.get("input_height", group_name, operator_name)
        params["input_width"] = provider.get("input_width", group_name, operator_name)

    elif operator_name == "dense":
        params["in_features"] = provider.get("in_features", group_name, operator_name)
        params["out_features"] = provider.get("out_features", group_name, operator_name)

    elif operator_name == "softmax":
        params["features"] = provider.get("features", group_name, operator_name)

    elif operator_name == "avg_pool2d":
        params["pool_channels"] = provider.get("pool_channels", group_name, operator_name)
        params["pool_size"] = provider.get("pool_size", group_name, operator_name)
        params["stride"] = provider.get("stride", group_name, operator_name)
        params["padding"] = params["pool_size"] // 2

        min_input_size = calculate_min_input_size(
            params["pool_size"], params["stride"], params["padding"]
        )
        params["input_height"] = provider.get(
            "input_height", group_name, operator_name, min_val=min_input_size
        )
        params["input_width"] = provider.get(
            "input_width", group_name, operator_name, min_val=min_input_size
        )

    elif operator_name == "bias_add":
        params["channels"] = provider.get("channels", group_name, operator_name)
        params["input_height"] = provider.get("input_height", group_name, operator_name)
        params["input_width"] = provider.get("input_width", group_name, operator_name)

    elif operator_name == "matmul":
        params["M"] = provider.get("M", group_name, operator_name)
        params["N"] = provider.get("N", group_name, operator_name)
        params["K"] = provider.get("K", group_name, operator_name)

    return params


# 主函数
if __name__ == "__main__":
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 生成10套算子
    for i in range(10):
        print(f"生成第 {i + 1}/10 套算子...")

        # 为当前算子集创建目录
        set_dir = os.path.join(cur_dir, f"set_{i}")
        global native_dir, device_dir
        native_dir = os.path.join(set_dir, "native")
        device_dir = os.path.join(set_dir, "device")
        os.makedirs(native_dir, exist_ok=True)
        os.makedirs(device_dir, exist_ok=True)

        print(f"导出到: {native_dir} 和 {device_dir}")

        # 遍历所有算子类型
        for operator_name, groups in config.items():
            # 遍历该算子的所有配置组
            for group_name in groups.keys():
                # 创建参数提供器
                provider = ParameterProvider(config, seed=42 + i)

                try:
                    # 生成参数
                    params = generate_params_from_config(provider, operator_name, group_name)

                    # 构建TensorFlow模型
                    model, op_name = build_tf_model(params, operator_name)
                    if model is None:
                        continue

                    # 转换为TFLite格式
                    tflite_model = convert_to_tflite(model, op_name)
                    if tflite_model is None:
                        continue

                    # 转换为TVM格式
                    convert_tflite_to_tvm(tflite_model, op_name)
                except Exception as e:
                    print(f"生成算子 {operator_name}_{group_name} 失败: {str(e)}")
                    continue

    print("所有算子导出成功!")
    print(f"算子库保存在: {cur_dir}/set_*")