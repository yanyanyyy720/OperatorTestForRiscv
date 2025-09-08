import os
import tvm
from tvm import relay
import numpy as np
import random
import yaml
import math

# 当前脚本所在目录
cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/op_lib"


class ParameterProvider:
    """参数提供器，基于配置文件生成参数"""

    def __init__(self, config, seed=None):
        """
        config: 从配置文件解析出的字典结构
        seed: 随机种子
        """
        self.config = config
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.generated_params = {}

    def get(self, name, group_name, operator_name, min_val=None, max_val=None):
        """获取参数值，优先使用配置文件中的范围"""
        # 如果参数已生成，直接返回
        if name in self.generated_params:
            return self.generated_params[name]

        # 从配置中获取参数范围
        try:
            param_config = self.config[operator_name][group_name][name]
            config_min, config_max = param_config

            # 如果提供了min_val/max_val，确保在配置范围内
            if min_val is not None:
                config_min = max(config_min, min_val)
            if max_val is not None:
                config_max = min(config_max, max_val)

            min_val, max_val = config_min, config_max
        except KeyError:
            # 如果配置中没有该参数，使用传入的范围或备用逻辑
            if min_val is None or max_val is None:
                # 处理特殊参数（备用逻辑）
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
                    min_val, max_val = 1, 64  # 默认范围

        # 确保范围有效
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        # 生成参数值
        if isinstance(min_val, float) or isinstance(max_val, float):
            value = self.rng.uniform(min_val, max_val)
        else:
            value = self.rng.randint(min_val, max_val)

        # 特殊处理：确保卷积核为奇数
        if "kernel" in name and value % 2 == 0:
            value += 1

        self.generated_params[name] = value
        return value


def export_op(func, op_name, input_vars, params=None):
    """导出算子到两个平台（本地和设备）"""
    # 1. 创建函数并构造IR模块
    func = relay.Function(input_vars, func)
    mod = tvm.IRModule.from_expr(func)

    # 2. 执行类型推断以获取输出配置信息
    mod = relay.transform.InferType()(mod)
    entry_func = mod["main"]

    # 3. 构建配置内容
    config_lines = []
    config_lines.append(f"[operator]")
    config_lines.append(f"name = {op_name}")

    config_lines.append("\n[inputs]")
    for i, var in enumerate(input_vars):
        shape_str = ",".join(str(dim) for dim in var.type_annotation.shape)
        config_lines.append(f"input_{i}_name = {var.name_hint}")
        config_lines.append(f"input_{i}_shape = {shape_str}")
        config_lines.append(f"input_{i}_dtype = {var.type_annotation.dtype}")

    config_lines.append("\n[outputs]")
    # 处理单输出和多输出(TupleType)两种情况
    if isinstance(entry_func.ret_type, relay.TupleType):
        for j, field in enumerate(entry_func.ret_type.fields):
            shape_str = ",".join(str(dim) for dim in field.shape)
            config_lines.append(f"output_{j}_shape = {shape_str}")
            config_lines.append(f"output_{j}_dtype = {field.dtype}")
    else:
        shape_str = ",".join(str(dim) for dim in entry_func.ret_type.shape)
        config_lines.append(f"output_0_shape = {shape_str}")
        config_lines.append(f"output_0_dtype = {entry_func.ret_type.dtype}")

    # 4. 保存配置信息到INI文件（两个平台共享相同的配置）
    config_path_native = os.path.join(native_dir, f"{op_name}.cfg")
    config_path_device = os.path.join(device_dir, f"{op_name}.cfg")

    with open(config_path_native, "w") as f:
        f.write("\n".join(config_lines))
    with open(config_path_device, "w") as f:
        f.write("\n".join(config_lines))

    # 5. 编译并导出模型到两个平台
    # 平台1: 本地目标
    target_native = tvm.target.Target("llvm")
    cc_native = "/usr/bin/g++"

    with tvm.transform.PassContext(opt_level=3):
        lib_native = relay.build(mod, target=target_native)

    so_path_native = os.path.join(native_dir, f"{op_name}.so")
    lib_native.export_library(so_path_native, cc=cc_native)

    # 平台2: 设备目标
    target_device = tvm.target.Target(
        "llvm -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c"
    )
    cc_device = "/usr/bin/riscv64-linux-gnu-g++"

    with tvm.transform.PassContext(opt_level=3):
        lib_device = relay.build(mod, target=target_device)

    so_path_device = os.path.join(device_dir, f"{op_name}.so")
    lib_device.export_library(so_path_device, cc=cc_device)

    print(f"成功导出: {op_name} [native: {so_path_native}, device: {so_path_device}]")


def calculate_min_input_size(kernel_size, stride, padding):
    """计算最小输入尺寸以确保输出尺寸至少为1"""
    # 输出尺寸公式: (input_size + 2*padding - kernel_size) // stride + 1
    # 要确保输出尺寸 >= 1，则:
    #   input_size + 2*padding - kernel_size >= 0
    #   input_size >= kernel_size - 2*padding
    min_size = kernel_size - 2 * padding
    return max(1, min_size)  # 确保至少为1


def generate_params_from_config(provider, operator_name, group_name):
    """根据配置生成参数"""
    params = {}

    # 通用参数
    # params["batch"] = 1
    params["dtype"] = "float32"
    params["batch"] = provider.get("batch",group_name,operator_name)
    # 算子特定参数
    if operator_name == "conv2d":
        params["in_channels"] = provider.get("in_channels", group_name, operator_name)
        params["out_channels"] = provider.get("out_channels", group_name, operator_name)
        params["kernel_size"] = provider.get("kernel_size", group_name, operator_name)
        params["stride"] = provider.get("stride", group_name, operator_name)
        params["padding"] = params["kernel_size"] // 2  # 自动计算填充

        # 计算最小输入尺寸
        min_input_size = calculate_min_input_size(
            params["kernel_size"], params["stride"], params["padding"]
        )

        # 获取输入尺寸，确保满足最小要求
        params["input_height"] = provider.get(
            "input_height", group_name, operator_name,
            min_val=min_input_size
        )
        params["input_width"] = provider.get(
            "input_width", group_name, operator_name,
            min_val=min_input_size
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
        params["padding"] = params["pool_size"] // 2  # 自动计算填充

        # 计算最小输入尺寸
        min_input_size = calculate_min_input_size(
            params["pool_size"], params["stride"], params["padding"]
        )

        # 获取输入尺寸，确保满足最小要求
        params["input_height"] = provider.get(
            "input_height", group_name, operator_name,
            min_val=min_input_size
        )
        params["input_width"] = provider.get(
            "input_width", group_name, operator_name,
            min_val=min_input_size
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


def export_conv2d(params):
    """导出卷积算子"""
    # 计算输出尺寸以验证参数有效性
    output_height = (params["input_height"] + 2 * params["padding"] - params["kernel_size"]) // params["stride"] + 1
    output_width = (params["input_width"] + 2 * params["padding"] - params["kernel_size"]) // params["stride"] + 1

    # 确保输出尺寸有效
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"无效的卷积参数: 输入尺寸({params['input_height']}x{params['input_width']}), "
                         f"卷积核({params['kernel_size']}), 步长({params['stride']}), "
                         f"填充({params['padding']})导致输出尺寸为{output_height}x{output_width}")

    data = relay.var("data",
                     shape=(params["batch"], params["in_channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    weight = relay.var("weight",
                       shape=(params["out_channels"], params["in_channels"],
                              params["kernel_size"], params["kernel_size"]),
                       dtype=params["dtype"])

    conv = relay.nn.conv2d(
        data,
        weight,
        channels=params["out_channels"],
        kernel_size=(params["kernel_size"], params["kernel_size"]),
        strides=(params["stride"], params["stride"]),
        padding=(params["padding"], params["padding"])
    )

    export_op(conv, params["op_name"], [data, weight], params)


def export_relu(params):
    """导出ReLU算子"""
    data = relay.var("data",
                     shape=(params["batch"], params["channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    relu = relay.nn.relu(data)
    export_op(relu, params["op_name"], [data], params)


def export_batch_norm(params):
    """导出批归一化算子"""
    data = relay.var("data",
                     shape=(params["batch"], params["channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    gamma = relay.var("gamma", shape=(params["channels"],), dtype=params["dtype"])
    beta = relay.var("beta", shape=(params["channels"],), dtype=params["dtype"])
    moving_mean = relay.var("moving_mean", shape=(params["channels"],), dtype=params["dtype"])
    moving_var = relay.var("moving_var", shape=(params["channels"],), dtype=params["dtype"])

    bn = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var)[0]
    export_op(bn, params["op_name"], [data, gamma, beta, moving_mean, moving_var], params)


def export_dense(params):
    """导出全连接层算子"""
    data = relay.var("data", shape=(params["batch"], params["in_features"]), dtype=params["dtype"])
    weight = relay.var("weight", shape=(params["out_features"], params["in_features"]), dtype=params["dtype"])

    dense = relay.nn.dense(data, weight, units=params["out_features"])
    export_op(dense, params["op_name"], [data, weight], params)


def export_softmax(params):
    """导出Softmax算子"""
    data = relay.var("data", shape=(params["batch"], params["features"]), dtype=params["dtype"])
    softmax = relay.nn.softmax(data)
    export_op(softmax, params["op_name"], [data], params)


def export_avg_pool2d(params):
    """导出平均池化算子"""
    # 计算输出尺寸以验证参数有效性
    output_height = (params["input_height"] + 2 * params["padding"] - params["pool_size"]) // params["stride"] + 1
    output_width = (params["input_width"] + 2 * params["padding"] - params["pool_size"]) // params["stride"] + 1

    # 确保输出尺寸有效
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"无效的池化参数: 输入尺寸({params['input_height']}x{params['input_width']}), "
                         f"池化尺寸({params['pool_size']}), 步长({params['stride']}), "
                         f"填充({params['padding']})导致输出尺寸为{output_height}x{output_width}")

    data = relay.var("data",
                     shape=(params["batch"], params["pool_channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    pool = relay.nn.avg_pool2d(
        data,
        pool_size=(params["pool_size"], params["pool_size"]),
        strides=(params["stride"], params["stride"]),
        padding=(params["padding"], params["padding"])
    )

    export_op(pool, params["op_name"], [data], params)


def export_bias_add(params):
    """导出偏置加法算子"""
    data = relay.var("data",
                     shape=(params["batch"], params["channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    bias = relay.var("bias", shape=(params["channels"],), dtype=params["dtype"])
    bias_add = relay.nn.bias_add(data, bias)
    export_op(bias_add, params["op_name"], [data, bias], params)


def export_matmul(params):
    """导出矩阵乘法算子"""
    a = relay.var("a", shape=(params["M"], params["K"]), dtype=params["dtype"])
    b = relay.var("b", shape=(params["K"], params["N"]), dtype=params["dtype"])
    matmul = relay.nn.matmul(a, b)
    export_op(matmul, params["op_name"], [a, b], params)


# 算子导出函数映射
EXPORT_FUNCTIONS = {
    "conv2d": export_conv2d,
    "relu": export_relu,
    "batch_norm": export_batch_norm,
    "dense": export_dense,
    "softmax": export_softmax,
    "avg_pool2d": export_avg_pool2d,
    "bias_add": export_bias_add,
    "matmul": export_matmul
}

# 主函数
if __name__ == "__main__":
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 生成1000套算子
    for i in range(10):
        print(f"生成第 {i + 1}/10 套算子...")

        # 为当前算子集创建目录
        set_dir = os.path.join(cur_dir , f"op_lib/set_{i}")
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
                    params["op_name"] = f"{operator_name}_{group_name}"

                    # 导出算子
                    export_func = EXPORT_FUNCTIONS[operator_name]
                    export_func(params)
                except Exception as e:
                    print(f"生成算子 {operator_name}_{group_name} 失败: {str(e)}")
                    continue

    print("所有算子导出成功!")
    print(f"算子库保存在: {cur_dir}/set_*")