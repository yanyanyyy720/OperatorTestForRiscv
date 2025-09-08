import os
import tvm
from tvm import relay
import numpy as np
from abc import ABC, abstractmethod
import random

# 当前脚本所在目录
cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/op_lib"


class ParameterProvider(ABC):
    """参数提供器抽象基类，支持随机和固定参数模式"""

    @abstractmethod
    def get(self, name, default=None, min_val=None, max_val=None):
        pass


class RandomParameterProvider(ParameterProvider):
    """随机参数提供器"""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.generated_params = {}

    def get(self, name, default=None, min_val=None, max_val=None):
        # 如果参数已生成，直接返回
        if name in self.generated_params:
            return self.generated_params[name]

        # 否则生成新参数
        if min_val is not None and max_val is not None:
            if max_val < 1:
                value = self.rng.uniform(min_val, max_val)
            else:
                value = self.rng.randint(min_val, max_val)
        elif default is not None:
            value = default
        else:
            # 处理特殊参数
            if "channel" in name:
                value = self.rng.randint(1, 16)
            elif "dim" in name or "size" in name or "feature" in name:
                value = self.rng.randint(1, 128)
            elif "kernel" in name:
                value = self.rng.randint(1, 7) * 2 + 1  # 奇数内核
            elif "scale" in name:
                value = self.rng.randint(1, 4)
            elif "rate" in name:
                value = self.rng.uniform(0.1, 0.5)
            elif "alpha" in name:
                value = self.rng.uniform(0.01, 0.3)
            elif "stride" in name:
                value = self.rng.randint(1, 3)
            else:
                value = self.rng.randint(1, 64)  # 默认范围

        self.generated_params[name] = value
        return value


class FixedParameterProvider(ParameterProvider):
    """固定参数提供器，用于非随机模式"""

    def __init__(self, param_config):
        self.param_config = param_config or {}

    def get(self, name, default=None, min_val=None, max_val=None):
        if name in self.param_config:
            return self.param_config[name]
        return default


class ParameterConfig:
    """算子参数配置类，处理参数依赖和范围限制"""

    def __init__(self, provider):
        self.provider = provider
        self.params = {}

    def get_common_params(self):
        """获取通用参数"""
        self.params["batch"] = 1
        self.params["dtype"] = "float32"
        return self

    def get_conv_params(self):
        """获取卷积相关参数"""
        self.params["in_channels"] = self.provider.get("in_channels", min_val=1, max_val=128)
        self.params["out_channels"] = self.provider.get("out_channels", min_val=1, max_val=128)

        # 内核大小
        self.params["kernel_size"] = self.provider.get("kernel_size", min_val=3, max_val=7)
        if self.params["kernel_size"] % 2 == 0:  # 确保奇数内核
            self.params["kernel_size"] += 1

        # 计算合理的步长
        max_stride = min(3, self.params["kernel_size"] - 1)
        self.params["stride"] = self.provider.get("stride", min_val=1, max_val=max_stride)

        # 自动计算填充
        self.params["padding"] = self.params["kernel_size"] // 2

        # 输入尺寸
        min_input = max(16, self.params["kernel_size"] + 2)
        self.params["input_height"] = self.provider.get("input_height", min_val=min_input, max_val=128)
        self.params["input_width"] = self.provider.get("input_width",
                                                       default=self.params["input_height"],
                                                       min_val=min_input,
                                                       max_val=128)
        return self

    def get_pool_params(self):
        """获取池化相关参数"""
        self.params["pool_channels"] = self.provider.get("pool_channels", min_val=1, max_val=128)

        # 池化大小
        self.params["pool_size"] = self.provider.get("pool_size", min_val=2, max_val=5)

        # 计算合理的步长
        max_stride = min(3, self.params["pool_size"])
        self.params["stride"] = self.provider.get("stride", min_val=1, max_val=max_stride)

        # 自动计算填充
        self.params["padding"] = self.params["pool_size"] // 2

        # 输入尺寸
        min_input = max(16, self.params["pool_size"] + 2)
        self.params["input_height"] = self.provider.get("input_height", min_val=min_input, max_val=64)
        self.params["input_width"] = self.provider.get("input_width",
                                                       default=self.params["input_height"],
                                                       min_val=min_input,
                                                       max_val=64)
        return self

    def get_dense_params(self):
        """获取全连接层相关参数"""
        self.params["in_features"] = self.provider.get("in_features", min_val=64, max_val=1024)
        self.params["out_features"] = self.provider.get("out_features", min_val=64, max_val=1024)
        return self

    def get_activation_params(self):
        """获取激活函数相关参数"""
        self.params["channels"] = self.provider.get("activation_channels", min_val=1, max_val=128)

        # 激活函数的输入尺寸
        self.params["input_height"] = self.provider.get("activation_height", min_val=16, max_val=64)
        self.params["input_width"] = self.provider.get("activation_width",
                                                       default=self.params["input_height"],
                                                       min_val=16,
                                                       max_val=64)
        return self

    def get_upsample_params(self):
        """获取上采样相关参数"""
        self.params["channels"] = self.provider.get("upsample_channels", min_val=1, max_val=128)
        self.params["scale"] = self.provider.get("scale", min_val=1, max_val=4)

        # 输入尺寸
        self.params["input_height"] = self.provider.get("upsample_height", min_val=16, max_val=32)
        self.params["input_width"] = self.provider.get("upsample_width",
                                                       default=self.params["input_height"],
                                                       min_val=16,
                                                       max_val=32)
        return self

    def get_matmul_params(self):
        """获取矩阵乘法相关参数"""
        self.params["M"] = self.provider.get("matmul_M", min_val=16, max_val=64)
        self.params["N"] = self.provider.get("matmul_N", min_val=16, max_val=64)
        self.params["K"] = self.provider.get("matmul_K", min_val=16, max_val=64)
        return self

    def get_params(self):
        """返回所有参数"""
        return self.params


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


# 以下是各个算子的导出函数，使用相同的参数生成
def export_conv2d(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_conv_params().get_params()

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

    export_op(conv, "conv2d", [data, weight], params)


def export_max_pool2d(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_pool_params().get_params()

    data = relay.var("data",
                     shape=(params["batch"], params["pool_channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    pool = relay.nn.max_pool2d(
        data,
        pool_size=(params["pool_size"], params["pool_size"]),
        strides=(params["stride"], params["stride"]),
        padding=(params["padding"], params["padding"])
    )

    export_op(pool, "max_pool2d", [data], params)


def export_relu(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_activation_params().get_params()

    data = relay.var("data",
                     shape=(params["batch"], params["channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    relu = relay.nn.relu(data)
    export_op(relu, "relu", [data], params)


def export_batch_norm(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_activation_params().get_params()

    data = relay.var("data",
                     shape=(params["batch"], params["channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    gamma = relay.var("gamma", shape=(params["channels"],), dtype=params["dtype"])
    beta = relay.var("beta", shape=(params["channels"],), dtype=params["dtype"])
    moving_mean = relay.var("moving_mean", shape=(params["channels"],), dtype=params["dtype"])
    moving_var = relay.var("moving_var", shape=(params["channels"],), dtype=params["dtype"])

    bn = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var)[0]
    export_op(bn, "batch_norm", [data, gamma, beta, moving_mean, moving_var], params)


def export_dense(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_dense_params().get_params()

    data = relay.var("data", shape=(params["batch"], params["in_features"]), dtype=params["dtype"])
    weight = relay.var("weight", shape=(params["out_features"], params["in_features"]), dtype=params["dtype"])

    dense = relay.nn.dense(data, weight, units=params["out_features"])
    export_op(dense, "dense", [data, weight], params)


def export_softmax(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_params()
    params["features"] = param_cfg.provider.get("softmax_features", min_val=32, max_val=256)

    data = relay.var("data", shape=(params["batch"], params["features"]), dtype=params["dtype"])
    softmax = relay.nn.softmax(data)
    export_op(softmax, "softmax", [data], params)


def export_conv2d_transpose(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_conv_params().get_params()

    # 转置卷积特殊处理
    data = relay.var("data",
                     shape=(params["batch"], params["in_channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    weight = relay.var("weight",
                       shape=(params["in_channels"], params["out_channels"],
                              params["kernel_size"], params["kernel_size"]),
                       dtype=params["dtype"])

    conv_t = relay.nn.conv2d_transpose(
        data,
        weight,
        channels=params["out_channels"],
        kernel_size=(params["kernel_size"], params["kernel_size"]),
        strides=(params["stride"], params["stride"]),
        padding=(params["padding"], params["padding"])
    )

    export_op(conv_t, "conv2d_transpose", [data, weight], params)


def export_avg_pool2d(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_pool_params().get_params()

    data = relay.var("data",
                     shape=(params["batch"], params["pool_channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    pool = relay.nn.avg_pool2d(
        data,
        pool_size=(params["pool_size"], params["pool_size"]),
        strides=(params["stride"], params["stride"]),
        padding=(params["padding"], params["padding"])
    )

    export_op(pool, "avg_pool2d", [data], params)


def export_bias_add(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_activation_params().get_params()

    data = relay.var("data",
                     shape=(params["batch"], params["channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    bias = relay.var("bias", shape=(params["channels"],), dtype=params["dtype"])
    bias_add = relay.nn.bias_add(data, bias)
    export_op(bias_add, "bias_add", [data, bias], params)


def export_upsampling(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_upsample_params().get_params()

    data = relay.var("data",
                     shape=(params["batch"], params["channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    up = relay.nn.upsampling(data, scale_h=params["scale"], scale_w=params["scale"])
    export_op(up, "upsampling", [data], params)


def export_dropout(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_params()
    params["features"] = param_cfg.provider.get("dropout_features", min_val=128, max_val=1024)
    params["rate"] = param_cfg.provider.get("rate", min_val=0.1, max_val=0.5)

    data = relay.var("data", shape=(params["batch"], params["features"]), dtype=params["dtype"])
    dropout = relay.nn.dropout(data, rate=params["rate"])
    export_op(dropout, "dropout", [data], params)


def export_matmul(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_matmul_params().get_params()

    a = relay.var("a", shape=(params["M"], params["K"]), dtype=params["dtype"])
    b = relay.var("b", shape=(params["K"], params["N"]), dtype=params["dtype"])
    matmul = relay.nn.matmul(a, b)
    export_op(matmul, "matmul", [a, b], params)


def export_adaptive_avg_pool2d(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_activation_params().get_params()

    data = relay.var("data",
                     shape=(params["batch"], params["channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    pool = relay.nn.adaptive_avg_pool2d(data, (1, 1))
    export_op(pool, "adaptive_avg_pool2d", [data], params)


def export_leaky_relu(param_provider=None):
    param_cfg = ParameterConfig(param_provider or RandomParameterProvider())
    params = param_cfg.get_common_params().get_activation_params().get_params()
    params["alpha"] = param_cfg.provider.get("alpha", min_val=0.01, max_val=0.3)

    data = relay.var("data",
                     shape=(params["batch"], params["channels"], params["input_height"], params["input_width"]),
                     dtype=params["dtype"])

    leaky_relu = relay.nn.leaky_relu(data, alpha=params["alpha"])
    export_op(leaky_relu, "leaky_relu", [data], params)


# 主函数
if __name__ == "__main__":
    # 生成1000套算子
    for i in range(1000):
        print(f"生成第 {i + 1}/1000 套算子...")

        # 为当前算子集创建目录
        set_dir = os.path.join(cur_dir, f"set_{i}")
        global native_dir, device_dir
        native_dir = os.path.join(set_dir, "native")
        device_dir = os.path.join(set_dir, "device")
        os.makedirs(native_dir, exist_ok=True)
        os.makedirs(device_dir, exist_ok=True)

        print(f"导出到: {native_dir} 和 {device_dir}")

        # 使用相同的随机种子确保不同集合有不同的随机参数
        rng_seed = 42 + i
        provider = RandomParameterProvider(rng_seed)

        # 导出所有算子
        export_conv2d(provider)
        # export_max_pool2d(provider)
        export_relu(provider)
        export_batch_norm(provider)
        export_dense(provider)
        export_softmax(provider)
        # export_conv2d_transpose(provider)
        export_avg_pool2d(provider)
        export_bias_add(provider)
        # export_upsampling(provider)
        # export_dropout(provider)
        export_matmul(provider)
        # export_adaptive_avg_pool2d(provider)
        # export_leaky_relu(provider)

    print("所有算子导出成功!")
    print(f"算子库保存在: {cur_dir}/set_*")