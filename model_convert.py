import os
import tvm
from tvm import relay
import tensorflow as tf
import tflite


def export_model(model, op_name, tflite_output_dir, native_output_dir, device_output_dir):
    """
    统一导出模型到TFLite和TVM格式
    返回三元组: (tflite_path, native_path, device_path)
    """
    # 导出TFLite模型
    tflite_path = convert_to_tflite(model, op_name, tflite_output_dir)
    if not tflite_path:
        return None, None, None

    # 读取TFLite模型内容
    with open(tflite_path, "rb") as f:
        tflite_model = f.read()

    # 导出TVM模型
    tvm_paths = convert_tflite_to_tvm(tflite_model, op_name, native_output_dir, device_output_dir)
    if not tvm_paths:
        return tflite_path, None, None

    return tflite_path, tvm_paths.get("native"), tvm_paths.get("device")


def export_tvm_module(op_name, mod, params, native_output_dir, device_output_dir,
                      native_target="llvm",
                      device_target="llvm -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c",
                      native_cc="/usr/bin/g++", device_cc="/usr/bin/riscv64-linux-gnu-g++"):
    """导出TVM模块到指定路径，分离native和device版本"""
    paths = {}

    # 确保输出目录存在
    os.makedirs(native_output_dir, exist_ok=True)
    os.makedirs(device_output_dir, exist_ok=True)

    # 执行类型推断
    mod = relay.transform.InferType()(mod)

    # 编译并导出Native版本
    target_native = tvm.target.Target(native_target)
    with tvm.transform.PassContext(opt_level=3):
        lib_native = relay.build(mod, target=target_native, params=params)

    so_path_native = os.path.join(native_output_dir, f"{op_name}.so")
    lib_native.export_library(so_path_native, cc=native_cc)
    paths["native"] = so_path_native

    # 编译并导出Device版本
    target_device = tvm.target.Target(device_target)
    with tvm.transform.PassContext(opt_level=3):
        lib_device = relay.build(mod, target=target_device, params=params)

    so_path_device = os.path.join(device_output_dir, f"{op_name}.so")
    lib_device.export_library(so_path_device, cc=device_cc)
    paths["device"] = so_path_device

    print(f"✅ 成功导出TVM模块: {op_name}")
    print(f"  Native: {so_path_native}")
    print(f"  Device: {so_path_device}")

    return paths


def convert_tflite_to_tvm(tflite_model, op_name, native_output_dir, device_output_dir):
    """将TFLite模型转换为TVM格式并导出到指定路径"""
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
        return export_tvm_module(op_name, mod, params, native_output_dir, device_output_dir)
    except Exception as e:
        print(f"❌ 转换为TVM失败: {str(e)}")
        return {}


def convert_to_tflite(model, op_name, tflite_output_dir):
    """将TensorFlow模型转换为TFLite格式并保存到指定路径"""
    try:
        # 确保输出目录存在
        os.makedirs(tflite_output_dir, exist_ok=True)

        # 转换为具体函数
        concrete_func = model.__call__.get_concrete_function()

        # 转换为TFLite模型
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()

        # 保存TFLite模型
        tflite_path = os.path.join(tflite_output_dir, f"{op_name}.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"✅ 生成TFLite模型: {tflite_path}")

        return tflite_path
    except Exception as e:
        print(f"❌ 转换为TFLite失败: {str(e)}")
        return None