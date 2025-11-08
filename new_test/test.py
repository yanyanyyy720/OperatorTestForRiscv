# simplified_exporter.py
import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple
import numpy as np

from network_util import execute_command

# 导入您现有的模块
try:
    from define import (
        TFLiteOperatorGenerator,
        build_op,
        convert_to_tflite,
        generate_test_inputs
    )
except ImportError:
    # 如果导入失败，定义简单版本用于测试
    import tensorflow as tf


    class TFLiteOperatorGenerator:
        def __init__(self):
            self.supported_ops = ['ADD', 'SUB', 'MUL', 'DIV', 'CONV_2D', 'TANH', 'RELU']
            self.size_configs = {
                'small': {'shape_range': [(16, 16), (32, 32)]},
                'medium': {'shape_range': [(32, 32), (64, 64)]},
                'large': {'shape_range': [(64, 64), (128, 128)]},
                'xlarge': {'shape_range': [(128, 128), (256, 256)]},
                'xxlarge': {'shape_range': [(256, 256), (512, 512)]}
            }


    def build_op(op_name, size):
        # 创建一个简单的模型用于测试
        if op_name == "ADD":
            input1 = tf.keras.layers.Input(shape=(10,), name='input1')
            input2 = tf.keras.layers.Input(shape=(10,), name='input2')
            output = tf.keras.layers.Add()([input1, input2])
            return tf.keras.Model(inputs=[input1, input2], outputs=output)
        else:
            input_layer = tf.keras.layers.Input(shape=(10,))
            output_layer = tf.keras.layers.Dense(1)(input_layer)
            return tf.keras.Model(inputs=input_layer, outputs=output_layer)


    def convert_to_tflite(model, op_name):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        return converter.convert()


    def generate_test_inputs(model, input_types):
        if len(model.inputs) == 2:
            return [np.random.randn(1, 10).astype(np.float32),
                    np.random.randn(1, 10).astype(np.float32)]
        return [np.random.randn(1, 10).astype(np.float32)]

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

# 扩展的大小配置
SIZE_CONFIGS = {
    'small': {'shape_range': [(16, 16), (32, 32)]},
    'medium': {'shape_range': [(32, 32), (64, 64)]},
    'large': {'shape_range': [(64, 64), (128, 128)]},
    'xlarge': {'shape_range': [(128, 128), (256, 256)]},
    'xxlarge': {'shape_range': [(256, 256), (512, 512)]}
}


def export_op_with_platforms(op_name: str, size: str, run_id: int, base_output_dir: str = "./exported_models") -> Dict:
    """
    导出单个算子到TFLite和两个TVM平台(rv和rvv)
    """
    # 创建运行目录
    run_dir = os.path.join(base_output_dir, f"run_{run_id}")
    tflite_dir = os.path.join(run_dir, "tflite", size)
    tvm_rv_dir = os.path.join(run_dir, "tvm", "rv", size)
    tvm_rvv_dir = os.path.join(run_dir, "tvm", "rvv", size)

    # 创建输出目录
    os.makedirs(tflite_dir, exist_ok=True)
    os.makedirs(tvm_rv_dir, exist_ok=True)
    os.makedirs(tvm_rvv_dir, exist_ok=True)

    result = {
        'op_name': op_name,
        'size': size,
        'run_id': run_id,
        'tflite_path': None,
        'tvm_paths': {},
        'status': 'failed'
    }

    try:
        # 1. 构建Keras模型
        print(f"🔧 运行{run_id}: 构建 {op_name} ({size}) 模型...")
        model = build_op(op_name, size)

        # 2. 导出为TFLite
        print(f"  导出为TFLite格式...")
        tflite_model = convert_to_tflite(model, op_name)

        if tflite_model is not None:
            tflite_path = os.path.join(tflite_dir, f"{op_name}_{size}.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            result['tflite_path'] = tflite_path
            print(f"    ✓ TFLite模型已保存: {tflite_path}")
        else:
            print("    ⚠️ TFLite转换返回None")
            result['error'] = "TFLite转换返回None"
            return result

        # 3. 导出为TVM（如果可用）
        if TVM_AVAILABLE:
            result['tvm_paths'] = {}

            # 为每个平台导出TVM模型
            for platform_name, platform_config in TARGET_PLATFORMS.items():
                print(f"  导出为TVM {platform_name}格式...")

                # 选择对应的输出目录
                if platform_name == "rv":
                    platform_dir = tvm_rv_dir
                else:  # rvv
                    platform_dir = tvm_rvv_dir

                tvm_paths = convert_tflite_to_tvm_simple(
                    tflite_model, op_name, size, platform_dir, platform_config
                )

                if tvm_paths:
                    result['tvm_paths'][platform_name] = tvm_paths
                    print(f"    ✓ TVM {platform_name}模型导出成功")
                else:
                    print(f"    ⚠️ TVM {platform_name}模型导出失败")
        else:
            print("    ⚠️ TVM不可用，跳过TVM导出")

        result['status'] = 'success'

    except Exception as e:
        print(f"❌ 导出 {op_name} 失败: {e}")
        result['error'] = str(e)
        import traceback
        traceback.print_exc()

    return result


def convert_tflite_to_tvm_simple(tflite_model: bytes, op_name: str, size: str,
                                 output_dir: str, platform_config: Dict) -> Dict:
    """
    直接将TFLite模型转换为TVM格式
    """
    if not TVM_AVAILABLE:
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
        base_name = f"{op_name}_{size}"
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

        # 保存元数据
        metadata = {
            'op_name': op_name,
            'size': size,
            'platform': platform_config["dir"],
            'conversion_method': 'tflite_to_tvm',
            'export_time': str(np.datetime64('now'))
        }

        metadata_path = f"{base_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            'so_path': so_path,
            'graph_json_path': graph_json_path,
            'params_path': params_path,
            'metadata_path': metadata_path
        }

    except Exception as e:
        print(f"    ❌ TFLite到TVM转换失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_tvm(tvm_path: str, output_dir: str) -> Dict:
    """
    执行TVM模型验证
    """
    try:
        # 构建输出路径
        output_path = os.path.join(output_dir, "tvm_output")

        # 执行验证命令
        cmd = f"/home/yan/workdir/cpp/validator-riscv /home/yan/workdir/new_test/{tvm_path} /home/yan/workdir/new_test/{output_path} --no-save-outputs"
        result = execute_command(cmd)

        print(f"执行结果: {result}")
        return {'success': True, 'result': result}

    except Exception as e:
        print(f"❌ 执行TVM验证失败: {e}")
        return {'success': False, 'error': str(e)}


def run_op_for_all_platforms(op_name: str, size: str, run_id: int, base_output_dir: str = "./exported_models") -> Dict:
    """
    为单个算子运行所有平台的验证
    """
    # 导出模型
    export_result = export_op_with_platforms(op_name, size, run_id, base_output_dir)

    if export_result['status'] != 'success':
        return export_result

    # 运行验证
    validation_results = {}

    # 为每个平台运行验证
    for platform_name, tvm_paths in export_result.get('tvm_paths', {}).items():
        if tvm_paths and 'so_path' in tvm_paths:
            print(f"🧪 运行 {platform_name} 平台验证...")

            # 创建平台特定的输出目录
            platform_output_dir = os.path.join(
                base_output_dir,
                f"run_{run_id}",
                "output",
                platform_name,
                size
            )
            os.makedirs(platform_output_dir, exist_ok=True)

            # 运行验证
            validation_result = run_tvm(tvm_paths['so_path'], platform_output_dir)
            validation_results[platform_name] = validation_result

            if validation_result.get('success'):
                print(f"   ✅ {platform_name} 验证成功")
            else:
                print(f"   ❌ {platform_name} 验证失败: {validation_result.get('error', '未知错误')}")

    export_result['validation_results'] = validation_results
    return export_result


def run_all_ops_multiple_times(num_runs: int = 10,
                               sizes: List[str] = ["small", "medium", "large"],
                               base_output_dir: str = "./exported_models",
                               ops_to_run: List[str] = None) -> Dict:
    """
    多次运行所有算子的导出和验证
    """
    # 获取支持的算子
    generator = TFLiteOperatorGenerator()

    if ops_to_run is None:
        ops_to_run = generator.supported_ops

    all_results = []

    for run_id in range(1, num_runs + 1):
        print(f"\n🔄 开始第 {run_id}/{num_runs} 次运行")
        run_results = []

        for op_name in ops_to_run:
            for size in sizes:
                print(f"\n🎯 处理算子: {op_name} (大小: {size})")

                # 运行单个算子的导出和验证
                result = run_op_for_all_platforms(op_name, size, run_id, base_output_dir)
                result['size'] = size
                run_results.append(result)

                # 显示简要结果
                if result['status'] == 'success':
                    print(f"   ✅ {op_name} ({size}) 导出成功")
                    # 显示验证结果
                    for platform, val_result in result.get('validation_results', {}).items():
                        status = "成功" if val_result.get('success') else "失败"
                        print(f"      {platform}: {status}")
                else:
                    print(f"   ❌ {op_name} ({size}) 导出失败: {result.get('error', '未知错误')}")

        # 保存本次运行的结果
        run_summary = {
            'run_id': run_id,
            'total_ops': len(ops_to_run) * len(sizes),
            'success_count': sum(1 for r in run_results if r['status'] == 'success'),
            'failed_count': sum(1 for r in run_results if r['status'] != 'success'),
            'results': run_results
        }

        # 保存运行摘要
        run_summary_path = os.path.join(base_output_dir, f"run_{run_id}", "run_summary.json")
        with open(run_summary_path, 'w') as f:
            json.dump(run_summary, f, indent=2)

        all_results.append(run_summary)

        print(f"\n📊 第 {run_id} 次运行完成:")
        print(f"   成功: {run_summary['success_count']}/{run_summary['total_ops']}")
        print(f"   失败: {run_summary['failed_count']}/{run_summary['total_ops']}")

    # 生成总摘要
    total_summary = {
        'total_runs': num_runs,
        'total_ops_per_run': len(ops_to_run) * len(sizes),
        'sizes': sizes,
        'runs': all_results
    }

    # 保存总摘要
    total_summary_path = os.path.join(base_output_dir, "total_summary.json")
    with open(total_summary_path, 'w') as f:
        json.dump(total_summary, f, indent=2)

    # 计算统计信息
    total_success = sum(run['success_count'] for run in all_results)
    total_failures = sum(run['failed_count'] for run in all_results)
    total_operations = num_runs * len(ops_to_run) * len(sizes)

    print(f"\n🎉 所有运行完成!")
    print(f"   总运行次数: {num_runs}")
    print(f"   每次运行的算子数: {len(ops_to_run)}")
    print(f"   测试的大小: {', '.join(sizes)}")
    print(f"   总操作数: {total_operations}")
    print(f"   总成功: {total_success}")
    print(f"   总失败: {total_failures}")
    if total_operations > 0:
        print(f"   成功率: {total_success / total_operations * 100:.2f}%")

    return total_summary


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="多次运行算子导出和验证")

    parser.add_argument("--runs", type=int, default=100,
                        help="运行次数")
    parser.add_argument("--sizes", type=str, nargs="+",
                        choices=['small', 'medium', 'large', 'xlarge', 'xxlarge'],
                        default=['small', 'medium', 'large'],
                        help="模型大小列表")
    parser.add_argument("--output-dir", type=str, default="./exported_models2",
                        help="输出目录")
    parser.add_argument("--ops", type=str, nargs="+",
                        help="指定要运行的算子列表")
    parser.add_argument("--list-ops", action="store_true",
                        help="列出所有支持的算子")
    parser.add_argument("--single-op", type=str,
                        help="运行单个算子")
    parser.add_argument("--single-run", type=int,
                        help="运行单次测试")

    args = parser.parse_args()

    # 列出支持的算子
    if args.list_ops:
        generator = TFLiteOperatorGenerator()
        print("📋 支持的算子列表:")
        for i, op in enumerate(generator.supported_ops, 1):
            print(f"  {i:2d}. {op}")
        return

    # 确定要运行的算子
    generator = TFLiteOperatorGenerator()
    if args.ops:
        ops_to_run = [op for op in args.ops if op in generator.supported_ops]
        if len(ops_to_run) != len(args.ops):
            print(f"⚠️ 警告: 部分算子不被支持，已过滤")
    elif args.single_op:
        if args.single_op in generator.supported_ops:
            ops_to_run = [args.single_op]
        else:
            print(f"❌ 算子 {args.single_op} 不被支持")
            return
    else:
        ops_to_run = generator.supported_ops

    print(f"🎯 将运行以下算子: {', '.join(ops_to_run)}")
    print(f"📏 测试大小: {', '.join(args.sizes)}")

    # 运行测试
    if args.single_run:
        # 单次运行
        print(f"\n🔄 开始单次运行 (ID: {args.single_run})")
        result = run_all_ops_multiple_times(
            num_runs=1,
            sizes=args.sizes,
            base_output_dir=args.output_dir,
            ops_to_run=ops_to_run
        )
    else:
        # 多次运行
        result = run_all_ops_multiple_times(
            num_runs=args.runs,
            sizes=args.sizes,
            base_output_dir=args.output_dir,
            ops_to_run=ops_to_run
        )


if __name__ == "__main__":
    main()