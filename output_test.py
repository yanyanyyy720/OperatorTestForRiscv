import os
import glob
import shutil
import numpy as np
import struct
from pathlib import Path
from datetime import datetime
from network_util import execute_command


def execute_validator(so_file_path, input_path=None, output_dir=None, rounds=10,
                      save_outputs=True, save_random_inputs=True):
    """
    执行验证器程序

    Args:
        so_file_path: .so文件路径
        input_path: 输入数据路径（文件或文件夹）
        output_dir: 输出目录
        rounds: 运行轮数
        save_outputs: 是否保存输出
        save_random_inputs: 是否保存随机输入
    """
    # 构建命令
    cmd_parts = ["/home/yan/workdir/cpp/validator-riscv"]

    # 添加输入路径（如果有）
    if input_path and os.path.exists(input_path):
        cmd_parts.extend(["--input", "/home/yan/workdir/" + input_path])

    # 添加输出目录
    if output_dir:
        cmd_parts.extend(["--output", "/home/yan/workdir/" + output_dir])

    # 添加运行轮数
    cmd_parts.extend(["--rounds", str(rounds)])

    # 根据选项添加参数
    if not save_outputs:
        cmd_parts.append("--no-save-outputs")

    if save_random_inputs:
        cmd_parts.append("--save-random-inputs")

    # 添加.so文件路径
    cmd_parts.append("/home/yan/workdir/" + so_file_path)

    # 执行命令
    cmd = " ".join(cmd_parts)
    print(f"执行命令: {cmd}")
    result = execute_command(cmd)
    print(result)

    return result


def find_matching_so_files(op_lib_dir):
    """
    查找同set_i下同名的rv和rvv目录下的so文件

    Args:
        op_lib_dir: op_lib目录路径

    Returns:
        list: 包含匹配信息的列表，每个元素为 (set_dir, so_name, rv_so_path, rvv_so_path)
    """
    matches = []

    # 查找所有的set_i目录
    set_pattern = os.path.join(op_lib_dir, "set_*")
    set_dirs = glob.glob(set_pattern)
    set_dirs.sort()
    set_dirs = [set_dirs[2],set_dirs[6],set_dirs[7]]
    for set_dir in set_dirs:
        set_name = os.path.basename(set_dir)

        # 获取rv和rvv目录下的所有so文件
        rv_so_dir = os.path.join(set_dir, "rv")
        rvv_so_dir = os.path.join(set_dir, "rvv")

        if not os.path.exists(rv_so_dir) or not os.path.exists(rvv_so_dir):
            continue

        # 获取rv目录下的所有so文件名（不含路径）
        rv_so_files = {}
        for so_file in glob.glob(os.path.join(rv_so_dir, "*.so")):
            so_name = os.path.basename(so_file)
            rv_so_files[so_name] = so_file

        # 获取rvv目录下的所有so文件名（不含路径）
        rvv_so_files = {}
        for so_file in glob.glob(os.path.join(rvv_so_dir, "*.so")):
            so_name = os.path.basename(so_file)
            rvv_so_files[so_name] = so_file

        # 查找同名的so文件
        common_names = set(rv_so_files.keys()) & set(rvv_so_files.keys())

        for so_name in common_names:
            matches.append((set_dir, so_name, rv_so_files[so_name], rvv_so_files[so_name]))

    return matches


def find_input_files(input_dir, round_num=None):
    """
    查找输入文件，支持新的目录结构

    Args:
        input_dir: 输入目录
        round_num: 指定轮次（可选）

    Returns:
        list: 输入文件路径列表
    """
    input_files = []

    if round_num is not None:
        # 查找指定轮次的输入文件
        round_dir = os.path.join(input_dir, f"round_{round_num}")
        if os.path.exists(round_dir):
            pattern = os.path.join(round_dir, "input_*.bin")
            input_files.extend(glob.glob(pattern))
    else:
        # 查找所有轮次的输入文件
        round_dirs = glob.glob(os.path.join(input_dir, "round_*"))
        for round_dir in round_dirs:
            pattern = os.path.join(round_dir, "input_*.bin")
            input_files.extend(glob.glob(pattern))

    # 按文件名排序
    input_files.sort()
    return input_files


def find_output_files(output_dir, round_num=None):
    """
    查找输出文件，支持新的目录结构

    Args:
        output_dir: 输出目录
        round_num: 指定轮次（可选）

    Returns:
        list: 输出文件路径列表
    """
    output_files = []

    if round_num is not None:
        # 查找指定轮次的输出文件
        round_dir = os.path.join(output_dir, f"round_{round_num}")
        if os.path.exists(round_dir):
            pattern = os.path.join(round_dir, "output_*.bin")
            output_files.extend(glob.glob(pattern))
    else:
        # 查找所有轮次的输出文件
        round_dirs = glob.glob(os.path.join(output_dir, "round_*"))
        for round_dir in round_dirs:
            pattern = os.path.join(round_dir, "output_*.bin")
            output_files.extend(glob.glob(pattern))

    # 按文件名排序
    output_files.sort()
    return output_files


def read_binary_tensor(file_path, dtype='float32'):
    """
    从二进制文件读取张量数据

    Args:
        file_path: 二进制文件路径
        dtype: 数据类型，默认float32

    Returns:
        numpy.ndarray: 张量数据
    """
    try:
        # 读取二进制数据
        with open(file_path, 'rb') as f:
            data = f.read()

        # 根据数据类型解析
        if dtype == 'float32':
            # 假设数据是float32类型
            array = np.frombuffer(data, dtype=np.float32)
        elif dtype == 'float16':
            array = np.frombuffer(data, dtype=np.float16)
        elif dtype == 'int32':
            array = np.frombuffer(data, dtype=np.int32)
        elif dtype == 'int8':
            array = np.frombuffer(data, dtype=np.int8)
        else:
            # 默认使用float32
            array = np.frombuffer(data, dtype=np.float32)

        return array
    except Exception as e:
        print(f"读取二进制文件失败 {file_path}: {e}")
        return None


def calculate_tensor_statistics(tensor):
    """
    计算张量的统计信息

    Args:
        tensor: numpy数组

    Returns:
        dict: 包含统计信息的字典
    """
    if tensor is None or tensor.size == 0:
        return {
            'min': 0, 'max': 0, 'mean': 0, 'std': 0,
            'abs_max': 0, 'abs_mean': 0, 'size': 0
        }

    abs_tensor = np.abs(tensor)
    return {
        'min': np.min(tensor),
        'max': np.max(tensor),
        'mean': np.mean(tensor),
        'std': np.std(tensor),
        'abs_max': np.max(abs_tensor),
        'abs_mean': np.mean(abs_tensor),
        'size': tensor.size
    }


def compare_tensors(rv_tensor, rvv_tensor, tolerance=1e-6):
    """
    比较两个张量的差异

    Args:
        rv_tensor: RV张量
        rvv_tensor: RVV张量
        tolerance: 容差阈值

    Returns:
        dict: 包含比较结果的字典
    """
    if rv_tensor is None or rvv_tensor is None:
        return {
            'absolute_error': 0,
            'relative_error': 0,
            'max_absolute_error': 0,
            'max_relative_error': 0,
            'mean_absolute_error': 0,
            'mean_relative_error': 0,
            'mse': 0,
            'rmse': 0,
            'correlation': 0,
            'valid_comparison': False,
            'tensors_equal': False
        }

    # 确保张量形状一致
    if rv_tensor.shape != rvv_tensor.shape:
        min_size = min(rv_tensor.size, rvv_tensor.size)
        rv_flat = rv_tensor.flatten()[:min_size]
        rvv_flat = rvv_tensor.flatten()[:min_size]
    else:
        rv_flat = rv_tensor.flatten()
        rvv_flat = rvv_tensor.flatten()

    # 检查张量是否完全相等（考虑浮点误差）
    if rv_flat.shape == rvv_flat.shape:
        if np.allclose(rv_flat, rvv_flat, rtol=tolerance, atol=tolerance):
            return {
                'absolute_error': 0,
                'relative_error': 0,
                'max_absolute_error': 0,
                'max_relative_error': 0,
                'mean_absolute_error': 0,
                'mean_relative_error': 0,
                'mse': 0,
                'rmse': 0,
                'correlation': 1.0,
                'valid_comparison': True,
                'tensors_equal': True
            }

    # 计算各种误差指标
    diff = rv_flat - rvv_flat
    abs_diff = np.abs(diff)

    # 避免除以零
    rv_abs = np.abs(rv_flat)
    rv_abs[rv_abs == 0] = 1e-10  # 避免除以零

    relative_diff = abs_diff / rv_abs

    # 计算各种误差指标
    result = {
        'absolute_error': np.sum(abs_diff),
        'relative_error': np.sum(relative_diff),
        'max_absolute_error': np.max(abs_diff) if abs_diff.size > 0 else 0,
        'max_relative_error': np.max(relative_diff) if relative_diff.size > 0 else 0,
        'mean_absolute_error': np.mean(abs_diff) if abs_diff.size > 0 else 0,
        'mean_relative_error': np.mean(relative_diff) if relative_diff.size > 0 else 0,
        'mse': np.mean(diff ** 2) if diff.size > 0 else 0,
        'rmse': np.sqrt(np.mean(diff ** 2)) if diff.size > 0 else 0,
        'correlation': np.corrcoef(rv_flat, rvv_flat)[0, 1] if len(rv_flat) > 1 else 1,
        'valid_comparison': True,
        'tensors_equal': False
    }

    return result


def check_outputs_exist_and_equal(rv_output_dir, rvv_output_dir, rounds, tolerance=1e-6):
    """
    检查RV和RVV的输出文件是否存在且相等，考虑完整的目录结构

    Args:
        rv_output_dir: RV输出目录（应该是包含output_data子目录的路径）
        rvv_output_dir: RVV输出目录（应该是包含output_data子目录的路径）
        rounds: 期望的轮数
        tolerance: 容差阈值

    Returns:
        tuple: (outputs_exist, outputs_equal, details)
    """
    # 检查输出目录是否存在
    if not os.path.exists(rv_output_dir) or not os.path.exists(rvv_output_dir):
        return False, False, "输出目录不存在"

    # 检查每个轮次的输出文件
    all_equal = True
    comparison_details = {}
    missing_rounds = []

    for round_num in range(rounds):
        round_dir_rv = os.path.join(rv_output_dir, f"round_{round_num}")
        round_dir_rvv = os.path.join(rvv_output_dir, f"round_{round_num}")

        # 检查轮次目录是否存在
        if not os.path.exists(round_dir_rv) or not os.path.exists(round_dir_rvv):
            missing_rounds.append(round_num)
            all_equal = False
            continue

        # 查找该轮次的输出文件
        rv_output_files = glob.glob(os.path.join(round_dir_rv, "output_*.bin"))
        rvv_output_files = glob.glob(os.path.join(round_dir_rvv, "output_*.bin"))

        # 检查文件数量
        if len(rv_output_files) == 0 or len(rvv_output_files) == 0:
            missing_rounds.append(round_num)
            all_equal = False
            continue

        # 按文件名排序，确保对应文件匹配
        rv_output_files.sort()
        rvv_output_files.sort()

        # 检查文件数量是否一致
        if len(rv_output_files) != len(rvv_output_files):
            all_equal = False
            comparison_details[
                f"round_{round_num}"] = f"文件数量不一致: RV={len(rv_output_files)}, RVV={len(rvv_output_files)}"
            continue

        # 比较每个输出文件
        round_equal = True
        round_details = {}

        for i, (rv_file, rvv_file) in enumerate(zip(rv_output_files, rvv_output_files)):
            # 检查文件名是否对应
            rv_filename = os.path.basename(rv_file)
            rvv_filename = os.path.basename(rvv_file)

            if rv_filename != rvv_filename:
                round_equal = False
                round_details[f"file_{i}"] = f"文件名不匹配: {rv_filename} vs {rvv_filename}"
                continue

            # 读取张量数据
            rv_tensor = read_binary_tensor(rv_file)
            rvv_tensor = read_binary_tensor(rvv_file)

            # 比较张量
            comparison = compare_tensors(rv_tensor, rvv_tensor, tolerance)
            round_details[rv_filename] = comparison

            if not comparison['tensors_equal']:
                round_equal = False
                all_equal = False

        comparison_details[f"round_{round_num}"] = {
            'round_equal': round_equal,
            'details': round_details
        }

        if not round_equal:
            all_equal = False

    # 检查是否所有轮次都存在
    outputs_exist = len(missing_rounds) == 0

    if missing_rounds:
        details_msg = f"缺失轮次: {missing_rounds}"
    else:
        details_msg = "所有轮次完整"

    return outputs_exist, all_equal, {
        'missing_rounds': missing_rounds,
        'round_comparisons': comparison_details,
        'message': details_msg
    }

def compare_rv_rvv_outputs(rv_output_dir, rvv_output_dir, so_base_name, tolerance=1e-6):
    """
    比较RV和RVV的输出差异，包括数值比较

    Args:
        rv_output_dir: RV输出目录
        rvv_output_dir: RVV输出目录
        so_base_name: 算子基础名称
        tolerance: 容差阈值

    Returns:
        dict: 比较结果
    """
    print(f"比较 {so_base_name} 的RV和RVV输出...")

    # 查找RV的输出文件
    rv_output_files = find_output_files(os.path.join(rv_output_dir, "output_data"))

    # 查找RVV的输出文件
    rvv_output_files = find_output_files(os.path.join(rvv_output_dir, "output_data"))

    # 简单的文件比较
    rv_files = {os.path.basename(f): f for f in rv_output_files}
    rvv_files = {os.path.basename(f): f for f in rvv_output_files}

    common_files = set(rv_files.keys()) & set(rvv_files.keys())

    comparison_results = {}
    all_errors = []

    for filename in common_files:
        rv_file = rv_files[filename]
        rvv_file = rvv_files[filename]

        # 文件大小比较
        rv_size = os.path.getsize(rv_file) if os.path.exists(rv_file) else 0
        rvv_size = os.path.getsize(rvv_file) if os.path.exists(rvv_file) else 0

        # 数值比较
        rv_tensor = read_binary_tensor(rv_file)
        rvv_tensor = read_binary_tensor(rvv_file)

        rv_stats = calculate_tensor_statistics(rv_tensor)
        rvv_stats = calculate_tensor_statistics(rvv_tensor)
        error_metrics = compare_tensors(rv_tensor, rvv_tensor, tolerance)

        comparison_results[filename] = {
            'file_comparison': {
                'rv_size': rv_size,
                'rvv_size': rvv_size,
                'size_diff': abs(rv_size - rvv_size)
            },
            'rv_stats': rv_stats,
            'rvv_stats': rvv_stats,
            'error_metrics': error_metrics
        }

        if error_metrics['valid_comparison']:
            all_errors.append(error_metrics)

        print(f"  {filename}:")
        print(f"    文件大小: RV={rv_size} bytes, RVV={rvv_size} bytes, 差异={abs(rv_size - rvv_size)} bytes")
        if error_metrics['valid_comparison']:
            if error_metrics['tensors_equal']:
                print(f"    ✓ 张量完全相等")
            else:
                print(f"    平均绝对误差: {error_metrics['mean_absolute_error']:.6f}")
                print(f"    平均相对误差: {error_metrics['mean_relative_error']:.6f}")
                print(f"    最大绝对误差: {error_metrics['max_absolute_error']:.6f}")
                print(f"    RMSE: {error_metrics['rmse']:.6f}")
                print(f"    相关性: {error_metrics['correlation']:.6f}")

    # 计算总体误差统计
    if all_errors:
        overall_errors = {
            'mean_absolute_error': np.mean([e['mean_absolute_error'] for e in all_errors]),
            'mean_relative_error': np.mean([e['mean_relative_error'] for e in all_errors]),
            'max_absolute_error': np.max([e['max_absolute_error'] for e in all_errors]),
            'max_relative_error': np.max([e['max_relative_error'] for e in all_errors]),
            'mean_rmse': np.mean([e['rmse'] for e in all_errors]),
            'mean_correlation': np.mean([e['correlation'] for e in all_errors]),
            'total_files': len(all_errors),
            'all_tensors_equal': all(e.get('tensors_equal', False) for e in all_errors)
        }
        comparison_results['overall'] = overall_errors

        print(f"  总体误差统计:")
        if overall_errors['all_tensors_equal']:
            print(f"    ✓ 所有张量完全相等")
        else:
            print(f"    平均绝对误差: {overall_errors['mean_absolute_error']:.6f}")
            print(f"    平均相对误差: {overall_errors['mean_relative_error']:.6f}")
            print(f"    最大绝对误差: {overall_errors['max_absolute_error']:.6f}")
            print(f"    平均RMSE: {overall_errors['mean_rmse']:.6f}")
            print(f"    平均相关性: {overall_errors['mean_correlation']:.6f}")

    return comparison_results


def process_matching_so_files(op_lib_dir, output_base_dir, rounds=10,
                              save_outputs=True, save_random_inputs=True,
                              compare_results=True, resume=True, tolerance=1e-6):
    """
    处理匹配的so文件：先执行rv，再执行rvv，然后比较结果

    Args:
        op_lib_dir: op_lib目录路径
        output_base_dir: 输出基础目录
        rounds: 运行轮数
        save_outputs: 是否保存输出
        save_random_inputs: 是否保存随机输入
        compare_results: 是否比较RV和RVV的结果
        resume: 是否启用端点续行功能
        tolerance: 容差阈值
    """
    # 查找匹配的so文件
    matches = find_matching_so_files(op_lib_dir)

    if not matches:
        print("未找到匹配的rv和rvv so文件")
        return

    print(f"找到 {len(matches)} 对匹配的so文件")

    all_comparisons = {}

    for set_dir, so_name, rv_so_path, rvv_so_path in matches:
        set_name = os.path.basename(set_dir)
        so_base_name = so_name.replace('.so', '')

        print(f"\n{'=' * 60}")
        print(f"处理 {set_name}/{so_name}:")
        print(f"  RV: {rv_so_path}")
        print(f"  RVV: {rvv_so_path}")
        print(f"{'=' * 60}")

        # 创建输出目录结构
        rv_output_dir = os.path.join(output_base_dir, set_name, "rv", so_base_name)
        rvv_output_dir = os.path.join(output_base_dir, set_name, "rvv", so_base_name)

        # 端点续行功能：检查输出是否已存在且相等
        skip_operator = False
        if resume:
            outputs_exist, outputs_equal, details = check_outputs_exist_and_equal(
                os.path.join(rv_output_dir, "output_data"),
                os.path.join(rvv_output_dir, "output_data"),
                rounds,
                tolerance
            )

            if outputs_exist and outputs_equal:
                print(f"✓ 输出文件已存在且相等，跳过算子 {so_name}")
                skip_operator = True
            elif outputs_exist and not outputs_equal:
                print(f"⚠ 输出文件已存在但不相等，重新执行算子 {so_name}")
                print(f"  详情: {details}")
            else:
                print(f"↻ 输出文件不存在或不全，执行算子 {so_name}")
                print(f"  详情: {details}")

        if skip_operator:
            # 如果跳过算子，仍然进行比较以生成报告
            if compare_results:
                comparison = compare_rv_rvv_outputs(rv_output_dir, rvv_output_dir, so_base_name, tolerance)
                all_comparisons[f"{set_name}/{so_base_name}"] = comparison
            continue

        # 第一步：执行RV so文件（生成随机输入并保存输出）
        print("步骤1: 执行RV so文件（生成随机输入并保存输出）...")
        execute_validator(
            so_file_path=rv_so_path,
            output_dir=rv_output_dir,
            rounds=rounds,
            save_outputs=save_outputs,
            save_random_inputs=save_random_inputs
        )

        # 第二步：执行RVV so文件，使用RV的输入作为输入
        print("步骤2: 执行RVV so文件（使用RV的输入作为输入）...")

        # 查找RV生成的输入文件目录
        rv_input_dir = os.path.join(rv_output_dir, "input_data")

        # 如果RV生成了输入文件，则使用这些文件作为RVV的输入
        input_path = None
        if os.path.exists(rv_input_dir) and len(find_input_files(rv_input_dir)) > 0:
            input_path = rv_input_dir
            print(f"使用RV生成的输入文件: {rv_input_dir}")

        execute_validator(
            so_file_path=rvv_so_path,
            input_path=input_path,  # 使用RV的输入目录作为RVV的输入
            output_dir=rvv_output_dir,
            rounds=rounds,
            save_outputs=save_outputs,
            save_random_inputs=False  # RVV不需要保存随机输入，因为它使用RV的输入
        )

        # 第三步：比较RV和RVV的输出
        if compare_results:
            print("步骤3: 比较RV和RVV的输出结果...")
            comparison = compare_rv_rvv_outputs(rv_output_dir, rvv_output_dir, so_base_name, tolerance)
            all_comparisons[f"{set_name}/{so_base_name}"] = comparison

        print(f"完成 {so_name} 的处理")
        print(f"RV输出目录: {rv_output_dir}")
        print(f"RVV输出目录: {rvv_output_dir}")

    # 生成比较报告
    if compare_results and all_comparisons:
        generate_comparison_report(all_comparisons, output_base_dir)

    return all_comparisons


def generate_comparison_report(comparisons, output_dir):
    """
    生成详细的比较报告

    Args:
        comparisons: 比较结果字典
        output_dir: 输出目录
    """
    report_file = os.path.join(output_dir, "rv_rvv_comparison_report.txt")

    with open(report_file, 'w') as f:
        f.write("RV和RVV输出详细比较报告\n")
        f.write("=" * 60 + "\n\n")

        for op_name, comp_data in comparisons.items():
            f.write(f"算子: {op_name}\n")
            f.write("-" * 40 + "\n")

            if not comp_data or 'overall' not in comp_data:
                f.write("  无有效比较数据\n\n")
                continue

            # 写入总体统计
            overall = comp_data.get('overall', {})
            f.write("总体误差统计:\n")
            if overall.get('all_tensors_equal', False):
                f.write("  ✓ 所有张量完全相等\n")
            else:
                f.write(f"  平均绝对误差: {overall.get('mean_absolute_error', 0):.6f}\n")
                f.write(f"  平均相对误差: {overall.get('mean_relative_error', 0):.6f}\n")
                f.write(f"  最大绝对误差: {overall.get('max_absolute_error', 0):.6f}\n")
                f.write(f"  平均RMSE: {overall.get('mean_rmse', 0):.6f}\n")
                f.write(f"  平均相关性: {overall.get('mean_correlation', 0):.6f}\n")
            f.write(f"  比较文件数: {overall.get('total_files', 0)}\n\n")

            # 写入每个文件的详细比较
            for filename, data in comp_data.items():
                if filename == 'overall':
                    continue

                f.write(f"文件: {filename}\n")

                # 文件大小比较
                file_comp = data.get('file_comparison', {})
                f.write(f"  文件大小比较:\n")
                f.write(f"    RV大小: {file_comp.get('rv_size', 0)} bytes\n")
                f.write(f"    RVV大小: {file_comp.get('rvv_size', 0)} bytes\n")
                f.write(f"    大小差异: {file_comp.get('size_diff', 0)} bytes\n")

                # 误差指标
                errors = data.get('error_metrics', {})
                f.write(f"  误差指标:\n")
                if errors.get('tensors_equal', False):
                    f.write(f"    ✓ 张量完全相等\n")
                else:
                    f.write(f"    平均绝对误差: {errors.get('mean_absolute_error', 0):.6f}\n")
                    f.write(f"    平均相对误差: {errors.get('mean_relative_error', 0):.6f}\n")
                    f.write(f"    最大绝对误差: {errors.get('max_absolute_error', 0):.6f}\n")
                    f.write(f"    最大相对误差: {errors.get('max_relative_error', 0):.6f}\n")
                    f.write(f"    MSE: {errors.get('mse', 0):.6f}\n")
                    f.write(f"    RMSE: {errors.get('rmse', 0):.6f}\n")
                    f.write(f"    相关性: {errors.get('correlation', 0):.6f}\n")
                f.write(f"    有效比较: {errors.get('valid_comparison', False)}\n")

                f.write("\n")

        f.write("报告生成完成时间: " + str(datetime.now()) + "\n")

    print(f"详细比较报告已保存到: {report_file}")


def analyze_results(output_base_dir):
    """
    分析结果目录结构

    Args:
        output_base_dir: 输出基础目录
    """
    print(f"分析结果目录: {output_base_dir}")

    # 查找所有set目录
    set_pattern = os.path.join(output_base_dir, "set_*")
    set_dirs = glob.glob(set_pattern)

    for set_dir in set_dirs:
        set_name = os.path.basename(set_dir)
        print(f"\n{set_name}:")

        # 查找rv和rvv目录
        rv_dir = os.path.join(set_dir, "rv")
        rvv_dir = os.path.join(set_dir, "rvv")

        if os.path.exists(rv_dir):
            print("  RV结果:")
            for so_dir in glob.glob(os.path.join(rv_dir, "*")):
                so_name = os.path.basename(so_dir)
                print(f"    {so_name}:")

                # 统计输入文件
                input_dir = os.path.join(so_dir, "input_data")
                if os.path.exists(input_dir):
                    input_files = find_input_files(input_dir)
                    print(f"      输入文件: {len(input_files)} 个")

                # 统计输出文件
                output_dir = os.path.join(so_dir, "output_data")
                if os.path.exists(output_dir):
                    output_files = find_output_files(output_dir)
                    print(f"      输出文件: {len(output_files)} 个")

        if os.path.exists(rvv_dir):
            print("  RVV结果:")
            for so_dir in glob.glob(os.path.join(rvv_dir, "*")):
                so_name = os.path.basename(so_dir)
                print(f"    {so_name}:")

                # 统计输入文件
                input_dir = os.path.join(so_dir, "input_data")
                if os.path.exists(input_dir):
                    input_files = find_input_files(input_dir)
                    print(f"      输入文件: {len(input_files)} 个")

                # 统计输出文件
                output_dir = os.path.join(so_dir, "output_data")
                if os.path.exists(output_dir):
                    output_files = find_output_files(output_dir)
                    print(f"      输出文件: {len(output_files)} 个")


def main():
    import argparse

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='处理RV和RVV so文件并进行比较')
    parser.add_argument('--input', type=str, help='输入数据路径（文件或文件夹）')
    parser.add_argument('--output', type=str, required=True, help='输出目录路径')
    parser.add_argument('--rounds', type=int, default=10, help='运行轮数（默认: 10）')
    parser.add_argument('--no-save-outputs', action='store_true', help='不保存输出数据（默认保存）')
    parser.add_argument('--no-save-random-inputs', action='store_true', help='不保存随机生成的输入数据（默认保存）')
    parser.add_argument('--no-compare', action='store_true', help='不比较RV和RVV的结果')
    parser.add_argument('--analyze', action='store_true', help='只分析现有结果，不执行新测试')
    parser.add_argument('--no-resume', action='store_true', help='禁用端点续行功能（默认启用）')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='张量比较容差（默认: 1e-6）')
    parser.add_argument('op_lib_dir', type=str, help='op_lib目录路径')

    args = parser.parse_args()

    # 设置目录路径
    op_lib_directory = args.op_lib_dir
    output_directory = args.output

    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    if args.analyze:
        # 只分析现有结果
        analyze_results(output_directory)
        return

    print("开始处理RV和RVV so文件...")
    print(f"OP_LIB目录: {op_lib_directory}")
    print(f"输出目录: {output_directory}")
    print(f"运行轮数: {args.rounds}")
    print(f"保存输出: {not args.no_save_outputs}")
    print(f"保存随机输入: {not args.no_save_random_inputs}")
    print(f"比较结果: {not args.no_compare}")
    print(f"端点续行: {not args.no_resume}")
    print(f"容差阈值: {args.tolerance}")
    print("-" * 60)

    # 处理所有匹配的so文件
    comparisons = process_matching_so_files(
        op_lib_dir=op_lib_directory,
        output_base_dir=output_directory,
        rounds=args.rounds,
        save_outputs=not args.no_save_outputs,
        save_random_inputs=not args.no_save_random_inputs,
        compare_results=not args.no_compare,
        resume=not args.no_resume,
        tolerance=args.tolerance
    )

    print("\n所有文件处理完成!")
    print(f"结果保存在: {output_directory}")


if __name__ == "__main__":
    main()