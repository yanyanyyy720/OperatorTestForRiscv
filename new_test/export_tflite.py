import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re


def parse_single_file_tflite(file_path):
    """
    解析tflite格式的性能数据文件

    Args:
        file_path: 文件路径

    Returns:
        dict: 解析结果
    """
    try:
        # 读取文件
        df = pd.read_csv(file_path, skipinitialspace=True, skip_blank_lines=True)
        df.columns = df.columns.str.strip()

        # 检查是否有足够的行数
        if len(df) < 3:
            raise ValueError("文件行数不足，至少需要3行数据")

        # 获取第二行和最后一行
        second_row = df.iloc[1]  # 第二行（索引为1）
        last_row = df.iloc[-1]  # 最后一行

        # 计算指标（最后一行减第二行除以2）
        total_time_ms = (last_row['Time(ms)'] - second_row['Time(ms)']) / 2
        total_page_faults = ((last_row['Minor Faults'] - second_row['Minor Faults']) +
                             (last_row['Major Faults'] - second_row['Major Faults'])) / 2
        total_cpu_time = (last_row['Total CPU(ms)'] - second_row['Total CPU(ms)']) / 2
        user_cpu_time = (last_row['User CPU(ms)'] - second_row['User CPU(ms)']) / 2

        # 计算RSS最大值（从第二行到最后一行）
        execution_data = df.iloc[1:]

        results = {
            'file_path': str(file_path),
            'rss_max_mb': execution_data['RSS(MB)'].max(),
            'avg_execution_time_ms': total_time_ms,
            'page_fault_rate_per_sec': total_page_faults / (total_time_ms / 1000) if total_time_ms > 0 else 0,
            'user_cpu_ratio': user_cpu_time / total_cpu_time if total_cpu_time > 0 else 0,
            'avg_cpu_time_ms': total_cpu_time,
            'total_time_ms': total_time_ms * 2,  # 恢复原始总时间
            'total_page_faults': total_page_faults * 2,  # 恢复原始总缺页数
            'threads_avg': execution_data['Threads'].mean(),
            'data_seg_mb': execution_data['Data Seg(MB)'].max()
        }
        return results
    except Exception as e:
        print(f"Error parsing tflite file {file_path}: {e}")
        return None


def parse_single_file_original(file_path):
    """
    解析原始格式的性能数据文件

    Args:
        file_path: 文件路径

    Returns:
        dict: 解析结果
    """
    try:
        # 读取文件
        df = pd.read_csv(file_path, skipinitialspace=True, skip_blank_lines=True)
        df.columns = df.columns.str.strip()

        # 找到执行开始和执行的记录
        start_record = df[df['Name'] == '执行开始'].iloc[0]
        end_record = df[df['Name'] == '执行完成'].iloc[0]

        # 计算执行时间段内的数据
        execution_data = df[(df['Time(ms)'] >= start_record['Time(ms)']) &
                            (df['Time(ms)'] <= end_record['Time(ms)'])]

        # 计算指标
        total_time_ms = end_record['Time(ms)'] - start_record['Time(ms)']
        total_page_faults = (end_record['Minor Faults'] - start_record['Minor Faults'] +
                             end_record['Major Faults'] - start_record['Major Faults'])
        total_cpu_time = end_record['Total CPU(ms)'] - start_record['Total CPU(ms)']
        user_cpu_time = end_record['User CPU(ms)'] - start_record['User CPU(ms)']

        results = {
            'file_path': str(file_path),
            'rss_max_mb': execution_data['RSS(MB)'].max(),
            'avg_execution_time_ms': total_time_ms / 10,
            'page_fault_rate_per_sec': total_page_faults / (total_time_ms / 1000) if total_time_ms > 0 else 0,
            'user_cpu_ratio': user_cpu_time / total_cpu_time if total_cpu_time > 0 else 0,
            'avg_cpu_time_ms': total_cpu_time / 10,
            'total_time_ms': total_time_ms,
            'total_page_faults': total_page_faults,
            'threads_avg': execution_data['Threads'].mean(),
            'data_seg_mb': execution_data['Data Seg(MB)'].max()
        }
        return results
    except Exception as e:
        print(f"Error parsing original file {file_path}: {e}")
        return None


def parse_single_file(file_path):
    """
    自动检测文件格式并选择合适的解析函数
    """
    try:
        # 读取文件前几行来检测格式
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        # 检查是否是tflite格式（包含Time(ms), RSS(MB)等列名）
        if 'Time(ms)' in first_line and 'RSS(MB)' in first_line:
            return parse_single_file_tflite(file_path)
        else:
            return parse_single_file_original(file_path)
    except Exception as e:
        print(f"Error detecting file format for {file_path}: {e}")
        return None


def analyze_file_sequence(files):
    """
    分析文件序列，计算统计指标

    Args:
        files: 文件路径序列

    Returns:
        dict: 统计结果，包含均值和原始数据
    """
    # 解析所有文件
    results = []
    valid_files = []

    for file_path in files:
        result = parse_single_file(file_path)
        if result is not None:
            results.append(result)
            valid_files.append(file_path)

    if not results:
        return {"error": "No valid files found"}

    # 转换为DataFrame便于计算
    df = pd.DataFrame(results)

    # 计算统计指标
    stats = {
        'file_count': len(results),
        'files': valid_files,
        'mean': {
            'rss_max_mb': df['rss_max_mb'].mean(),
            'avg_execution_time_ms': df['avg_execution_time_ms'].mean(),
            'page_fault_rate_per_sec': df['page_fault_rate_per_sec'].mean(),
            'user_cpu_ratio': df['user_cpu_ratio'].mean(),
            'avg_cpu_time_ms': df['avg_cpu_time_ms'].mean(),
            'threads_avg': df['threads_avg'].mean(),
            'data_seg_mb': df['data_seg_mb'].mean()
        },
        'std': {
            'rss_max_mb': df['rss_max_mb'].std(),
            'avg_execution_time_ms': df['avg_execution_time_ms'].std(),
            'page_fault_rate_per_sec': df['page_fault_rate_per_sec'].std(),
            'user_cpu_ratio': df['user_cpu_ratio'].std(),
            'avg_cpu_time_ms': df['avg_cpu_time_ms'].std(),
            'threads_avg': df['threads_avg'].std(),
            'data_seg_mb': df['data_seg_mb'].std()
        },
        'min': {
            'rss_max_mb': df['rss_max_mb'].min(),
            'avg_execution_time_ms': df['avg_execution_time_ms'].min(),
            'page_fault_rate_per_sec': df['page_fault_rate_per_sec'].min(),
            'user_cpu_ratio': df['user_cpu_ratio'].min(),
            'avg_cpu_time_ms': df['avg_cpu_time_ms'].min(),
            'threads_avg': df['threads_avg'].min(),
            'data_seg_mb': df['data_seg_mb'].min()
        },
        'max': {
            'rss_max_mb': df['rss_max_mb'].max(),
            'avg_execution_time_ms': df['avg_execution_time_ms'].max(),
            'page_fault_rate_per_sec': df['page_fault_rate_per_sec'].max(),
            'user_cpu_ratio': df['user_cpu_ratio'].max(),
            'avg_cpu_time_ms': df['avg_cpu_time_ms'].max(),
            'threads_avg': df['threads_avg'].max(),
            'data_seg_mb': df['data_seg_mb'].max()
        },
        'raw_data': results  # 包含所有原始数据
    }

    return stats


def parse_directory_structure(root_dir):
    """
    解析新的目录结构，按backend、size、op分组分析

    Args:
        root_dir: 根目录路径

    Returns:
        dict: 分组分析结果
    """
    root_path = Path(root_dir)
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # 遍历目录结构
    for run_dir in root_path.glob('run_*'):
        if not run_dir.is_dir():
            continue
        # 只处理 run_0 到 run_9
        run_number = int(run_dir.name.split('_')[1])
        if run_number > 9:
            continue
        # 查找output目录
        output_dir = run_dir / 'output'
        if not output_dir.exists():
            continue

        # 遍历tvm和tflite目录
        for backend in ['rv', 'tflite']:
            backend_dir = output_dir / backend
            if not backend_dir.exists():
                continue

            # 遍历size目录（large, medium, small）
            for size in ['large', 'medium', 'small']:
                size_dir = backend_dir / size
                if not size_dir.exists():
                    continue

                # 根据backend类型查找文件
                if backend == 'rv':
                    # tvm目录结构：tvm/size/tvm_output/op_metric.cfg
                    tvm_output_dir = size_dir / 'tvm_output'
                    if not tvm_output_dir.exists():
                        continue

                    # 查找所有metric文件
                    pattern = r'^(.+)_metric\.cfg$'
                    for metric_file in tvm_output_dir.glob('*_metric.cfg'):
                        match = re.match(pattern, metric_file.name)
                        if match:
                            op_name = match.group(1)
                            results[backend][size][op_name].append(str(metric_file))

                elif backend == 'tflite':
                    # tflite目录结构：tflite/size/op_size.tflite.cfg
                    pattern = r'^(.+)' + r'\.tflite\.cfg$'
                    for tflite_file in size_dir.glob(f'*_{size}.tflite.cfg'):
                        match = re.match(pattern, tflite_file.name)
                        if match:
                            op_name = match.group(1)
                            results[backend][size][op_name].append(str(tflite_file))

    # 对每个分组进行分析
    analysis_results = {}

    for backend, sizes in results.items():
        analysis_results[backend] = {}
        for size, ops in sizes.items():
            analysis_results[backend][size] = {}
            for op_name, files in ops.items():
                if files:  # 确保文件列表不为空
                    analysis_results[backend][size][op_name] = analyze_file_sequence(files)

    return analysis_results


def save_results_to_json(results, output_file):
    """
    将结果保存为JSON文件

    Args:
        results: 分析结果
        output_file: 输出文件路径
    """

    # 自定义JSON序列化器，处理numpy类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            return super().default(obj)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def print_summary(results):
    """
    打印分析结果摘要
    """
    print("=" * 80)
    print("性能数据分析汇总")
    print("=" * 80)

    for backend, sizes in results.items():
        print(f"\nBackend类型: {backend}")
        print("-" * 40)

        for size, ops in sizes.items():
            print(f"  Size: {size}")
            print("  " + "-" * 30)

            for op_name, stats in ops.items():
                if 'error' in stats:
                    print(f"    {op_name}: {stats['error']}")
                    continue

                mean = stats['mean']
                print(f"    {op_name}:")
                print(f"      文件数量: {stats['file_count']}")
                print(f"      RSS最大值均值(MB): {mean['rss_max_mb']:.2f}")
                print(f"      平均执行时间(ms): {mean['avg_execution_time_ms']:.3f}")
                print(f"      缺页率均值(次/秒): {mean['page_fault_rate_per_sec']:.2f}")
                print(f"      用户态CPU占比均值: {mean['user_cpu_ratio']:.3f}")
                print(f"      CPU时间均值(ms): {mean['avg_cpu_time_ms']:.2f}")


# 使用示例
if __name__ == "__main__":
    # 示例1: 分析文件序列
    # files = [
    #     "path/to/file1.txt",
    #     "path/to/file2.txt",
    #     # 添加更多文件路径
    # ]
    #
    # sequence_results = analyze_file_sequence(files)
    # print("文件序列分析结果:")
    # print(json.dumps(sequence_results, indent=2, ensure_ascii=False))

    # 示例2: 解析目录结构
    root_directory = "../new_test/exported_models"

    if os.path.exists(root_directory):
        directory_results = parse_directory_structure(root_directory)

        # 打印摘要
        print_summary(directory_results)

        # 保存详细结果到JSON
        save_results_to_json(directory_results, "performance_analysis_results.json")
        print(f"\n详细结果已保存到: performance_analysis_results.json")

        # 显示JSON结构示例
        print("\nJSON结果结构示例:")
        sample_structure = {
            "tvm": {
                "large": {
                    "ADD": {
                        "file_count": 10,
                        "mean": {
                            "rss_max_mb": 3.2,
                            "avg_execution_time_ms": 1.317,
                            # ... 其他指标
                        },
                        "raw_data": [
                            # 每个文件的原始数据
                        ]
                    }
                }
            },
            "tflite": {
                "large": {
                    "ABS": {
                        "file_count": 10,
                        "mean": {
                            "rss_max_mb": 2.8,
                            "avg_execution_time_ms": 1.125,
                            # ... 其他指标
                        },
                        "raw_data": [
                            # 每个文件的原始数据
                        ]
                    }
                }
            }
        }
        print(json.dumps(sample_structure, indent=2, ensure_ascii=False))
    else:
        print(f"目录 {root_directory} 不存在，请提供有效的根目录路径")

        # 创建示例目录结构进行演示
        print("\n创建示例目录结构进行演示...")
        # 这里可以添加创建示例目录和文件的代码