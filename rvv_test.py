import os
import glob
from pathlib import Path

from network_util import execute_command


def process_so_file(so_file_path, cfg_file_path):
    """
    处理.so文件到.cfg文件的转换
    你可以在这里自定义处理逻辑

    Args:
        so_file_path: 输入的.so文件路径
        cfg_file_path: 输出的.cfg文件路径
    """
    # 这里是你的自定义处理逻辑
    # 例如：读取.so文件信息，生成对应的.cfg内容

    # print(f"/home/yan/workdir/cpp/validator-riscv /home/yan/workdir/{so_file_path} /home/yan/workdir/{cfg_file_path}")
    # # 示例：基于.so文件名创建简单的配置
    result = execute_command(
            f"/home/yan/workdir/cpp/validator-riscv /home/yan/workdir/{so_file_path} /home/yan/workdir/{cfg_file_path}")
    print(result)

    print(f"Processed: {so_file_path} -> {cfg_file_path}")


def process_op_lib_so_files(op_lib_dir, output_dir, processing_function):
    """
    处理op_lib目录下的所有.so文件，为每个.so文件创建对应的.cfg文件路径
    并调用处理函数

    Args:
        op_lib_dir: op_lib目录路径
        output_dir: 输出目录路径
        processing_function: 处理函数，接受两个参数(so_path, cfg_path)
    """
    # 查找所有的set_i目录
    set_pattern = os.path.join(op_lib_dir, "set_*")
    set_dirs = glob.glob(set_pattern)

    so_files_count = 0

    for set_dir in set_dirs:
        set_name = os.path.basename(set_dir)

        # 处理rv和rvv子目录
        for subdir in ["rv", "rvv"]:
            so_pattern = os.path.join(set_dir, subdir, "*.so")
            so_files = glob.glob(so_pattern)
            so_files_count += len(so_files)

            for so_file in so_files:
                # 构造对应的输出路径
                so_filename = os.path.basename(so_file)
                cfg_filename = so_filename.replace('.so', '.cfg')

                # 输出目录路径
                output_subdir = os.path.join(output_dir, set_name, subdir)
                output_cfg_path = os.path.join(output_subdir, cfg_filename)

                # 调用处理函数
                processing_function(so_file, output_subdir)

    print(f"Found {so_files_count} .so files to process")


# 使用示例
if __name__ == "__main__":
    # 设置你的目录路径
    op_lib_directory = "./op_lib"  # 修改为你的op_lib目录
    output_directory = "./output_dir"  # 修改为你的输出目录

    # 处理所有.so文件
    process_op_lib_so_files(op_lib_directory, output_directory, process_so_file)
    print("All files have been processed successfully!")