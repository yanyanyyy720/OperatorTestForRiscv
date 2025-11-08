import os
import glob
from pathlib import Path

from network_util import execute_command


def process_tflite_files(process_func, base_dir="./exported_models", i_range=range(1, 11)):
    """
    处理目标文件夹下的tflite文件

    Args:
        process_func: 处理函数，接受两个参数(输入tflite路径, 输出cfg路径)
        base_dir: 基础目录路径，默认为当前目录
        i_range: i的范围，默认为1到10
    """

    # 定义模型规模子目录
    sizes = ["large", "medium", "small"]

    for i in i_range:
        # 构造输入和输出基础路径
        input_base = os.path.join(base_dir, f"run_{i}", "tflite")
        output_base = os.path.join(base_dir, f"run_{i}", "output", "tflite")

        # 遍历每个规模子目录
        for size in sizes:
            # 构造模式匹配路径
            pattern = os.path.join(input_base, size, "*.tflite")

            # 查找所有匹配的tflite文件
            tflite_files = glob.glob(pattern)

            for tflite_path in tflite_files:
                # 从完整路径中提取文件名（不含扩展名）
                filename = Path(tflite_path).stem

                # 构造输出路径，包含size文件夹
                # 例如: run_2/output/tflite/large/ABS_large.cfg
                output_dir = os.path.join(output_base, size)
                # cfg_path = os.path.join(output_dir, f"{filename}.cfg")

                # 确保输出目录存在（包括size子目录）
                os.makedirs(output_dir, exist_ok=True)

                # 调用处理函数
                process_func(tflite_path, output_dir)
                print(f"Processed: {tflite_path} -> {output_dir}")


# 示例处理函数
def example_process_func(input_path, output_path):
    """
    示例处理函数，您需要根据实际需求实现具体的处理逻辑
    """
    # 这里应该是您实际的tflite文件处理代码
    # 例如：模型转换、配置生成等
    print(execute_command(f"/home/yan/workdir/new_test/hello_world_test ./workdir/new_test/{input_path} ./workdir/new_test/{output_path}"))


    # 示例：只是创建并写入一些基本信息
    # with open(output_path, 'w') as f:
    #     f.write(f"# Config file for {input_path}\n")
    #     f.write(f"processed_time: {os.path.getmtime(input_path)}\n")
    #     f.write(f"file_size: {os.path.getsize(input_path)} bytes\n")


# 使用示例
if __name__ == "__main__":
    # 方法1: 使用默认参数（i从1到10，当前目录）
    print(execute_command("mount -t nfs 192.168.10.9:/home/yab/PycharmProjects/LLM"))
    process_tflite_files(example_process_func)

    # 方法2: 指定自定义参数
    # process_tflite_files(
    #     process_func=example_process_func,
    #     base_dir="/path/to/your/directory",
    #     i_range=range(1, 6)  # i从1到5
    # )