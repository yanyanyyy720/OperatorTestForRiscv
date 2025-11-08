import ast
import sys
import inspect
import traceback
import builtins
import tensorflow as tf

from network_util import execute_command


def code2object(code_text):
    """
    从给定的Python代码文本中创建Model类的实例

    参数:
        code_text (str): 包含类定义的Python代码文本

    返回:
        tuple: (success: bool, result: object | str)
        - 成功时: (True, Model类的实例)
        - 失败时: (False, 详细的错误信息字符串)
    """
    # 1. 检查代码中是否包含名为Model的类
    try:
        # 先编译代码以捕获语法错误
        compiled_code = compile(code_text, "<string>", "exec")

        # 使用AST解析检查类名
        tree = ast.parse(code_text)
        has_model_class = any(
            isinstance(node, ast.ClassDef) and node.name == "Model"
            for node in ast.walk(tree)
        )
        if not has_model_class:
            return False, "代码中未找到名为'Model'的类"
    except SyntaxError as e:
        # 捕获语法错误并返回详细信息
        error_type = type(e).__name__
        error_msg = str(e)
        error_line = e.lineno if hasattr(e, 'lineno') else "未知"
        error_offset = e.offset if hasattr(e, 'offset') else "未知"
        return False, (
            f"语法错误: {error_type}\n"
            f"位置: 第 {error_line} 行, 第 {error_offset} 列\n"
            f"详细信息: {error_msg}\n"
            f"错误上下文: {e.text.strip() if hasattr(e, 'text') and e.text else '无'}"
        )
    except Exception as e:
        # 捕获其他解析错误
        error_trace = traceback.format_exc()
        return False, f"代码解析错误: {str(e)}\n{error_trace}"

    # 2. 创建安全的执行环境（包含tf和完整的builtins）
    namespace = {
        "__builtins__": builtins.__dict__,  # 完整的Python内置函数集
        "tf": tf,  # 预先导入TensorFlow
        "__name__": "__exec__",
        "__file__": "<string>",
    }

    # 3. 执行代码
    try:
        # 使用之前编译的代码执行
        exec(compiled_code, namespace)
    except Exception as e:
        # 捕获执行错误并返回详细信息
        error_type = type(e).__name__
        error_msg = str(e)
        tb = traceback.extract_tb(sys.exc_info()[2])

        # 提取错误位置信息
        error_location = ""
        if tb:
            last_frame = tb[-1]
            filename = last_frame.filename
            lineno = last_frame.lineno
            function = last_frame.name
            error_location = f"位置: 文件 {filename}, 第 {lineno} 行, 函数 {function}"

        # 提取错误上下文
        error_context = ""
        try:
            if tb:
                # 获取错误行号
                error_line = tb[-1].lineno
                # 提取错误行附近的代码
                lines = code_text.splitlines()
                start = max(0, error_line - 3)
                end = min(len(lines), error_line + 2)
                context_lines = lines[start:end]

                # 标记错误行
                for i, line in enumerate(context_lines, start=start):
                    if i == error_line - 1:  # 行号从1开始，索引从0开始
                        context_lines[i - start] = f"--> {i + 1}: {line}"
                    else:
                        context_lines[i - start] = f"    {i + 1}: {line}"

                error_context = "错误上下文:\n" + "\n".join(context_lines)
        except:
            error_context = "无法提取错误上下文"

        return False, (
            f"代码执行错误: {error_type}\n"
            f"{error_location}\n"
            f"详细信息: {error_msg}\n"
            f"{error_context}\n"
            f"完整堆栈跟踪:\n{traceback.format_exc()}"
        )

    # 4. 获取Model类
    if "Model" not in namespace:
        return False, "执行后未找到'Model'类"

    ModelClass = namespace["Model"]

    # 5. 检查是否为类
    if not isinstance(ModelClass, type):
        return False, "'Model'不是有效的类"

    # 6. 尝试实例化（无参数）
    try:
        # 检查__init__方法是否需要参数
        if hasattr(ModelClass, "__init__"):
            init_signature = inspect.signature(ModelClass.__init__)
            # 检查是否有必需的参数（不包括self）
            required_params = [
                p for p in init_signature.parameters.values()
                if p.default is p.empty and p.name != "self"
            ]
            if required_params:
                param_names = ", ".join(p.name for p in required_params)
                return False, f"Model类需要初始化参数: {param_names}"

        # 实例化对象
        instance = ModelClass()
        return True, instance
    except TypeError as e:
        # 处理参数错误
        if "positional arguments" in str(e):
            return False, "Model类需要初始化参数"

        # 捕获其他类型错误
        error_type = type(e).__name__
        error_msg = str(e)
        tb = traceback.extract_tb(sys.exc_info()[2])

        # 提取错误位置信息
        error_location = ""
        if tb:
            last_frame = tb[-1]
            filename = last_frame.filename
            lineno = last_frame.lineno
            function = last_frame.name
            error_location = f"位置: 文件 {filename}, 第 {lineno} 行, 函数 {function}"

        return False, (
            f"实例化错误: {error_type}\n"
            f"{error_location}\n"
            f"详细信息: {error_msg}\n"
            f"完整堆栈跟踪:\n{traceback.format_exc()}"
        )
    except Exception as e:
        # 捕获其他实例化错误
        error_type = type(e).__name__
        error_msg = str(e)
        tb = traceback.extract_tb(sys.exc_info()[2])

        # 提取错误位置信息
        error_location = ""
        if tb:
            last_frame = tb[-1]
            filename = last_frame.filename
            lineno = last_frame.lineno
            function = last_frame.name
            error_location = f"位置: 文件 {filename}, 第 {lineno} 行, 函数 {function}"

        return False, (
            f"实例化过程中发生错误: {error_type}\n"
            f"{error_location}\n"
            f"详细信息: {error_msg}\n"
            f"完整堆栈跟踪:\n{traceback.format_exc()}"
        )

def run_tvm(tvm_path):
    result = execute_command(f"/home/yan/workdir/cpp/validator-riscv /home/yan/workdir/{tvm_path} /home/yan/workdir/output/tvm_output")
    print(result)
# 使用示例
if __name__ == "__main__":
    # 测试用例1：语法错误
    code1 = """
class Model
    def __init__(self):
        self.value = 42
"""
    success, result = code2object(code1)
    print(f"测试1 - 语法错误: {success}, {result}")

    # 测试用例2：执行错误
    code2 = """
class Model:
    def __init__(self):
        # 故意制造错误
        undefined_function()
"""
    success, result = code2object(code2)
    print(f"测试2 - 执行错误: {success}, {result}")

    # 测试用例3：实例化错误
    code3 = """
class Model:
    def __init__(self):
        self.value = 1 / 0  # 除以零错误
"""
    success, result = code2object(code3)
    print(f"测试3 - 实例化错误: {success}, {result}")

    # 测试用例4：需要参数
    code4 = """
class Model:
    def __init__(self, name):
        self.name = name
"""
    success, result = code2object(code4)
    print(f"测试4 - 需要参数: {success}, {result}")

    # 测试用例5：成功案例
    code5 = """
class Model:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def predict(self, x):
        return self.model(x)
"""
    success, result = code2object(code5)
    if success:
        print("测试5 - 成功创建实例")
        # 使用模型进行预测
        dummy_input = tf.random.normal([1, 10])
        output = result.predict(dummy_input)
        print(f"预测结果形状: {output.shape}")
    else:
        print(f"测试5 - 失败: {result}")