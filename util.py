import re


def extract_python_code(text: str) -> str:
    """
    从大模型输出中提取被 ```python 和 ``` 包裹的 Python 代码块

    参数:
        text (str): 大模型输出的完整文本

    返回:
        str: 提取到的 Python 代码字符串。如果未找到匹配则返回空字符串
    """
    pattern = r'```python(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # 返回第一个匹配到的代码块，并去除首尾空白
        return matches[0].strip()
    return ""