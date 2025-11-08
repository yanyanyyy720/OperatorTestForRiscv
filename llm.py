from langchain import requests
from langchain.chains.question_answering.map_reduce_prompt import messages
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
import json
from pathlib import Path

# 设置环境变量（根据您的需求选择）
os.environ["OPENAI_API_BASE"] = "https://api.geekai.pro/v1"
os.environ["OPENAI_API_KEY"] = "sk-xaS1LZa4mHHn6t5HQINZk8wweS222b301TVc0RmXs0X9YUB5"


def check_api_balance(api_key, base_url):
    """查询API余额（如果服务商支持此功能）"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        # 注意：这不是标准OpenAI API，需要查看geekai.pro的文档
        response = requests.get(f"{base_url}/billing/usage", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"无法获取余额信息: {response.status_code}")
            return None
    except Exception as e:
        print(f"查询余额时出错: {str(e)}")
        return None


import json
from pathlib import Path


class PromptManager:
    """管理 prompt 模板的类，使用更易读的格式存储"""

    def __init__(self, prompt_dir="prompts"):
        self.prompt_dir = Path(prompt_dir)
        self.prompts = {}
        self.load_prompts()

    def load_prompts(self):
        """从指定目录加载所有 prompt 模板"""
        self.prompt_dir.mkdir(parents=True, exist_ok=True)

        # 加载所有 prompt 文件
        for file in self.prompt_dir.glob("*.txt"):
            try:
                prompt_name = file.stem  # 使用文件名作为 prompt 名称

                # 读取内容文件
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 尝试读取元数据文件（如果存在）
                meta_file = self.prompt_dir / f"{prompt_name}.meta.json"
                description = ""
                variables = []

                if meta_file.exists():
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                        description = meta_data.get("description", "")
                        variables = meta_data.get("variables", [])

                # 存储 prompt 数据
                self.prompts[prompt_name] = {
                    "name": prompt_name,
                    "content": content,
                    "description": description,
                    "variables": variables
                }
            except Exception as e:
                print(f"Error loading prompt file {file}: {str(e)}")

    def get_prompt(self, prompt_name):
        """获取指定名称的 prompt"""
        return self.prompts.get(prompt_name)

    def add_prompt(self, prompt_name, content, description="", variables=None):
        """添加新的 prompt 模板"""
        # 存储到内存
        prompt_data = {
            "name": prompt_name,
            "content": content,
            "description": description,
            "variables": variables or []
        }
        self.prompts[prompt_name] = prompt_data

        # 保存到文件系统
        # 1. 保存内容为纯文本文件（保留所有格式）
        content_file = self.prompt_dir / f"{prompt_name}.txt"
        with open(content_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # 2. 保存元数据为 JSON 文件
        meta_data = {
            "description": description,
            "variables": variables or []
        }
        meta_file = self.prompt_dir / f"{prompt_name}.meta.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)

        return prompt_data


class ChatModelWrapper:
    def __init__(self, system_prompt="", prompt_name=None, model_name="gpt-5-mini-2025-08-07",
                 temperature=0.7, max_tokens=10000, prompt_manager=None):
        """
        初始化大模型封装类，支持全局规则设定

        参数:
        system_prompt: 全局系统提示 (默认: "")
        prompt_name: 要加载的 prompt 名称 (默认: None)
        model_name: 模型名称 (默认: "claude-sonnet-4-20250514")
        temperature: 生成温度 (默认: 0.7)
        max_tokens: 最大输出长度 (默认: 4000)
        prompt_manager: PromptManager 实例 (默认: None)
        """
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.__system_prompt = ""  # 私有属性存储系统提示
        self.prompt_manager = prompt_manager or PromptManager()

        # 优先使用 prompt_name 加载系统提示
        if prompt_name:
            self.set_system_prompt_by_name(prompt_name)
        elif system_prompt:
            self.__set_system_prompt(system_prompt)

    def __set_system_prompt(self, system_prompt):
        """
        内部方法：设置/更新全局系统提示

        参数:
        system_prompt: 系统角色设定
        """
        self.__system_prompt = system_prompt

    def set_system_prompt_by_name(self, prompt_name):
        """
        根据名称设置系统提示

        参数:
        prompt_name: prompt 模板名称
        """
        prompt_data = self.prompt_manager.get_prompt(prompt_name)
        if not prompt_data:
            raise ValueError(f"Prompt '{prompt_name}' not found")

        # 获取 prompt 内容
        prompt_content = prompt_data["content"]

        # 设置系统提示
        self.__set_system_prompt(prompt_content)
        return prompt_content

    def chat(self, user_input):
        """
        执行聊天对话（只发送系统提示和当前用户输入）

        参数:
        user_input: 用户输入内容

        返回:
        模型生成的回复
        """
        # 构建消息列表（系统提示 + 当前用户输入）
        messages = []
        if self.__system_prompt:
            messages.append(SystemMessage(content=self.__system_prompt))
        messages.append(HumanMessage(content=user_input))

        # 获取模型回复
        response = self.model(messages)

        return response.content

    def add_custom_prompt(self, prompt_name, content, description="", variables=None):
        """
        添加自定义 prompt 模板

        参数:
        prompt_name: prompt 名称
        content: prompt 内容
        description: prompt 描述 (可选)
        variables: 变量列表 (可选)
        """
        return self.prompt_manager.add_prompt(prompt_name, content, description, variables)


# 使用示例
if __name__ == "__main__":
    import requests

    url = "https://api.geekai.pro/v1/token/quota"

    payload = {}
    headers = {
        'Authorization': 'Bearer sk-7rfCFLwLLSRj5vqQOZ87pOiF6Yq3TIuNMWGnRmWz48FOWYim'
    }
    check_api_balance("sk-7rfCFLwLLSRj5vqQOZ87pOiF6Yq3TIuNMWGnRmWz48FOWYim",url)
    response = requests.request("GET", url, headers=headers, data=payload)
    #
    print(response.text)

#     # 创建聊天模型实例
#     chat_model = ChatModelWrapper()
#
#     # 添加自定义 prompt
#     chat_model.add_custom_prompt(
#         prompt_name="generate_new_op",
#         content='''Please generate a TensorFlow model following this template:
# ```python
# class Squared(tf.Module):
# @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
# def __call__(self, x):
# return tf.square(x)
# ```
# Requirements:
# - The model must be compilable with static shape definitions
# - Use dimensions suitable for microprocessors to ease computational load
# - The model should have sufficient randomness, so you have high autonomy to decide various aspects of the model
# - Strictly follow the example template structure
# - Use only a single operator
# - Assume tf is already imported
# - Use only low-level TensorFlow APIs, not Keras
# - Return only the class code
# - Use Different module each time
# The model should be simple, efficient for embedded systems, and follow the exact pattern shown in the template with appropriate input signature decoration.''',
#         description="用于创建单算子的prompt",
#         variables=[]
#     )
#
#     # 使用带变量的 prompt
#     chat_model.set_system_prompt_by_name("generate_new_op")
#     response = chat_model.chat(
#         "生成卷积模型")
#     print(f"摘要结果: {response}")