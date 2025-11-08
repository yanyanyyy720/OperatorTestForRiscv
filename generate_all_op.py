from llm import ChatModelWrapper
from op_list import *

chat = ChatModelWrapper(prompt_name="generate_model",temperature=0)

for op in tflite_all_op_list:
    result = chat.chat("generate a model which use this api : " + op)
    print(result)
    break