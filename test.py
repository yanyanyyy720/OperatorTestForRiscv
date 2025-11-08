from code_manager import CodeManager
from code_runner import code2object, run_tvm
from model_convert import export_model
from llm import ChatModelWrapper
import random
from util import  extract_python_code
cm = CodeManager("code/init_model")
chat = ChatModelWrapper(prompt_name="fuse")

# 执行融合过程若干次（例如5次）
for _ in range(100):  # 可根据需要调整次数
    # 从模型库中选取两个算子（按分数排序）
    selected_models = cm.top_k(2)
    if len(selected_models) < 2:
        continue  # 确保有两个模型可用

    model1 = selected_models[0]
    model2 = selected_models[1]

    # 准备发送给大模型的提示
    prompt = f"Operator 1:\n{model1[0]}\n\nOperator 2:\n{model2[0]}"

    # 获取融合后的代码
    fused_code = chat.chat(prompt)

    fused_code = extract_python_code(fused_code)
    # 尝试编译融合后的模型
    success, fused_model = code2object(fused_code)
    print(fused_code)


    # 惩罚原始模型的得分
    cm.update(model1[2], None, model1[1] * 0.7)  # 更新分数
    cm.update(model2[2], None, model2[1] * 0.7)

    if not success:
        continue  # 编译失败则跳过
    # 执行新模型
    try:
        newId = cm.add(fused_code)
        # 导出模型
        tflite_path, native_path, device_path = export_model(
            model=fused_model,
            op_name=newId,  # 生成唯一操作名
            tflite_output_dir="./output/tflite_models",
            native_output_dir="./output/tvm_native",
            device_output_dir="./output/tvm_device"
        )

        # 在设备上运行模型
        run_tvm(tvm_path=device_path)

    except Exception as e:
        print(f"Error executing fused model: {e}")