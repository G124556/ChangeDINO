import torch
import torch.nn as nn
import torch.nn.functional as F
from module2 import ComprehensiveChangeDetectionModel

# 首先，复制粘贴您的 ComprehensiveChangeDetectionModel 类定义到这里
# class ComprehensiveChangeDetectionModel(nn.Module):
#     ...  # 您的模型定义

# 测试函数
def test_comprehensive_change_detection_model():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = ComprehensiveChangeDetectionModel().to(device)

    # 生成模拟输入数据
    batch_size = 16

    # xA 和 xB: 多尺度特征 [64, 128, 256, 512, 1024]
    xA = [torch.rand(batch_size, 64, 64, 64).to(device),
          torch.rand(batch_size, 128, 32, 32).to(device),
          torch.rand(batch_size, 256, 16, 16).to(device),
          torch.rand(batch_size, 512, 8, 8).to(device),
          [torch.rand(batch_size, 1024).to(device), torch.rand(batch_size, 1024, 8, 8).to(device)]]
    xB = [torch.rand(batch_size, 64, 64, 64).to(device),
          torch.rand(batch_size, 128, 32, 32).to(device),
          torch.rand(batch_size, 256, 16, 16).to(device),
          torch.rand(batch_size, 512, 8, 8).to(device),
          [torch.rand(batch_size, 1024).to(device), torch.rand(batch_size, 1024, 8, 8).to(device)]]

    # outA 和 outB
    outA = {'pred_logits': torch.rand(batch_size, 900, 256).to(device),
            'pred_boxes': torch.rand(batch_size, 900, 4).to(device)}
    outB = {'pred_logits': torch.rand(batch_size, 900, 256).to(device),
            'pred_boxes': torch.rand(batch_size, 900, 4).to(device)}

    # memoryA 和 memoryB
    memoryA = torch.rand(batch_size, 1360, 256).to(device)
    memoryB = torch.rand(batch_size, 1360, 256).to(device)

    # memory_textA 和 memory_textB
    memory_textA = torch.rand(batch_size, 4, 256).to(device)
    memory_textB = torch.rand(batch_size, 4, 256).to(device)

    # referenceA 和 referenceB
    referenceA = [torch.rand(batch_size, 900, 4).to(device) for _ in range(7)]
    referenceB = [torch.rand(batch_size, 900, 4).to(device) for _ in range(7)]

    # 运行模型
    # try:
    outputs = model(xA, xB, outA, outB, memoryA, memoryB, memory_textA, memory_textB, referenceA, referenceB)
    print("Model forward pass successful!")
    print("Output shapes:")
    for i, output in enumerate(outputs):
        if isinstance(output, torch.Tensor):
            print(f"Output {i}: {output.shape}")
        else:
            print(f"Output {i}: {type(output)}")
    # except Exception as e:
    #     print(f"Error occurred: {e}")


if __name__ == "__main__":
    test_comprehensive_change_detection_model()