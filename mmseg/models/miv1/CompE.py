import torch


def compute_information_entropy(feature_map):  # feature_map: (H,W)
    # 1. 归一化特征图到[0,1]，使其符合概率分布
    f = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-6)

    # 2. 将特征值转换为概率分布
    p = f / (torch.sum(f) + 1e-6)

    # 3. 计算信息熵: -Σ(pi * log(pi))
    entropy = -torch.sum(p * torch.log2(p + 1e-6))

    return entropy


# 测试例子1 - 特征非常明显
feature1 = torch.zeros(8, 8)
feature1[3:5, 3:5] = 1.0

# 测试例子2 - 特征比较明显但不那么极端
feature2 = torch.zeros(8, 8)
feature2[4:5, 3:5] = 1.0

# 测试例子3 - 特征不明显
feature3 = torch.ones(8, 8) * 0.5 + torch.rand(8, 8) * 0.1

# 计算熵值
entropy1 = compute_information_entropy(feature1)
entropy2 = compute_information_entropy(feature2)
entropy3 = compute_information_entropy(feature3)

print("Feature 1 entropy:", entropy1.item())
print("Feature 2 entropy:", entropy2.item())
print("Feature 3 entropy:", entropy3.item())

# 可视化特征图
print("\nFeature 1:")
print(feature1)
print("\nFeature 2:")
print(feature2)
print("\nFeature 3:")
print(feature3)