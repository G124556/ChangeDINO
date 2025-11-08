import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class RegionConsistencyModule(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.k_size = k_size

        # 特征描述子生成
        self.feature_embedding = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 相似性分析
        self.similarity_analysis = RegionSimilarityAnalysis(k_size)

        # 动态特征聚合
        self.feature_aggregation = DynamicFeatureAggregation(channels, k_size)

        # 输出调制
        self.output_modulation = OutputModulation(channels)

    def forward(self, x):
        # 1. 生成特征描述子
        feat_embed = self.feature_embedding(x)

        # 2. 区域相似性分析
        similarity = self.similarity_analysis(feat_embed)

        # 3. 特征聚合
        enhanced = self.feature_aggregation(x, similarity)

        # 4. 输出调制
        output = self.output_modulation(x, enhanced)

        return output


class RegionSimilarityAnalysis(nn.Module):
    def __init__(self, k_size):
        super().__init__()
        self.k_size = k_size
        self.padding = k_size // 2

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 提取局部区域特征
        patches = F.unfold(x,
                           kernel_size=self.k_size,
                           padding=self.padding)  # [B, C*k*k, HW]

        patches = patches.view(B, C, self.k_size * self.k_size, H * W)

        # 2. 计算区域内像素的相似度
        center = x.view(B, C, 1, H * W)  # [B, C, 1, HW]

        # 计算余弦相似度
        similarity = (center * patches).sum(dim=1)  # [B, k*k, HW]
        norm_center = torch.norm(center, dim=1)
        norm_patches = torch.norm(patches, dim=1)
        similarity = similarity / (norm_center * norm_patches + 1e-7)

        # 归一化相似度
        similarity = F.softmax(similarity, dim=1)  # [B, k*k, HW]

        return similarity


class DynamicFeatureAggregation(nn.Module):
    def __init__(self, channels, k_size):
        super().__init__()
        self.k_size = k_size
        self.channels = channels

        self.enhance = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, similarity):
        B, C, H, W = x.shape

        # 1. 展开特征进行聚合
        x_unfold = F.unfold(x,
                            kernel_size=self.k_size,
                            padding=self.k_size // 2)

        # 2. 重塑维度以进行加权
        x_unfold = x_unfold.view(B, self.channels, -1, H * W)
        similarity = similarity.unsqueeze(1)  # [B, 1, k*k, HW]

        # 3. 加权聚合
        weighted = x_unfold * similarity
        aggregated = weighted.sum(dim=2)  # [B, C, HW]

        # 4. 重塑回原始维度
        aggregated = aggregated.view(B, C, H, W)

        # 5. 特征增强
        enhanced = self.enhance(torch.cat([x, aggregated], dim=1))

        return enhanced


class OutputModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, original, enhanced):
        weights = self.weight_gen(torch.cat([original, enhanced], dim=1))
        output = original + weights * enhanced
        return output


def tet_region_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建测试数据
    B, C, H, W = 16, 256, 32, 32
    x = torch.randn(B, C, H, W).to(device)

    # 创建模型
    model = RegionConsistencyModule(C).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(x)

    # 打印信息
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 计算相似度分布
    similarity = model.similarity_analysis(model.feature_embedding(x))
    print(f"Similarity shape: {similarity.shape}")
    print(f"Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")

    return output, similarity


if __name__ == "__main__":
    output, similarity = tet_region_consistency()