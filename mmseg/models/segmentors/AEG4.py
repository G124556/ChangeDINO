import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class SpatialEntropyGuidance(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        # 用于学习通道权重的注意力层
        self.channel_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )

    def compute_spatial_entropy(self, feat):
        """计算空间熵
        Args:
            feat: [B, C, H, W]
        Returns:
            spatial_entropy: [B, H, W]
        """
        B, C, H, W = feat.shape

        # 1. 计算每个通道的空间熵
        channel_entropies = []
        for c in range(C):
            # 获取单个通道的特征图
            channel_feat = feat[:, c, :, :]  # [B, H, W]

            # 归一化到[0,1]
            channel_feat = torch.sigmoid(channel_feat)

            # 计算空间熵
            # entropy = -channel_feat * torch.log(channel_feat + 1e-9) - \
            #           (1 - channel_feat) * torch.log(1 - channel_feat + 1e-9)



            entropy =  torch.log(channel_feat + 1e-9) - \
                      (1 - channel_feat) * torch.log(1 - channel_feat + 1e-9)

            channel_entropies.append(entropy)

        # 2. 堆叠所有通道的熵
        channel_entropies = torch.stack(channel_entropies, dim=1)  # [B, C, H, W]

        # 3. 学习通道权重
        channel_weights = self.channel_attention(
            torch.mean(torch.mean(feat, dim=2), dim=2)  # [B, C]
        ).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 4. 加权平均得到最终的空间熵图
        spatial_entropy = torch.sum(channel_entropies * channel_weights, dim=1)  # [B, H, W]

        return F.normalize(spatial_entropy, dim=(1, 2))

    def find_low_entropy_regions(self, entropy_map):
        """找出低熵区域
        Args:
            entropy_map: [B, H, W]
        Returns:
            low_entropy_mask: [B, 1, H, W]
        """
        B, H, W = entropy_map.shape

        # 使用自适应阈值
        threshold = torch.mean(entropy_map, dim=(1, 2), keepdim=True) * self.threshold
        low_entropy_mask = (entropy_map < threshold).float()

        return low_entropy_mask.unsqueeze(1)

    def feature_enhancement(self, vis_feat, text_feat):
        """特征增强
        Args:
            vis_feat: [B, C, H, W] - ResNet特征
            text_feat: [B, C, H, W] - DINO特征
        """
        # 1. 计算两种特征的空间熵
        vis_entropy = self.compute_spatial_entropy(vis_feat)
        text_entropy = self.compute_spatial_entropy(text_feat)

        # 2. 找出低熵区域
        vis_low_entropy = self.find_low_entropy_regions(vis_entropy)
        text_low_entropy = self.find_low_entropy_regions(text_entropy)

        # 3. 特征增强
        # 当视觉特征熵低时保留视觉特征，当DINO特征熵低时使用DINO引导
        enhanced_feat = vis_feat * vis_low_entropy + \
                        text_feat * text_low_entropy * (1 - vis_low_entropy)

        return enhanced_feat, vis_entropy, text_entropy


def tet_entropy_guidance():
    # 设置随机种子
    torch.manual_seed(42)

    # 创建测试数据
    batch_size = 2
    channels = 256
    H, W = 64, 64

    # 创建模拟的特征图
    # 在某些区域创建明显的模式
    vis_feat = torch.randn(batch_size, channels, H, W)
    text_feat = torch.randn(batch_size, channels, H, W)

    # 添加一些明显的模式
    vis_feat[0, :, 20:30, 20:30] = 2.0  # 模拟明显的视觉特征
    text_feat[0, :, 35:45, 35:45] = 2.0  # 模拟明显的DINO特征

    # 创建模型
    model = SpatialEntropyGuidance()

    # 前向传播
    enhanced_feat, vis_entropy, text_entropy = model.feature_enhancement(vis_feat, text_feat)

    # 可视化结果
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(vis_entropy[0].detach().cpu().numpy())
    plt.title('Visual Feature Entropy')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(text_entropy[0].detach().cpu().numpy())
    plt.title('DINO Feature Entropy')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(torch.mean(enhanced_feat[0], dim=0).detach().cpu().numpy())
    plt.title('Enhanced Feature')
    plt.colorbar()

    plt.show()

    # 打印一些统计信息
    print("\n=== 特征统计信息 ===")
    print(f"视觉特征熵范围: [{vis_entropy.min():.3f}, {vis_entropy.max():.3f}]")
    print(f"DINO特征熵范围: [{text_entropy.min():.3f}, {text_entropy.max():.3f}]")
    print(f"增强特征范围: [{enhanced_feat.min():.3f}, {enhanced_feat.max():.3f}]")

    return enhanced_feat, vis_entropy, text_entropy


if __name__ == "__main__":
    enhanced_feat, vis_entropy, text_entropy = tet_entropy_guidance()