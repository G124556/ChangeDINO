import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleEntropyGuidance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.visual_proj = nn.ModuleList([
            nn.Conv2d(dim, dim, 1) for _ in range(4)
        ])
        self.text_proj = nn.Linear(dim, dim)

    def compute_entropy(self, x):
        """计算特征的信息熵"""
        if x.dim() == 4:  # 视觉特征 [B, D, H, W]
            B, D, H, W = x.shape
            x_flat = x.flatten(2)  # [B, D, H*W]
            x_norm = F.softmax(x_flat, dim=1)
            entropy = -torch.sum(x_norm * torch.log(x_norm + 1e-9), dim=1)  # [B, H*W]
            entropy = entropy.view(B, H, W)
        else:  # 文本特征 [B, Q, D]
            x_norm = F.softmax(x, dim=-1)
            entropy = -torch.sum(x_norm * torch.log(x_norm + 1e-9), dim=-1)  # [B, Q]

        return F.softmax(entropy, dim=-1)

    def forward(self, visual_feats, text_feat):
        enhanced_feats = []

        # 1. 计算文本特征的熵
        text_entropy = self.compute_entropy(text_feat)  # [B, Q]
        B = text_entropy.size(0)

        # 2. 将文本熵重塑为 [B, 1, Q, 1] 以便进行插值
        text_entropy = text_entropy.view(B, 1, -1, 1)

        # 3. 处理每个尺度的视觉特征
        for i, feat in enumerate(visual_feats):
            B, D, H, W = feat.shape

            # 计算视觉特征的熵
            vis_entropy = self.compute_entropy(feat)  # [B, H, W]

            # 将文本熵插值到正确的空间尺寸
            text_entropy_up = F.interpolate(
                text_entropy,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [B, H, W]

            # 计算总熵权重
            total_entropy = vis_entropy + text_entropy_up
            weights = F.softmax(total_entropy.view(B, -1), dim=1).view(B, H, W)

            # 应用权重
            weighted_feat = feat * weights.unsqueeze(1)
            enhanced = weighted_feat + feat
            enhanced_feats.append(enhanced)

        return enhanced_feats


def tet_entropy_guidance():
    # 设置随机种子
    torch.manual_seed(42)

    # 测试参数
    batch_size = 4
    dim = 256
    Q = 8

    # 创建测试数据
    visual_feats = [
        torch.randn(batch_size, dim, 64, 64),
        torch.randn(batch_size, dim, 32, 32),
        torch.randn(batch_size, dim, 16, 16),
        torch.randn(batch_size, dim, 8, 8)
    ]
    text_feat = torch.randn(batch_size, Q, dim)

    # 创建模型
    model = MultiScaleEntropyGuidance(dim)

    # 前向传播
    with torch.no_grad():
        enhanced_feats = model(visual_feats, text_feat)

    # 打印结果
    print("\n=== 输出检查 ===")
    for i, feat in enumerate(enhanced_feats):
        print(f"\nLevel {i}:")
        print(f"Shape: {feat.shape}")
        print(f"Value range: [{feat.min():.3f}, {feat.max():.3f}]")
        print(f"Mean: {feat.mean():.3f}")
        print(f"Std: {feat.std():.3f}")

        # 验证维度
        assert feat.shape == visual_feats[i].shape, f"Level {i} 维度不匹配"

    print("\n=== 所有测试通过! ===")

    return enhanced_feats


if __name__ == "__main__":
    enhanced_feats = tet_entropy_guidance()