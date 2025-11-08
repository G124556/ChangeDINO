import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class EntropyGuidance(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-9

    def compute_spatial_entropy(self, feat):
        """计算空间熵
        Args:
            feat: [B, C, H, W]
        Returns:
            entropy: [B, C, 1, 1]
        """
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1)  # [B, C, H*W]
        feat_norm = F.softmax(feat_flat, dim=2)
        entropy = -torch.sum(feat_norm * torch.log(feat_norm + self.eps), dim=2)
        return entropy.view(B, C, 1, 1)  # [B, C, 1, 1]

    def compute_mutual_information(self, vis_feat, text_feat):
        """计算对应通道的互信息
        Args:
            vis_feat: [B, C, H, W]
            text_feat: [B, C, H, W]
        Returns:
            mi: [B, C, 1, 1]
        """
        B, C, H, W = vis_feat.shape

        # 展平特征
        vis_flat = vis_feat.view(B, C, -1)  # [B, C, H*W]
        text_flat = text_feat.view(B, C, -1)  # [B, C, H*W]

        # 在空间维度上归一化
        vis_norm = F.softmax(vis_flat, dim=2)  # [B, C, H*W]
        text_norm = F.softmax(text_flat, dim=2)  # [B, C, H*W]

        # 计算channel维度的互信息
        joint = torch.bmm(vis_norm, text_norm.transpose(-1, -2))  # [B, C, H*W] x [B, H*W, C] -> [B, C, C]
        joint = joint.diagonal(dim1=1, dim2=2).unsqueeze(-1)  # 只取对角线元素 [B, C, 1]

        # 计算边缘分布
        p_vis = vis_norm.mean(dim=2, keepdim=True)  # [B, C, 1]
        p_text = text_norm.mean(dim=2, keepdim=True)  # [B, C, 1]

        # 计算互信息
        mi = torch.sum(joint * torch.log(joint / (p_vis * p_text + self.eps) + self.eps), dim=2)

        return mi.view(B, C, 1, 1)  # [B, C, 1, 1]

    def forward(self, vis_feats, text_feats):
        """特征引导
        Args:
            vis_feats: List of [B, C, H, W]
            text_feats: List of [B, C, H, W]
        """
        enhanced_feats = []

        for vis_feat, text_feat in zip(vis_feats, text_feats):
            # 1. 计算熵
            vis_entropy = self.compute_spatial_entropy(vis_feat)
            text_entropy = self.compute_spatial_entropy(text_feat)

            # 2. 计算互信息
            mi = self.compute_mutual_information(vis_feat, text_feat)

            # 3. 生成引导权重
            guide_weights = F.sigmoid((1 - text_entropy) + 0.5 * mi)

            # 4. 特征融合
            enhanced = vis_feat + guide_weights * text_feat
            enhanced_feats.append(enhanced)

        return enhanced_feats


def tst_entropy_guidance():
    # 生成测试特征
    batch_size = 16
    channels = 256
    scales = [(64, 64), (32, 32), (16, 16), (8, 8)]

    # 生成随机特征
    vis_feats = [torch.randn(batch_size, channels, h, w) for h, w in scales]
    text_feats = [torch.randn(batch_size, channels, h, w) for h, w in scales]

    # 创建模型
    model = EntropyGuidance()

    # 前向传播
    with torch.no_grad():
        enhanced_feats = model(vis_feats, text_feats)

    # 打印信息
    print("\n=== 特征统计信息 ===")
    for i, (vis_feat, text_feat, enhanced_feat) in enumerate(zip(vis_feats, text_feats, enhanced_feats)):
        print(f"\n尺度 {i}:")
        print(f"视觉特征范围: [{vis_feat.min():.3f}, {vis_feat.max():.3f}]")
        print(f"文本特征范围: [{text_feat.min():.3f}, {text_feat.max():.3f}]")
        print(f"增强特征范围: [{enhanced_feat.min():.3f}, {enhanced_feat.max():.3f}]")

    return enhanced_feats


if __name__ == "__main__":
    enhanced_feats = tst_entropy_guidance()