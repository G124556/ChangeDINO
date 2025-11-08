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
        feat_norm = F.softmax(feat_flat, dim=2)  # 在空间维度做归一化
        entropy = -torch.sum(feat_norm * torch.log(feat_norm + self.eps), dim=2)
        return entropy.view(B, C, 1, 1)  # [B, C, 1, 1]

    def compute_mutual_information(self, vis_feat, text_feat):
        """计算对应通道的互信息
        Args:
            vis_feat: [B, C, H, W]
            text_feat: [B, C, H, W]
        """
        B, C, H, W = vis_feat.shape

        # 展平并归一化特征
        vis_flat = vis_feat.view(B, C, -1)  # [B, C, H*W]
        text_flat = text_feat.view(B, C, -1)  # [B, C, H*W]

        vis_norm = F.softmax(vis_flat, dim=2)
        text_norm = F.softmax(text_flat, dim=2)

        # 计算每个通道的互信息
        joint = torch.bmm(vis_norm, text_norm.transpose(1, 2))  # [B, C, C]

        # 计算边缘分布
        p_vis = vis_norm.mean(dim=2).unsqueeze(2)  # [B, C, 1]
        p_text = text_norm.mean(dim=2).unsqueeze(1)  # [B, 1, C]

        mutual_info = torch.sum(joint * torch.log(joint / (p_vis * p_text) + self.eps), dim=(1, 2))
        return mutual_info.view(B, C, 1, 1)  # [B, C, 1, 1]

    def forward(self, vis_feats, text_feats):
        """
        Args:
            vis_feats: List of [B, C, H, W] - backbone特征
            text_feats: List of [B, C, H, W] - DINO特征
        """
        enhanced_feats = []

        for vis_feat, text_feat in zip(vis_feats, text_feats):
            # 1. 计算空间熵
            vis_entropy = self.compute_spatial_entropy(vis_feat)  # [B, C, 1, 1]
            text_entropy = self.compute_spatial_entropy(text_feat)  # [B, C, 1, 1]

            # 2. 计算互信息
            mi = self.compute_mutual_information(vis_feat, text_feat)  # [B, C, 1, 1]

            # 3. 生成引导权重
            # 当DINO特征熵低（确定性高）且互信息高时，增强引导作用
            guide_weights = F.sigmoid((1 - text_entropy) + 0.5 * mi)  # [B, C, 1, 1]

            # 4. 特征融合
            enhanced = vis_feat + guide_weights * text_feat
            enhanced_feats.append(enhanced)

        return enhanced_feats


def generate_test_features():
    """生成测试用的随机特征"""
    torch.manual_seed(42)
    batch_size = 16
    channels = 256
    scales = [(64, 64), (32, 32), (16, 16), (8, 8)]

    vis_feats = []
    text_feats = []

    for h, w in scales:
        # 生成backbone特征
        vis_feat = torch.randn(batch_size, channels, h, w) * 0.1

        # 生成DINO特征，添加一些结构化的模式
        text_feat = torch.randn(batch_size, channels, h, w) * 0.1
        # 模拟DINO检测到的目标区域
        text_feat[:, :channels // 2, h // 4:h // 2, w // 4:w // 2] += 0.2

        vis_feats.append(vis_feat)
        text_feats.append(text_feat)

    return vis_feats, text_feats


def visualize_features(original, enhanced, entropy, mi, scale_idx=0):
    """可视化特征、熵和互信息"""
    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.imshow(original[0, 0].detach().cpu())
    plt.title('Original Feature')
    plt.colorbar()

    plt.subplot(142)
    plt.imshow(enhanced[0, 0].detach().cpu())
    plt.title('Enhanced Feature')
    plt.colorbar()

    plt.subplot(143)
    plt.imshow(entropy[0].squeeze().detach().cpu())
    plt.title('Entropy')
    plt.colorbar()

    plt.subplot(144)
    plt.imshow(mi[0].squeeze().detach().cpu())
    plt.title('Mutual Information')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def tet_entropy_guidance():
    # 生成测试特征
    vis_feats, text_feats = generate_test_features()

    # 创建模型
    model = EntropyGuidance()

    # 前向传播
    with torch.no_grad():
        enhanced_feats = model(vis_feats, text_feats)

    # 打印特征信息
    print("\n=== 特征统计信息 ===")
    for i, (vis_feat, text_feat, enhanced_feat) in enumerate(zip(vis_feats, text_feats, enhanced_feats)):
        print(f"\n尺度 {i}:")
        print(f"视觉特征范围: [{vis_feat.min():.3f}, {vis_feat.max():.3f}]")
        print(f"文本特征范围: [{text_feat.min():.3f}, {text_feat.max():.3f}]")
        print(f"增强特征范围: [{enhanced_feat.min():.3f}, {enhanced_feat.max():.3f}]")

        # 验证维度
        assert enhanced_feat.shape == vis_feat.shape, f"维度不匹配: {enhanced_feat.shape} vs {vis_feat.shape}"

    # 计算并可视化第一个尺度的熵和互信息
    entropy = model.compute_spatial_entropy(vis_feats[0])
    mi = model.compute_mutual_information(vis_feats[0], text_feats[0])
    visualize_features(vis_feats[0], enhanced_feats[0], entropy, mi)

    print("\n测试完成!")
    return enhanced_feats


if __name__ == "__main__":
    enhanced_feats = tet_entropy_guidance()