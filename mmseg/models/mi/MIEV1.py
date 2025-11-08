import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SemanticGuidanceModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 特征转换层 - 将文本特征通道数调整为与视觉特征匹配
        self.text_transforms = nn.ModuleList([
            nn.Conv2d(256, 64, 1),  # 64x64
            nn.Conv2d(256, 128, 1),  # 32x32
            nn.Conv2d(256, 256, 1),  # 16x16
            nn.Conv2d(256, 512, 1)  # 8x8
        ])

        # 上采样前的特征调整
        self.pre_upsamples = nn.ModuleList([
            nn.Identity(),  # 64x64保持不变
            nn.Conv2d(256, 256, 1),  # 32x32
            nn.Conv2d(256, 256, 1),  # 16x16
            nn.Conv2d(256, 256, 1)  # 8x8
        ])

        # 高级语义生成
        self.semantic_generation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048)
        )

    def compute_entropy(self, feat):
        """计算空间熵"""
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1)
        feat_norm = F.softmax(feat_flat, dim=2)
        entropy = -torch.sum(feat_norm * torch.log(feat_norm + 1e-9), dim=2)
        return entropy.unsqueeze(-1).unsqueeze(-1)   # [B, C]

    # def compute_mutual_information(self, vis_feat, text_feat):
    #     """计算互信息（使用简化的相似度计算）"""
    #     B, C, H, W = vis_feat.shape
    #
    #     # 展平并归一化
    #     vis_flat = F.normalize(vis_feat.view(B, C, -1), dim=2)
    #     text_flat = F.normalize(text_feat.view(B, C, -1), dim=2)
    #
    #     # 计算相似度作为互信息的近似
    #     similarity = torch.sum(vis_flat * text_flat, dim=2)  # [B, C]
    #     mi = (similarity + 1) / 2
    #
    #     return mi  # [B, C]

    def compute_mutual_information(self, vis_feat, text_feat):
        """计算特征间的余弦相似度
        Args:
            vis_feat: [B, C, H, W]
            text_feat: [B, C, H, W]
        Returns:
            similarity: [B, C, 1]
        """
        B, C, H, W = vis_feat.shape

        # 1. 展平空间维度
        vis_flat = vis_feat.view(B, C, -1)  # [B, C, HW]
        text_flat = text_feat.view(B, C, -1)  # [B, C, HW]

        # 2. 计算点积
        dot_product = torch.sum(vis_flat * text_flat, dim=2)  # [B, C]

        # 3. 计算L2范数
        vis_norm = torch.norm(vis_flat, p=2, dim=2)  # [B, C]
        text_norm = torch.norm(text_flat, p=2, dim=2)  # [B, C]

        # 4. 计算余弦相似度
        cosine_sim = dot_product / (vis_norm * text_norm + 1e-9)  # [B, C]

        # 5. 添加维度并归一化到[0,1]范围
        return cosine_sim.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1]











    def feature_guidance(self, vis_feat, text_feat):
        """特征引导"""
        B, C, H, W = vis_feat.shape
        if text_feat.shape[-2:] != vis_feat.shape[-2:]:
            text_feat = F.interpolate(
                text_feat,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        # 1. 计算熵
        # vis_entropy = self.compute_entropy(vis_feat)  # [B, C]
        text_entropy = self.compute_entropy(text_feat)  # [B, C]

        entropy_mean = text_entropy.mean(dim=1, keepdim=True)  # [B, 1, 1, 1]
        entropy_std = text_entropy.std(dim=1, keepdim=True)   # [B, 1, 1, 1]
        normalized_entropy = (text_entropy - entropy_mean) / (entropy_std + 1e-9)
        text_entropy = torch.sigmoid(normalized_entropy)  # 映射到(0,1)





        # 2. 计算互信息
        mi = self.compute_mutual_information(vis_feat, text_feat)  # [B, C]

        # 3. 生成引导权重
        guide_weight = F.sigmoid(
            text_entropy * (1-mi)
        ).view(B, C, 1, 1)  # [B, C, 1, 1]

        # 4. 特征融合
        guided_feat = vis_feat + guide_weight * text_feat
        # guided_feat =(1-mi)*text_feat*text_entropy+mi*vis_feat*(1-text_entropy)
        # guided_feat =(1-mi)*text_feat*text_entropy+mi*vis_feat*(1-text_entropy)
        # guided_feat =mi*text_feat*text_entropy+(1-mi)*vis_feat*(1-text_entropy)

        return guided_feat

    # def forward(self, vis_feats, text_feats):
    #     """
    #     Args:
    #         vis_feats: List of [B, Ci, Hi, Wi]
    #         text_feats: List of [B, 256, Hi, Wi]
    #     Returns:
    #         guided_feats: List of [B, Ci, Hi, Wi]
    #         semantic_info: [B, 2, 1024]
    #     """
    #     B = vis_feats[0].shape[0]
    #     guided_feats = []
    #     text_features = []
    #
    #     # 1. 特征引导
    #     for i, (vis_feat, text_feat) in enumerate(zip(vis_feats, text_feats)):
    #         # 调整文本特征通道数
    #         text_transformed = self.text_transforms[i](text_feat)
    #
    #         # 特征引导
    #         guided_feat = self.feature_guidance(vis_feat, text_transformed)
    #         guided_feats.append(guided_feat)
    #
    #         # 收集文本特征用于生成语义信息
    #         text_features.append(self.pre_upsamples[i](text_feat))
    #
    #     # 2. 生成高级语义信息
    #     semantic_features = torch.cat(text_features, dim=1)  # [B, 256*4, H, W]
    #     semantic_vector = self.semantic_generation(semantic_features)  # [B, 2048]
    #     semantic_info = semantic_vector.view(B, 2, 1024)
    #
    #     return guided_feats, semantic_info
    def forward(self, vis_feats, text_feats):
        """
        Args:
            vis_feats: List of [B, Ci, Hi, Wi]
            text_feats: List of [B, 256, hi, wi]
        Returns:
            guided_feats: List of [B, Ci, Hi, Wi]
            semantic_info: [B, 2, 1024]
        """
        B = vis_feats[0].shape[0]
        guided_feats = []
        text_features = []

        # 1. 特征引导
        for i, (vis_feat, text_feat) in enumerate(zip(vis_feats, text_feats)):
            # 调整文本特征通道数
            text_transformed = self.text_transforms[i](text_feat)

            # 特征引导
            guided_feat = self.feature_guidance(vis_feat, text_transformed)
            guided_feats.append(guided_feat)

            # 收集文本特征用于生成语义信息
            processed_text = self.pre_upsamples[i](text_feat)

            # 将所有文本特征调整到相同的空间尺寸(例如16x16)
            processed_text = F.adaptive_avg_pool2d(processed_text, (16, 16))
            text_features.append(processed_text)

        # 2. 生成高级语义信息
        semantic_features = torch.cat(text_features, dim=1)  # [B, 256*4, 16, 16]
        semantic_vector = self.semantic_generation(semantic_features)  # [B, 2048]
        semantic_info = semantic_vector.view(B, 2, 1024)

        return  semantic_info,guided_feats


def tet_semantic_guidance():
    """测试模块"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 生成测试数据
    B = 16

    # 视觉特征
    vis_feats = [
        torch.randn(B, 64, 64, 64).to(device),
        torch.randn(B, 128, 32, 32).to(device),
        torch.randn(B, 256, 16, 16).to(device),
        torch.randn(B, 512, 8, 8).to(device)
    ]

    # 文本特征
    text_feats = [
        torch.randn(B, 256, 32, 32).to(device),
        torch.randn(B, 256, 16, 16).to(device),
        torch.randn(B, 256, 8, 8).to(device),
        torch.randn(B, 256, 4, 4).to(device)
    ]

    # 创建模型
    model = SemanticGuidanceModule().to(device)

    # 前向传播
    with torch.no_grad():
        semantic_info, guided_feats = model(vis_feats, text_feats)

    # 打印信息
    print("\n=== 输出信息 ===")
    print(f"语义信息形状: {semantic_info.shape}")

    for i, feat in enumerate(guided_feats):
        print(f"\n特征层级 {i}:")
        print(f"引导后特征形状: {feat.shape}")
        print(f"数值范围: [{feat.min():.3f}, {feat.max():.3f}]")
        print(f"均值: {feat.mean():.3f}")

    # 验证维度
    assert semantic_info.shape == (B, 2, 1024), f"语义信息维度错误: {semantic_info.shape}"
    assert len(guided_feats) == 4, "特征数量错误"
    assert guided_feats[0].shape == (B, 64, 64, 64), f"特征0维度错误: {guided_feats[0].shape}"
    assert guided_feats[1].shape == (B, 128, 32, 32), f"特征1维度错误: {guided_feats[1].shape}"
    assert guided_feats[2].shape == (B, 256, 16, 16), f"特征2维度错误: {guided_feats[2].shape}"
    assert guided_feats[3].shape == (B, 512, 8, 8), f"特征3维度错误: {guided_feats[3].shape}"

    print("\n测试通过!")
    return guided_feats, semantic_info


if __name__ == "__main__":
    guided_feats, semantic_info = tet_semantic_guidance()