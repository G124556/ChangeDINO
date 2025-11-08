import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAwareModule(nn.Module):
    def __init__(self, text_dim=256):
        super().__init__()
        self.text_dim = text_dim

        # 视觉特征投影层
        self.vision_projections = nn.ModuleList([
            nn.Conv2d(64, text_dim, 1),  # 第一层
            nn.Conv2d(128, text_dim, 1),  # 第二层
            nn.Conv2d(256, text_dim, 1),  # 第三层
            nn.Conv2d(512, text_dim, 1)  # 第四层
        ])

        # 输出投影层
        self.out_projections = nn.ModuleList([
            nn.Conv2d(text_dim, 64, 1),  # 第一层
            nn.Conv2d(text_dim, 128, 1),  # 第二层
            nn.Conv2d(text_dim, 256, 1),  # 第三层
            nn.Conv2d(text_dim, 512, 1)  # 第四层
        ])

        # 语义生成器
        self.semantic_generator = nn.Sequential(
            nn.Conv2d(text_dim * 4, text_dim, 1),
            nn.BatchNorm2d(text_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(text_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * 1024)
        )

    def forward(self, vision_features, text_features):
        if isinstance(vision_features, tuple):
            vision_features = list(vision_features)
        if isinstance(text_features, tuple):
            text_features = list(text_features)

        batch_size = vision_features[0].shape[0]
        enhanced_features = []

        # 处理每个尺度的特征
        for i, (v_feat, t_feat, v_proj, out_proj) in enumerate(zip(
                vision_features, text_features, self.vision_projections,
                self.out_projections)):
            # 1. 调整文本特征到与视觉特征相同的空间尺寸
            _, _, H, W = v_feat.shape
            t_feat = F.interpolate(t_feat, size=(H, W), mode='bilinear', align_corners=False)

            # 2. 投影到相同维度
            v_proj_feat = v_proj(v_feat)  # B, C, H, W

            # 3. Flatten操作
            B, C, H, W = v_proj_feat.shape
            v_flat = v_proj_feat.view(B, C, -1)  # B, C, HW
            t_flat = t_feat.view(B, C, -1)  # B, C, HW

            # 4. 生成语义感知映射矩阵 (C×C)
            semantic_map = torch.matmul(v_flat, t_flat.transpose(-2, -1))  # B, C, C
            semantic_map = semantic_map / (self.text_dim ** 0.5)
            semantic_map = F.softmax(semantic_map, dim=-1)

            # 5. 应用语义映射
            enhanced = torch.matmul(semantic_map, t_flat)  # B, C, HW
            enhanced = enhanced.view(B, -1, H, W)

            # 6. 投影回原始维度并做残差连接
            enhanced = out_proj(enhanced)

            # 7. 残差连接
            enhanced_feat = enhanced + v_feat
            enhanced_features.append(enhanced_feat)

        # 生成语义信息
        aligned_text_features = []
        smallest_size = min(feat.shape[2:] for feat in text_features)

        for feat in text_features:
            if feat.shape[2:] != smallest_size:
                feat = F.adaptive_avg_pool2d(feat, smallest_size)
            aligned_text_features.append(feat)

        concat_features = torch.cat(aligned_text_features, dim=1)
        semantic_info = self.semantic_generator(concat_features)
        semantic_info = semantic_info.view(batch_size, 2, 1024)

        return semantic_info,enhanced_features


def tet_semantic_aware_module():
    # 创建模型
    model = SemanticAwareModule()
    model = model.cuda()

    # 创建测试数据
    vision_features = [
        torch.randn(16, 64, 64, 64).cuda(),
        torch.randn(16, 128, 32, 32).cuda(),
        torch.randn(16, 256, 16, 16).cuda(),
        torch.randn(16, 512, 8, 8).cuda()
    ]

    text_features = [
        torch.randn(16, 256, 32, 32).cuda(),
        torch.randn(16, 256, 16, 16).cuda(),
        torch.randn(16, 256, 8, 8).cuda(),
        torch.randn(16, 256, 4, 4).cuda()
    ]

    # 前向传播
    enhanced_features, semantic_info = model(vision_features, text_features)

    # 打印输出形状
    print("\nInput feature shapes:")
    for i, (v, t) in enumerate(zip(vision_features, text_features)):
        print(f"Layer {i + 1} - Vision: {v.shape}, Text: {t.shape}")

    print("\nEnhanced feature shapes:")
    for i, feat in enumerate(enhanced_features):
        print(f"Layer {i + 1}: {feat.shape}")
    print("\nSemantic info shape:", semantic_info.shape)

    # 检查输出维度
    assert enhanced_features[0].shape == (16, 64, 64, 64)
    assert enhanced_features[1].shape == (16, 128, 32, 32)
    assert enhanced_features[2].shape == (16, 256, 16, 16)
    assert enhanced_features[3].shape == (16, 512, 8, 8)
    assert semantic_info.shape == (16, 2, 1024)
    print("\nAll shape checks passed!")

    return enhanced_features, semantic_info


if __name__ == "__main__":
    enhanced_features, semantic_info = tet_semantic_aware_module()