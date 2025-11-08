import torch
import torch.nn as nn
import torch.nn.functional as F
class EnhancedSemanticGuidanceModule(nn.Module):
    def __init__(self, fusion_stages=3):
        super().__init__()
        self.fusion_stages = fusion_stages

        # 多阶段特征转换
        self.text_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 1)
            ),  # 64x64
            nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1)
            ),  # 32x32
            nn.Sequential(
                nn.Conv2d(256, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 1)
            ),  # 16x16
            nn.Sequential(
                nn.Conv2d(256, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 1)
            )  # 8x8
        ])

        # 多阶段特征融合模块
        self.fusion_modules = nn.ModuleList([
            self._make_fusion_module(64),  # 64x64
            self._make_fusion_module(128),  # 32x32
            self._make_fusion_module(256),  # 16x16
            self._make_fusion_module(512)  # 8x8
        ])

        # 语义生成网络
        self.semantic_generation = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048)
        )

        # 跨尺度注意力
        self.cross_scale_attention = nn.ModuleList([
            self._make_cross_attention(prev_dim, curr_dim)
            for prev_dim, curr_dim in zip([64, 128+64, 256], [128+64, 256, 512])
        ])

    def _make_fusion_module(self, channels):
        return nn.ModuleList([
            # 空间注意力
            nn.Sequential(
                nn.Conv2d(channels * 2, channels // 2, 1),
                nn.BatchNorm2d(channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, 1, 1),
                nn.Sigmoid()
            ),
            # 通道注意力
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels * 2, channels // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, channels, 1),
                nn.Sigmoid()
            ),
            # 特征融合
            nn.Sequential(
                nn.Conv2d(channels * 4, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        ])

    def _make_cross_attention(self, prev_dim, curr_dim):
        return nn.Sequential(
            nn.Conv2d(prev_dim + curr_dim, curr_dim, 1),
            nn.BatchNorm2d(curr_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(curr_dim, curr_dim, 3, padding=1),
            nn.BatchNorm2d(curr_dim),
            nn.ReLU(inplace=True)
        )

    def compute_entropy_map(self, x):
        """计算特征图的局部熵"""
        b, c, h, w = x.shape
        # 使用unfold创建局部窗口
        patches = F.unfold(x, kernel_size=3, padding=1)
        patches = patches.view(b, c, 9, h, w)

        # 计算每个窗口的概率分布
        prob = F.softmax(patches, dim=2)
        entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=2)

        return entropy

    def progressive_fusion(self, vis_feat, text_feat, fusion_module):
        """渐进式特征融合"""

        B, C, H, W = vis_feat.shape
        if text_feat.shape[-2:] != vis_feat.shape[-2:]:
            text_feat = F.interpolate(
                text_feat,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )


        # 1. 计算空间注意力
        concat_feat = torch.cat([vis_feat, text_feat], dim=1)
        spatial_attention = fusion_module[0](concat_feat)

        # 2. 计算通道注意力
        channel_attention = fusion_module[1](concat_feat)

        # 3. 加权特征
        weighted_vis = vis_feat * spatial_attention * channel_attention
        weighted_text = text_feat * spatial_attention * channel_attention

        # 4. 特征融合
        fused_feat = fusion_module[2](torch.cat([weighted_vis, weighted_text, concat_feat], dim=1))

        return fused_feat

    def forward(self, vis_feats, text_feats):
        """
        Args:
            vis_feats: List of [B, Ci, Hi, Wi]
            text_feats: List of [B, 256, Hi, Wi]
        Returns:
            semantic_info: [B, 2, 1024]
            guided_feats: List of [B, Ci, Hi, Wi]
        """
        B = vis_feats[0].shape[0]
        guided_feats = []
        semantic_feats = []

        prev_fusion = None

        # 1. 多尺度特征融合
        for i, (vis_feat, text_feat) in enumerate(zip(vis_feats, text_feats)):
            # 转换文本特征
            text_transformed = self.text_transforms[i](text_feat)

            # 计算熵图
            entropy_map = self.compute_entropy_map(text_transformed)
            entropy_weighted_text = text_transformed * entropy_map

            # 特征融合
            fused = self.progressive_fusion(
                vis_feat,
                entropy_weighted_text,
                self.fusion_modules[i]
            )

            # 跨尺度注意力
            if prev_fusion is not None and i < len(self.cross_scale_attention):
                # 上采样前一层特征
                prev_up = F.interpolate(
                    prev_fusion,
                    size=fused.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                # 跨尺度特征融合
                fused = self.cross_scale_attention[i](
                    torch.cat([prev_up, fused], dim=1)
                )

            prev_fusion = fused
            guided_feats.append(fused)

            # 收集语义特征
            semantic_feat = F.adaptive_avg_pool2d(text_transformed, (16, 16))
            semantic_feats.append(semantic_feat)

        # 2. 生成语义信息
        semantic_feats = torch.cat(semantic_feats, dim=1)  # [B, C*4, 16, 16]
        semantic_vector = self.semantic_generation(semantic_feats)  # [B, 2048]
        semantic_info = semantic_vector.view(B, 2, 1024)

        return semantic_info, guided_feats


def test_enhanced_module():
    """测试增强型语义引导模块"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建测试数据
    batch_size = 16
    vis_feats = [
        torch.randn(batch_size, 64, 64, 64).to(device),
        torch.randn(batch_size, 128, 32, 32).to(device),
        torch.randn(batch_size, 256, 16, 16).to(device),
        torch.randn(batch_size, 512, 8, 8).to(device)
    ]

    text_feats = [
        torch.randn(batch_size, 256, 32, 32).to(device),
        torch.randn(batch_size, 256, 16, 16).to(device),
        torch.randn(batch_size, 256, 8, 8).to(device),
        torch.randn(batch_size, 256, 4, 4).to(device)
    ]

    # 创建模型
    model = EnhancedSemanticGuidanceModule().to(device)

    # 前向传播
    with torch.no_grad():
        semantic_info, guided_feats = model(vis_feats, text_feats)

    # 打印输出信息
    print("\n=== Output Information ===")
    print(f"Semantic info shape: {semantic_info.shape}")

    for i, feat in enumerate(guided_feats):
        print(f"\nFeature level {i}:")
        print(f"Shape: {feat.shape}")
        print(f"Value range: [{feat.min():.3f}, {feat.max():.3f}]")
        print(f"Mean: {feat.mean():.3f}")

    return semantic_info, guided_feats