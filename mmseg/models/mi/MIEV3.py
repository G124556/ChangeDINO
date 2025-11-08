import torch
import torch.nn as nn
import torch.nn.functional as F
class ChangeSemanticGuidanceModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. 语义增强网络：增强特征的语义表达
        self.semantic_enhance = nn.ModuleList([
            self._make_semantic_enhance(64),  # 64x64 - 细粒度纹理特征
            self._make_semantic_enhance(128),  # 32x32 - 中等结构特征
            self._make_semantic_enhance(256),  # 16x16 - 建筑轮廓特征
            self._make_semantic_enhance(512)  # 8x8 - 全局上下文特征
        ])

        # 2. 变化敏感模块：捕获时序变化特征
        self.change_sensitive = nn.ModuleList([
            self._make_change_module(64),
            self._make_change_module(128),
            self._make_change_module(256),
            self._make_change_module(512)
        ])

        # 3. 语义一致性模块：确保语义理解的一致性
        self.semantic_consistency = nn.ModuleList([
            self._make_consistency_module(64),
            self._make_consistency_module(128),
            self._make_consistency_module(256),
            self._make_consistency_module(512)
        ])

        # 4. 多尺度整合模块：整合不同尺度的变化信息
        self.scale_fusion = nn.ModuleList([
            self._make_scale_fusion(256, 64),  # 融合高层语义到低层特征
            self._make_scale_fusion(256, 128),
            self._make_scale_fusion(256, 256),
            self._make_scale_fusion(256, 512)
        ])

        # 5. 最终语义生成：生成高级语义表达
        self.semantic_generation = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048)
        )

    def _make_semantic_enhance(self, channels):
        """语义增强模块：增强特征的语义表达能力"""
        return nn.Sequential(
            # 空间注意力：关注空间上的重要区域
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            ),
            # 通道注意力：增强判别性通道
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid()
            ),
            # 特征重标定：调整特征分布
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        )

    def _make_change_module(self, channels):
        """变化敏感模块：捕获时序变化模式"""
        return nn.Sequential(
            # 时序差异学习
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 变化模式增强
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 变化特征提取
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def _make_consistency_module(self, channels):
        """语义一致性模块：确保特征语义一致性"""
        return nn.Sequential(
            # 语义投影
            nn.Conv2d(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            # 语义对齐
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            # 特征重构
            nn.Conv2d(channels // 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def _make_scale_fusion(self, text_channels, vis_channels):
        """多尺度特征整合模块"""
        return nn.Sequential(
            # 特征适配
            nn.Conv2d(text_channels + vis_channels, vis_channels, 1),
            nn.BatchNorm2d(vis_channels),
            nn.ReLU(inplace=True),
            # 特征融合
            nn.Conv2d(vis_channels, vis_channels, 3, padding=1),
            nn.BatchNorm2d(vis_channels),
            nn.ReLU(inplace=True)
        )

    def compute_semantic_entropy(self, feat):
        """计算语义熵：反映特征的语义不确定性"""
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1)
        # 归一化特征
        feat_norm = F.softmax(feat_flat, dim=2)
        # 计算熵
        entropy = -torch.sum(feat_norm * torch.log(feat_norm + 1e-9), dim=2)
        # 标准化
        entropy = (entropy - entropy.mean(dim=1, keepdim=True)) / (entropy.std(dim=1, keepdim=True) + 1e-9)
        return torch.sigmoid(entropy).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

    def compute_semantic_consistency(self, feat1, feat2):
        """计算语义一致性：评估特征的语义相似度"""
        B, C, H, W = feat1.shape

        # 1. 特征归一化
        feat1_flat = F.normalize(feat1.view(B, C, -1), dim=2)
        feat2_flat = F.normalize(feat2.view(B, C, -1), dim=2)

        # 2. 计算相似度
        similarity = torch.sum(feat1_flat * feat2_flat, dim=2)

        # 3. 标准化相似度
        similarity = similarity.unsqueeze(-1).unsqueeze(-1)
        return similarity  # [B, C, 1, 1]

    def enhance_features(self, vis_feat, text_feat, enhance_module):
        """特征增强：增强特征的语义表达"""
        # 1. 空间注意力
        spatial_attn = enhance_module[0](vis_feat)

        # 2. 通道注意力
        channel_attn = enhance_module[1](vis_feat)

        # 3. 特征增强
        enhanced_vis = enhance_module[2](vis_feat * spatial_attn * channel_attn)
        enhanced_text = enhance_module[2](text_feat * spatial_attn * channel_attn)

        return enhanced_vis, enhanced_text

    def forward(self, vis_feats, text_feats):
        """
        Args:
            vis_feats: List of [B, Ci, Hi, Wi]
            text_feats: List of [B, 256, hi, wi]
        Returns:
            semantic_info: [B, 2, 1024]
            guided_feats: List of [B, Ci, Hi, Wi]
        """
        B = vis_feats[0].shape[0]
        guided_feats = []
        semantic_feats = []

        # 1. 多尺度特征处理
        for i, (vis_feat, text_feat) in enumerate(zip(vis_feats, text_feats)):
            # 调整文本特征尺寸
            if text_feat.shape[-2:] != vis_feat.shape[-2:]:
                text_feat = F.interpolate(text_feat,
                                          size=vis_feat.shape[-2:],
                                          mode='bilinear',
                                          align_corners=False)

            # 1.1 特征增强
            enhanced_vis, enhanced_text = self.enhance_features(
                vis_feat, text_feat,
                self.semantic_enhance[i]
            )

            # 1.2 计算语义熵和一致性
            semantic_entropy = self.compute_semantic_entropy(enhanced_text)
            consistency = self.compute_semantic_consistency(enhanced_vis, enhanced_text)

            # 1.3 变化特征提取
            change_feat = self.change_sensitive[i](
                torch.cat([enhanced_vis, enhanced_text], dim=1)
            )

            # 1.4 特征一致性约束
            consistent_feat = self.semantic_consistency[i](change_feat)

            # 1.5 多尺度特征融合
            fused_feat = self.scale_fusion[i](
                torch.cat([consistent_feat, text_feat], dim=1)
            )

            # 1.6 最终特征生成
            final_feat = fused_feat * semantic_entropy * consistency

            guided_feats.append(final_feat)
            semantic_feats.append(F.adaptive_avg_pool2d(enhanced_text, (16, 16)))

        # 2. 生成高级语义信息
        semantic_feats = torch.cat(semantic_feats, dim=1)
        semantic_vector = self.semantic_generation(semantic_feats)
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
    model = ChangeSemanticGuidanceModule().to(device)

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