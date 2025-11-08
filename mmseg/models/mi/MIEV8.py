import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
class ImportanceCalculator(nn.Module):
    """计算区域重要性的模块"""

    def __init__(self, channels_list):
        super().__init__()

        # 通道注意力
        self.channel_attns = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, c // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 4, c, 1),
                nn.Sigmoid()
            ) for c in channels_list
        ])

        # 空间注意力
        self.spatial_attns = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c // 8, 1),
                nn.BatchNorm2d(c // 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 8, 1, 1),
                nn.Sigmoid()
            ) for c in channels_list
        ])

        # 特征响应增强
        self.response_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, groups=c),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 1),
                nn.Sigmoid()
            ) for c in channels_list
        ])

    def forward(self, features):
        importance_maps = []

        for feat, c_attn, s_attn, r_enhance in zip(
                features, self.channel_attns, self.spatial_attns, self.response_enhance
        ):
            # 计算三种注意力
            # channel_imp = c_attn(feat)
            # spatial_imp = s_attn(feat)
            # response_imp = r_enhance(feat)
            #
            # # 融合三种重要性图
            # importance = channel_imp * spatial_imp * response_imp

            response_imp = r_enhance(feat)
            importance=response_imp
            importance_maps.append(importance)

        return importance_maps


class DynamicReceptiveField(nn.Module):
    """动态感受野模块"""

    def __init__(self, channels_list):
        super().__init__()

        # 多尺度感受野分支
        self.receptive_branches = nn.ModuleList([
            self._make_receptive_branch(c)
            for c in channels_list
        ])

        # 动态权重生成
        self.weight_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 3, 3, 1),
                nn.Softmax(dim=1)
            ) for c in channels_list
        ])

    def _make_receptive_branch(self, channels):
        return nn.ModuleList([
            # 小感受野
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ),
            # 中感受野
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ),
            # 大感受野
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=4, dilation=4, groups=channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        ])

    def forward(self, features, importance_maps):
        enhanced_features = []

        for feat, imp, branches, weight_gen in zip(
                features, importance_maps, self.receptive_branches, self.weight_generators
        ):
            # 多尺度特征提取
            multi_scale_feats = [branch(feat) for branch in branches]
            feats_cat = torch.cat(multi_scale_feats, dim=1)

            # 生成动态权重
            weights = weight_gen(feats_cat)  # [B, 3, 1, 1]

            # 加权融合
            weighted_feats = [
                w.unsqueeze(1) * f
                for w, f in zip(weights.chunk(3, dim=1), multi_scale_feats)
            ]
            enhanced = sum(weighted_feats)
            enhanced_features.append(enhanced)

        return enhanced_features


class FeatureEnhancement(nn.Module):
    """特征增强模块"""

    def __init__(self, vis_channels, text_channels):
        super().__init__()

        # 视觉特征增强
        self.vis_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1),
                nn.BatchNorm2d(c)
            ) for c in vis_channels
        ])

        # 文本特征转换
        self.text_transform = nn.ModuleList([
            nn.Conv2d(text_channels, c, 1)
            for c in vis_channels
        ])

        # 交互特征融合
        self.interaction = nn.ModuleList([
            self._make_interaction_block(c)
            for c in vis_channels
        ])

    def _make_interaction_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, vis_feats, text_feats):
        enhanced_feats = []

        for vis_feat, text_feat, v_enhance, t_trans, interact in zip(
                vis_feats, text_feats, self.vis_enhance,
                self.text_transform, self.interaction
        ):
            # 特征增强和转换
            enhanced_vis = v_enhance(vis_feat)
            transformed_text = t_trans(text_feat)

            # 特征交互
            fused = interact(torch.cat([enhanced_vis, transformed_text], dim=1))
            enhanced_feats.append(fused)

        return enhanced_feats


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, channels_list):
        super().__init__()

        # 特征转换
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ) for c in channels_list
        ])

        # 特征融合
        self.fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, 1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ) for c in channels_list[:-1]
        ])

    def forward(self, features):
        transformed = [t(f) for t, f in zip(self.transforms, features)]

        # 自顶向下的特征融合
        results = [transformed[-1]]
        for i in range(len(transformed) - 2, -1, -1):
            up_feat = F.interpolate(
                results[-1],
                size=transformed[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            fused = self.fusions[i](torch.cat([transformed[i], up_feat], dim=1))
            results.append(fused)

        return results[::-1]  # 翻转顺序以匹配输入顺序


class DynamicFeatureEnhancement(nn.Module):
    """主模块"""

    def __init__(self):
        super().__init__()

        self.channels_list = [64, 128, 256, 512]

        # 子模块
        self.importance_calculator = ImportanceCalculator(self.channels_list)
        self.dynamic_receptive = DynamicReceptiveField(self.channels_list)
        self.feature_enhancement = FeatureEnhancement(self.channels_list, 256)
        self.multi_scale_fusion = MultiScaleFusion(self.channels_list)

    def forward(self, vis_feats, text_feats):
        # 1. 计算重要性图
        importance_maps = self.importance_calculator(vis_feats)

        # 2. 动态感受野处理
        receptive_feats = self.dynamic_receptive(vis_feats, importance_maps)

        # 3. 特征增强
        enhanced_feats = self.feature_enhancement(receptive_feats, text_feats)

        # 4. 多尺度融合
        final_feats = self.multi_scale_fusion(enhanced_feats)

        return final_feats


def tet_dynamic_enhancement():
    """测试函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 生成测试数据
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
    model = DynamicFeatureEnhancement().to(device)

    # 前向传播
    with torch.no_grad():
        output_feats = model(vis_feats, text_feats)

    # 打印信息
    print("\n=== Output Information ===")
    for i, feat in enumerate(output_feats):
        print(f"\nFeature level {i}:")
        print(f"Shape: {feat.shape}")
        print(f"Value range: [{feat.min():.3f}, {feat.max():.3f}]")
        print(f"Mean: {feat.mean():.3f}")

    return output_feats


if __name__ == "__main__":
    output_feats = tet_dynamic_enhancement()