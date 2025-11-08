import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleInfoFusion(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256, 512]):
        super().__init__()
        self.in_channels_list = in_channels_list

        # 特征投影网络
        self.visual_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 1),
                nn.GroupNorm(8, c),
                nn.ReLU()
            ) for c in in_channels_list
        ])

        # 互信息估计器
        self.mi_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, 1),
                nn.ReLU(),
                nn.Conv2d(c, c, 1),
                nn.Sigmoid()
            ) for c in in_channels_list
        ])

        # 空间注意力
        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            ) for _ in in_channels_list
        ])

        # 通道注意力 - 修改为处理2D输入
        self.channel_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 空间维度压缩
                nn.Conv2d(c, c // 4, 1),
                nn.ReLU(),
                nn.Conv2d(c // 4, c, 1),
                nn.Sigmoid()
            ) for c in in_channels_list
        ])

    def compute_channel_entropy(self, x):
        B, C, H, W = x.shape
        # 保持2D特征图结构
        x_reshaped = x.view(B, C, -1)
        x_soft = F.softmax(x_reshaped, dim=2)
        entropy = -torch.sum(x_soft * torch.log(x_soft + 1e-10), dim=2)
        return entropy.view(B, C, 1, 1)  # 返回形状为 [B, C, 1, 1]

    def compute_spatial_entropy(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        x_soft = F.softmax(x_reshaped, dim=2)
        entropy = -torch.sum(x_soft * torch.log(x_soft + 1e-10), dim=2)
        return entropy.view(B, 1, H, W)

    def compute_mutual_info(self, x1, x2, mi_estimator):
        joint_feat = torch.cat([x1, x2], dim=1)
        mi_weights = mi_estimator(joint_feat)
        return mi_weights

    def forward(self, visual_feats, semantic_feats):
        """
        Args:
            visual_feats: List of tensors, each of shape [B, Ci, Hi, Wi]
            semantic_feats: List of tensors, each of shape [B, Ci, Hi, Wi]
        Returns:
            outputs: List of fused features at each scale
        """
        outputs = []

        for i, (v_feat, s_feat) in enumerate(zip(visual_feats, semantic_feats)):
            B, C, H, W = v_feat.shape

            # 1. 特征投影
            v_proj = self.visual_projs[i](v_feat)
            s_proj = self.visual_projs[i](s_feat)

            # 2. 通道注意力 - 直接使用2D卷积处理
            channel_weights = self.channel_attentions[i](v_proj)

            # 3. 空间注意力
            avg_pool = torch.mean(v_proj, dim=1, keepdim=True)
            max_pool, _ = torch.max(v_proj, dim=1, keepdim=True)
            spatial_feat = torch.cat([avg_pool, max_pool], dim=1)
            spatial_weights = self.spatial_attentions[i](spatial_feat)

            # 4. 互信息权重
            mi_weights = self.compute_mutual_info(v_proj, s_proj, self.mi_estimators[i])

            # 5. 特征融合
            channel_weighted = v_proj * channel_weights
            spatial_weighted = channel_weighted * spatial_weights
            fused_feat = spatial_weighted * mi_weights + v_feat

            outputs.append(fused_feat)

        return outputs


def tet_multi_scale_fusion():
    # 创建不同尺度的随机输入张量
    B = 16
    scales = [
        (64, 256, 256),  # C1, H1, W1
        (128, 128, 128),  # C2, H2, W2
        (256, 64, 64),  # C3, H3, W3
        (512, 32, 32)  # C4, H4, W4
    ]

    visual_feats = [torch.randn(B, c, h, w) for c, h, w in scales]
    semantic_feats = [torch.randn(B, c, h, w) for c, h, w in scales]

    # 初始化模块
    fusion_module = MultiScaleInfoFusion(in_channels_list=[c for c, _, _ in scales])

    # 前向传播
    outputs = fusion_module(visual_feats, semantic_feats)

    # 打印每个尺度的特征形状
    print("\nFeature shapes at each scale:")
    for i, output in enumerate(outputs):
        print(f"Scale {i + 1}: {output.shape}")

    # 验证输出维度是否与输入相同
    for i, (v_feat, out_feat) in enumerate(zip(visual_feats, outputs)):
        assert v_feat.shape == out_feat.shape, f"Scale {i + 1}: Output shape {out_feat.shape} doesn't match input shape {v_feat.shape}"

    print("\nAll feature shapes maintained correctly!")


if __name__ == "__main__":
    tet_multi_scale_fusion()