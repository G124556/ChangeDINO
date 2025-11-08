# 在文件开头添加
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU

import torch
torch.backends.cudnn.benchmark = True     # 加速卷积运算
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class InfoGuidanceModule(nn.Module):
    """基于信息论的特征引导模块"""


    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels

        # 特征变换层
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 修改熵权重生成器的输入维度为2*channels
        self.entropy_weight = nn.Sequential(
            nn.Linear(2 * channels, channels),  # 修改输入维度
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        # 修改互信息权重生成器
        self.mi_weight = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )


    def compute_entropy(self, feat):
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1)
        feat_norm = F.softmax(feat_flat, dim=2)
        entropy = -torch.sum(feat_norm * torch.log(feat_norm + 1e-9), dim=2)
        return entropy

    # def compute_mutual_information(self, vis_feat, text_feat):
    #     B, C, H, W = vis_feat.shape
    #
    #     kernel_size = 4
    #     stride = 4
    #
    #     vis_stats = F.avg_pool2d(vis_feat, kernel_size=kernel_size, stride=stride)
    #     text_stats = F.avg_pool2d(text_feat, kernel_size=kernel_size, stride=stride)
    #
    #     vis_flat = vis_stats.view(B, C, -1)
    #     text_flat = text_stats.view(B, C, -1)
    #
    #     joint = torch.bmm(vis_flat, text_flat.transpose(-1, -2))
    #     joint = joint / joint.sum(dim=(1, 2), keepdim=True)
    #
    #     p_vis = joint.sum(dim=2, keepdim=True)
    #     p_text = joint.sum(dim=1, keepdim=True).transpose(-1, -2)
    #
    #     mi = torch.sum(
    #         joint * torch.log(joint / (p_vis * p_text + 1e-9) + 1e-9),
    #         dim=(1, 2)
    #     )
    #
    #     return mi

    def compute_mutual_information(self, vis_feat, text_feat):
        """计算通道间的互信息（修复维度问题）
        Args:
            vis_feat: [B, C, H, W]
            text_feat: [B, C, H, W]
        Returns:
            mi: [B, C]  # 确保输出维度正确
        """
        B, C, H, W = vis_feat.shape

        # 1. 将特征展平并标准化
        vis_flat = vis_feat.view(B, C, -1)  # [B, C, HW]
        text_flat = text_feat.view(B, C, -1)  # [B, C, HW]

        # 2. 对空间维度做softmax归一化
        vis_norm = F.softmax(vis_flat, dim=2)  # [B, C, HW]
        text_norm = F.softmax(text_flat, dim=2)  # [B, C, HW]

        # 3. 计算每个通道的互信息
        mi = torch.zeros(B, C).to(vis_feat.device)

        for b in range(B):
            for c in range(C):
                # 获取当前batch和channel的特征
                v = vis_norm[b, c]  # [HW]
                t = text_norm[b, c]  # [HW]

                # 计算联合分布
                joint = torch.outer(v, t)  # [HW, HW]
                joint = joint / joint.sum()

                # 计算边缘分布
                p_v = v.sum()
                p_t = t.sum()

                # 计算互信息
                eps = 1e-9
                mi[b, c] = torch.sum(
                    joint * torch.log(joint / (p_v * p_t + eps) + eps)
                )

        return mi  # [B, C]

    def forward(self, vis_feat, text_feat):
        B, C, H, W = vis_feat.shape

        # 特征变换
        vis_trans = self.transform(vis_feat)
        text_trans = self.transform(text_feat)

        # 计算熵
        vis_entropy = self.compute_entropy(vis_trans)  # [B, C]
        text_entropy = self.compute_entropy(text_trans)  # [B, C]

        # 拼接熵信息并生成权重
        entropy_concat = torch.cat([vis_entropy, text_entropy], dim=1)  # [B, 2C]
        entropy_weight = self.entropy_weight(entropy_concat)  # [B, C]

        # 计算互信息并生成权重
        mi = self.compute_mutual_information(vis_trans, text_trans)  # [B, C]
        mi_weight = self.mi_weight(mi)  # [B, C]

        # 将权重扩展为4D张量
        entropy_weight = entropy_weight.view(B, C, 1, 1)
        mi_weight = mi_weight.view(B, C, 1, 1)

        # 加权特征
        weighted_vis = vis_trans * entropy_weight * mi_weight
        weighted_text = text_trans * (1 - entropy_weight) * mi_weight

        # 特征融合
        fused_feat = self.fusion(torch.cat([weighted_vis, weighted_text], dim=1))

        return fused_feat
        # return vis_feat


class MultiScaleInfoGuidance(nn.Module):
    def __init__(self, channels=256):
        super().__init__()

        self.guidance_modules = nn.ModuleList([
            InfoGuidanceModule(channels) for _ in range(4)
        ])

        self.scale_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 8, channels, 1),
                nn.Sigmoid()
            ) for _ in range(4)
        ])

    def forward(self, vis_feats, text_feats):
        enhanced_feats = []

        for i, (vis_feat, text_feat, guidance, attention) in enumerate(
                zip(vis_feats, text_feats, self.guidance_modules, self.scale_attention)):
            # 信息引导融合
            fused = guidance(vis_feat, text_feat)

            # 尺度注意力
            scale_weight = attention(fused)
            enhanced = fused * scale_weight

            enhanced_feats.append(enhanced)

        return enhanced_feats


def tet_multi_scale_guidance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """测试多尺度信息引导网络"""
    # 设置随机种子
    torch.manual_seed(42)

    # 生成测试数据
    batch_size = 16
    channels = 256
    scales = [(64, 64), (32, 32), (16, 16), (8, 8)]

    # 生成视觉特征和文本特征
    vis_feats = []
    text_feats = []

    for h, w in scales:
        # 生成视觉特征
        vis_feat = torch.randn(batch_size, channels, h, w).to(device)
        # 添加一些结构化模式
        vis_feat[:, :channels // 4, h // 4:h // 2, w // 4:w // 2] += 1.0
        vis_feats.append(vis_feat)

        # 生成文本特征
        text_feat = torch.randn(batch_size, channels, h, w).to(device)
        # 添加一些语义相关的模式
        text_feat[:, channels // 4:channels // 2, h // 4:h // 2, w // 4:w // 2] += 1.0
        text_feats.append(text_feat)

    # 创建模型
    model = MultiScaleInfoGuidance(channels=channels).to(device)

    # 打印模型结构
    print("\n=== 模型结构 ===")
    print(model)

    # 前向传播
    with torch.no_grad():
        enhanced_feats = model(vis_feats, text_feats)

    # 打印特征统计信息
    print("\n=== 特征统计信息 ===")
    for i, (vis_feat, text_feat, enhanced_feat) in enumerate(zip(vis_feats, text_feats, enhanced_feats)):
        print(f"\n尺度 {scales[i]}:")
        print(f"视觉特征 - 范围: [{vis_feat.min():.3f}, {vis_feat.max():.3f}], 均值: {vis_feat.mean():.3f}")
        print(f"文本特征 - 范围: [{text_feat.min():.3f}, {text_feat.max():.3f}], 均值: {text_feat.mean():.3f}")
        print(f"增强特征 - 范围: [{enhanced_feat.min():.3f}, {enhanced_feat.max():.3f}], 均值: {enhanced_feat.mean():.3f}")

        # 验证维度
        assert enhanced_feat.shape == vis_feat.shape, f"维度不匹配: {enhanced_feat.shape} vs {vis_feat.shape}"

    # 可视化特征
    # visualize_features(vis_feats[0], text_feats[0], enhanced_feats[0])

    return enhanced_feats


# def visualize_features(vis_feat, text_feat, enhanced_feat):
#     """可视化特征"""
#     plt.figure(figsize=(15, 5))
#
#     # 视觉特征
#     plt.subplot(131)
#     plt.imshow(vis_feat[0].mean(dim=0).detach().cpu())
#     plt.title('Visual Feature')
#     plt.colorbar()
#
#     # 文本特征
#     plt.subplot(132)
#     plt.imshow(text_feat[0].mean(dim=0).detach().cpu())
#     plt.title('Text Feature')
#     plt.colorbar()
#
#     # 增强特征
#     plt.subplot(133)
#     plt.imshow(enhanced_feat[0].mean(dim=0).detach().cpu())
#     plt.title('Enhanced Feature')
#     plt.colorbar()
#
#     plt.tight_layout()
#     plt.show()


# 辅助分析函数
def analyze_feature_statistics(feat, name="Feature"):
    """分析特征统计信息"""
    with torch.no_grad():
        mean = feat.mean().cpu().item()
        std = feat.std().cpu().item()
        max_val = feat.max().cpu().item()
        min_val = feat.min().cpu().item()

        print(f"\n{name} 统计信息:")
        print(f"平均值: {mean:.3f}")
        print(f"标准差: {std:.3f}")
        print(f"最大值: {max_val:.3f}")
        print(f"最小值: {min_val:.3f}")


def analyze_entropy_mi(model, vis_feat, text_feat):
    """分析熵和互信息"""
    with torch.no_grad():
        # 计算熵
        vis_entropy = model.guidance_modules[0].compute_entropy(vis_feat)
        text_entropy = model.guidance_modules[0].compute_entropy(text_feat)

        # 计算互信息
        mi = model.guidance_modules[0].compute_mutual_information(vis_feat, text_feat)

        print("\n=== 信息统计 ===")
        print(f"视觉特征熵 - 均值: {vis_entropy.mean():.3f}")
        print(f"文本特征熵 - 均值: {text_entropy.mean():.3f}")
        print(f"互信息 - 均值: {mi.mean():.3f}")


if __name__ == "__main__":
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 运行测试
    enhanced_feats = tet_multi_scale_guidance()