import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class EnhancedSemanticGuidanceModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 方向敏感卷积核 - 检测不同方向的边缘
        self.direction_convs = nn.ModuleList([
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False)
            for _ in range(4)  # 4个方向：垂直、水平、对角线
        ])

        # 角点检测模块
        self.corner_detector = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1)
        )

        # 边缘增强模块
        self.edge_enhancer = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, groups=64),  # 深度可分离卷积
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1)
        )

        # 几何特征聚合
        self.geometry_fusion = nn.Sequential(
            nn.Conv2d(64 * 6, 256, 1),  # 4方向 + 角点 + 边缘
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 空洞卷积模块 - 高熵区域处理
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(256, 128, 3, padding=rate, dilation=rate)
            for rate in [1, 2, 4, 8]
        ])

        # 轻量级残差模块 - 低熵区域处理
        self.light_residual = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256)
        )

    def compute_block_entropy(self, feat, block_size=8):
        """计算分块熵"""
        B, C, H, W = feat.shape

        # 分块
        blocks = F.unfold(feat, kernel_size=block_size, stride=block_size)
        blocks = blocks.view(B, C, block_size * block_size, -1)

        # 计算熵
        prob = F.softmax(blocks, dim=2)
        entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=2)

        # 重塑为空间形状
        H_out, W_out = H // block_size, W // block_size
        entropy = entropy.view(B, C, H_out, W_out)

        # 标准化
        entropy = (entropy - entropy.mean(dim=1, keepdim=True)) / (entropy.std(dim=1, keepdim=True) + 1e-9)
        return torch.sigmoid(entropy)

    def detect_building_patterns(self, feat):
        """检测建筑特征模式"""
        # 1. 方向特征提取
        direction_feats = []
        for conv in self.direction_convs:
            direction_feats.append(conv(feat))

        # 2. 角点检测
        corner_feat = self.corner_detector(feat)

        # 3. 边缘增强
        edge_feat = self.edge_enhancer(feat)

        # 4. 特征聚合
        geometry_feats = torch.cat(direction_feats + [corner_feat, edge_feat], dim=1)
        building_feat = self.geometry_fusion(geometry_feats)

        return building_feat

    def enhance_features(self, feat, entropy_map):
        """根据熵图增强特征"""
        # 1. 空洞卷积处理 - 高熵区域
        dilated_feats = []
        for conv in self.dilated_convs:
            dilated_feats.append(conv(feat))
        high_entropy_feat = torch.cat(dilated_feats, dim=1)
        high_entropy_feat = self.geometry_fusion(high_entropy_feat)

        # 2. 轻量级处理 - 低熵区域
        low_entropy_feat = self.light_residual(feat)

        # 3. 加权融合
        enhanced_feat = entropy_map * high_entropy_feat + (1 - entropy_map) * low_entropy_feat

        return enhanced_feat

    def forward(self, feat):
        """前向传播"""
        # 1. 计算分块熵
        entropy_map = self.compute_block_entropy(feat)

        # 2. 检测建筑特征
        building_feat = self.detect_building_patterns(feat)

        # 3. 特征增强
        enhanced_feat = self.enhance_features(building_feat, entropy_map)

        return enhanced_feat, entropy_map, building_feat


def test_module():
    """测试函数"""
    # 创建模拟数据
    batch_size, channels = 16, 256
    H, W = 64, 64
    x = torch.randn(batch_size, channels, H, W).cuda()

    # 创建模块
    model = EnhancedSemanticGuidanceModule().cuda()

    # 前向传播
    with torch.no_grad():
        enhanced_feat, entropy_map, building_feat = model(x)

    # 打印形状信息
    print(f"Input shape: {x.shape}")
    print(f"Enhanced feature shape: {enhanced_feat.shape}")
    print(f"Entropy map shape: {entropy_map.shape}")
    print(f"Building feature shape: {building_feat.shape}")

    return enhanced_feat, entropy_map, building_feat


if __name__ == "__main__":
    enhanced_feat, entropy_map, building_feat = test_module()