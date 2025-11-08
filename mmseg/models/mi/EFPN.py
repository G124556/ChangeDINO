import torch
import torch.nn as nn


class FPn(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256, 512], out_channels=256):
        super(FPn, self).__init__()

        # 横向连接层(lateral connections)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        # 自顶向下的路径的3x3卷积
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list))
        ])

        # 上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, features):
        """
        Args:
            features: 包含多尺度特征的列表
                    [
                        feature1: (16, 64, 64, 64),
                        feature2: (16, 128, 32, 32),
                        feature3: (16, 256, 16, 16),
                        feature4: (16, 512, 8, 8)
                    ]
        Returns:
            enhanced_features: 增强后的多尺度特征列表
        """

        # 自底向上的路径
        laterals = [
            lateral_conv(feature)
            for feature, lateral_conv in zip(features, self.lateral_convs)
        ]

        # 自顶向下的路径和特征融合
        enhanced_features = [laterals[-1]]  # 从最顶层开始
        for i in range(len(features) - 2, -1, -1):  # 从倒数第二层开始
            # 上采样高层特征
            top_down_feature = self.upsample(enhanced_features[0])
            # 特征融合
            enhanced = laterals[i] + top_down_feature
            # 3x3卷积精调特征
            enhanced = self.fpn_convs[i](enhanced)
            enhanced_features.insert(0, enhanced)

        return enhanced_features


# 使用示例
def tet_fpn():
    # 创建示例输入
    features = [
        torch.randn(16, 64, 64, 64),  # P2
        torch.randn(16, 128, 32, 32),  # P3
        torch.randn(16, 256, 16, 16),  # P4
        torch.randn(16, 512, 8, 8)  # P5
    ]

    # 初始化FPN
    fpn = FPn()

    # 前向传播
    enhanced_features = fpn(features)

    # 打印输出特征形状
    for i, feat in enumerate(enhanced_features):
        print(f"Enhanced feature {i + 2} shape:", feat.shape)


if __name__ == "__main__":
    tet_fpn()