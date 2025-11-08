import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexibleFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        """
        Args:
            in_channels_list: 输入特征的通道数列表，如[64, 128, 256, 512]
            out_channels_list: 输出特征的通道数列表，如[32, 64, 128, 256]
        """
        super(FlexibleFPN, self).__init__()

        assert len(in_channels_list) == len(out_channels_list), \
            "输入和输出的特征层数必须相同"

        self.num_levels = len(in_channels_list)

        # 横向连接层 - 先将所有输入特征转换为最高层的输出通道数
        top_out_channels = out_channels_list[-1]
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, top_out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        # 最终输出调整层 - 将特征调整为期望的输出通道数
        self.output_convs = nn.ModuleList([
            nn.Conv2d(top_out_channels, out_channels, kernel_size=3, padding=1)
            for out_channels in out_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: 输入特征列表 [C2, C3, C4, C5]
        Returns:
            [P2, P3, P4, P5]: FPN输出特征
        """
        assert len(features) == self.num_levels, \
            f"需要 {self.num_levels} 层特征，但得到 {len(features)} 层"

        # 1. 横向连接转换（统一通道数）
        laterals = []
        for i, feat in enumerate(features):
            laterals.append(self.lateral_convs[i](feat))

        # 2. 自顶向下特征融合
        fpn_feats = [laterals[-1]]  # 最顶层特征
        for i in range(len(features) - 2, -1, -1):
            # 上采样
            top_down = F.interpolate(fpn_feats[0],
                                     size=laterals[i].shape[-2:],
                                     mode='nearest')
            # 特征融合 (现在通道数已经一致)
            fpn_feats.insert(0, laterals[i] + top_down)

        # 3. 调整到期望的输出通道数
        out = []
        for i, feat in enumerate(fpn_feats):
            out.append(self.output_convs[i](feat))

        return out


def tet_fpn():
    # 指定输入输出通道数
    in_channels = [64, 128, 256, 512]  # 输入通道数
    out_channels = [32, 64, 128, 256]  # 期望的输出通道数

    # 创建模型
    fpn = FlexibleFPN(in_channels, out_channels)

    # 创建测试数据
    batch_size = 2
    input_features = [
        torch.randn(batch_size, in_c, 64 // (2 ** i), 64 // (2 ** i))
        for i, in_c in enumerate(in_channels)
    ]

    # 前向传播
    output_features = fpn(input_features)

    # 打印特征图信息
    print("Input feature shapes:")
    for i, feat in enumerate(input_features):
        print(f"C{i + 2}: {feat.shape}")

    print("\nOutput feature shapes:")
    for i, feat in enumerate(output_features):
        print(f"P{i + 2}: {feat.shape}")


if __name__ == "__main__":
    tet_fpn()