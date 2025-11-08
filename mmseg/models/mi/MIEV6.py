import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class EnhancedFeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedFeatureFusion, self).__init__()
        self.in_channels = in_channels

        # 局部变化检测分支
        self.local_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.local_bn = nn.BatchNorm2d(in_channels // 2)

        # 全局上下文分支
        self.global_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        # 修正：global_feat经过mean操作后通道数变为1
        self.global_bn = nn.BatchNorm2d(in_channels // 2)  # 修改这里

        # 熵引导注意力
        self.entropy_attention = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 最终融合层
        self.fusion_conv = nn.Conv2d(in_channels // 2 + 1, in_channels, kernel_size=1)  # 修改输入通道数

    def calculate_entropy(self, x, kernel_size=3):
        b, c, h, w = x.size()
        x_normalized = F.softmax(x.view(b, c, -1), dim=2).view(b, c, h, w)
        entropy = -(x_normalized * torch.log(x_normalized + 1e-8)).sum(dim=1, keepdim=True)
        entropy_pooled = F.avg_pool2d(entropy, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return entropy_pooled

    def forward(self, x1, x2):
        # 1. 计算特征熵
        entropy1 = self.calculate_entropy(x1)
        entropy2 = self.calculate_entropy(x2)

        # 2. 生成熵注意力图
        entropy_cat = torch.cat([entropy1, entropy2], dim=1)
        attention_weights = self.entropy_attention(entropy_cat)

        # 3. 局部变化特征处理
        local_feat1 = self.local_conv(x1)
        local_feat2 = self.local_conv(x2)
        local_feat = torch.abs(local_feat1 - local_feat2)
        local_feat = self.local_bn(local_feat)

        # 4. 全局上下文特征处理
        global_feat1 = self.global_conv(x1)
        global_feat2 = self.global_conv(x2)
        # 修改全局特征处理方式
        global_feat = global_feat1 + global_feat2
        global_feat = self.global_bn(global_feat)
        global_feat = torch.mean(global_feat, dim=1, keepdim=True)

        # 5. 特征融合
        weighted_local = local_feat * attention_weights
        weighted_global = global_feat * (1 - attention_weights)

        fused_feat = torch.cat([weighted_local, weighted_global], dim=1)
        output = self.fusion_conv(fused_feat)

        return output, {'entropy1': entropy1,
                        'entropy2': entropy2,
                        'attention': attention_weights,
                        'local_feat': local_feat,
                        'global_feat': global_feat}


def generate_test_data(batch_size=2, channels=64, height=32, width=32):
    """生成测试数据"""
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    change_mask = torch.zeros(batch_size, 1, height, width)
    change_mask[:, :, height // 4:height // 2, width // 4:width // 2] = 1
    x2 = x2 * (1 + change_mask)
    return x1, x2


def visualize_features(features_dict, batch_idx=0):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    entropy1 = features_dict['entropy1'][batch_idx, 0].detach().cpu().numpy()
    plt.imshow(entropy1, cmap='jet')
    plt.title('Entropy Map 1')
    plt.colorbar()

    plt.subplot(132)
    attention = features_dict['attention'][batch_idx, 0].detach().cpu().numpy()
    plt.imshow(attention, cmap='jet')
    plt.title('Attention Weights')
    plt.colorbar()

    plt.subplot(133)
    local_feat = features_dict['local_feat'][batch_idx, 0].detach().cpu().numpy()
    plt.imshow(local_feat, cmap='jet')
    plt.title('Local Feature Response')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def tet_fusion_module():
    """测试特征融合模块"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    in_channels = 64
    model = EnhancedFeatureFusion(in_channels).to(device)
    print("Model initialized")

    x1, x2 = generate_test_data(batch_size=2, channels=in_channels)
    x1, x2 = x1.to(device), x2.to(device)
    print("Test data generated")

    print("\nInput features statistics:")
    print(f"x1 shape: {x1.shape}, mean: {x1.mean():.4f}, std: {x1.std():.4f}")
    print(f"x2 shape: {x2.shape}, mean: {x2.mean():.4f}, std: {x2.std():.4f}")

    print("\nRunning forward pass...")
    with torch.no_grad():
        output, features = model(x1, x2)

    print("\nOutput features statistics:")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean():.4f}")
    print(f"Output std: {output.std():.4f}")

    print("\nIntermediate features statistics:")
    print(f"Entropy1 shape: {features['entropy1'].shape}")
    print(f"Attention shape: {features['attention'].shape}")
    print(f"Local feature shape: {features['local_feat'].shape}")
    print(f"Global feature shape: {features['global_feat'].shape}")

    print("\nVisualizing features...")
    visualize_features(features)

    return model, output, features


if __name__ == "__main__":
    model, output, features = tet_fusion_module()
    print("\nTest completed successfully!")