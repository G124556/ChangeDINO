import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMMHAModule(nn.Module):
    def __init__(self, in_channels, location_dim=256):
        super().__init__()
        # FiLM参数生成
        self.W1 = nn.Conv2d(location_dim, 512, 1)
        self.W2 = nn.Conv2d(512, in_channels * 2, 1)

        # MHA部分
        self.q_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.k_proj = nn.Conv2d(location_dim, in_channels, 1)
        self.v_proj = nn.Conv2d(location_dim, in_channels, 1)

        self.mha = nn.MultiheadAttention(in_channels, 8)

    def forward(self, x, location_feat):
        # 生成FiLM参数
        film_params = self.W2(F.relu(self.W1(location_feat)))
        gamma, beta = film_params.chunk(2, dim=1)

        # FiLM调制
        x_mod = gamma * x + beta

        # MHA处理
        B, C, H, W = x_mod.shape
        q = self.q_proj(x_mod).flatten(2).transpose(1, 2)
        k = self.k_proj(location_feat).flatten(2).transpose(1, 2)
        v = self.v_proj(location_feat).flatten(2).transpose(1, 2)

        out, _ = self.mha(q, k, v)
        out = out.transpose(1, 2).view(B, C, H, W)+x

        return out


class MultiscaleFiLMMHA(nn.Module):
    def __init__(self, location_dim=256):
        super().__init__()
        # 为每个尺度创建FiLMMHA模块
        self.film_mha_modules = nn.ModuleList([
            FiLMMHAModule(64, location_dim),  # Level 1: 64x64
            FiLMMHAModule(128, location_dim),  # Level 2: 32x32
            FiLMMHAModule(256, location_dim),  # Level 3: 16x16
            FiLMMHAModule(512, location_dim)  # Level 4: 8x8
        ])

        # 位置特征处理
        self.location_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(location_dim, location_dim),
                nn.ReLU(),
                nn.Linear(location_dim, location_dim)
            ) for _ in range(4)
        ])



    def process_location_features(self, location_features, target_size):
        """
        Args:
            location_features: (B, 1360, 256)
            target_size: 目标空间尺寸(如64,32,16,8)
        """
        B, N, D = location_features.shape

        # 1. 先通过线性层调整特征
        linear = nn.Linear(D, target_size * target_size)
        loc_feat = linear(location_features)  # (B, 1360, target_size*target_size)

        # 2. 调整维度顺序
        loc_feat = loc_feat.transpose(1, 2)  # (B, target_size*target_size, 1360)

        # 3. 再通过一个线性层调整通道数
        channel_proj = nn.Linear(1360, D)
        loc_feat = channel_proj(loc_feat)  # (B, target_size*target_size, 256)

        # 4. 重塑到目标空间尺寸
        loc_feat = loc_feat.transpose(1, 2).view(B, D, target_size, target_size)

        return loc_feat

    def forward(self, visual_features, location_features):
        """
        Args:
            visual_features: List of tensors [(B,64,64,64), (B,128,32,32), (B,256,16,16), (B,512,8,8)]
            location_features: (B,1360,256)
        Returns:
            List of modulated and attended features
        """
        enhanced_features = []

        for idx, (visual_feat, film_mha_module, loc_processor) in enumerate(zip(
                visual_features, self.film_mha_modules, self.location_processors)):
            B, C, H, W = visual_feat.shape

            # 处理位置特征
            loc_feat = self.process_location_features(location_features, H)

            # 应用FiLM+MHA模块
            enhanced_feat = film_mha_module(visual_feat, loc_feat)
            enhanced_features.append(enhanced_feat)

        return enhanced_features


# 测试代码
if __name__ == "__main__":
    # 设置参数
    batch_size = 16
    location_dim = 256
    num_queries = 1360

    # 创建测试数据
    visual_features = [
        torch.randn(batch_size, 64, 64, 64),
        torch.randn(batch_size, 128, 32, 32),
        torch.randn(batch_size, 256, 16, 16),
        torch.randn(batch_size, 512, 8, 8)
    ]
    location_features = torch.randn(batch_size, num_queries, location_dim)

    # 创建模型
    model = MultiscaleFiLMMHA(location_dim)

    # 前向传播
    enhanced_features = model(visual_features, location_features)

    # 打印输出形状
    for idx, feat in enumerate(enhanced_features):
        print(f"Level {idx + 1} output shape:", feat.shape)