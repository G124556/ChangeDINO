import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientFiLM(nn.Module):
    def __init__(self, location_dim=256):
        super().__init__()
        self.location_dim = location_dim

        # 为每个尺度的视觉特征处理
        self.visual_processors = nn.ModuleList([
            nn.Conv2d(64, 256, 1),  # Level 1
            nn.Conv2d(128, 256, 1),  # Level 2
            nn.Conv2d(256, 256, 1),  # Level 3
            nn.Conv2d(512, 256, 1)  # Level 4
        ])

        # 为每个尺度生成参数的网络
        self.param_generators = nn.ModuleList([
            self._build_generator(64),  # Level 1
            self._build_generator(128),  # Level 2
            self._build_generator(256),  # Level 3
            self._build_generator(512)  # Level 4
        ])

    def _build_generator(self, out_channels):
        return nn.Sequential(
            nn.Conv2d(512, 256, 1),  # 融合后的特征降维
            nn.ReLU(),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, out_channels * 2, 1)  # 生成gamma和beta
        )

    def process_location_features(self, location_features, target_size):
        B, N, D = location_features.shape

        # 简单的平均池化处理queries
        loc_feat = location_features.mean(dim=1)  # (B, D)

        # 扩展到空间维度
        loc_feat = loc_feat.view(B, D, 1, 1).expand(-1, -1, target_size, target_size)

        return loc_feat

    def forward(self, visual_features, location_features):
        modulated_features = []

        for idx, (visual_feat, processor, generator) in enumerate(zip(
                visual_features, self.visual_processors, self.param_generators)):
            B, C, H, W = visual_feat.shape

            # 处理视觉特征
            processed_visual = processor(visual_feat)  # (B, 256, H, W)

            # 处理并调整位置特征
            processed_location = self.process_location_features(location_features, H)  # (B, 256, H, W)

            # 拼接特征
            combined_feat = torch.cat([processed_visual, processed_location], dim=1)  # (B, 512, H, W)

            # 生成调制参数
            params = generator(combined_feat)  # (B, C*2, H, W)
            gamma, beta = params.chunk(2, dim=1)  # 各自(B, C, H, W)

            # 应用FiLM调制
            modulated = (1 + gamma) * visual_feat + beta
            modulated_features.append(modulated)

        return modulated_features


# 测试代码
if __name__ == "__main__":
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

    # 创建模型并测试
    model = EfficientFiLM(location_dim)
    modulated_features = model(visual_features, location_features)

    # 打印输出形状
    for idx, feat in enumerate(modulated_features):
        print(f"Level {idx + 1} output shape:", feat.shape)